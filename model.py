import os
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import skimage.transform
from PIL import Image
from collections import Counter
from tqdm.notebook import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu


dataset_coco = "./data/caption_datasets/dataset_flickr8k.json"


with open(dataset_coco, "r") as f:
    data = json.load(f)

max_len = 100
image_folder = "./img"
min_word_freq = 5
captions_per_image = 5
word_freq = Counter()

for img in data["images"]:
    for c in img["sentences"]:
        word_freq.update(c["tokens"])

words = [word for word in word_freq.keys() if word_freq[word] > min_word_freq]
word2id = {word: id for id, word in enumerate(words, 1)}
word2id["<unk>"] = len(word2id) + 1
word2id["<start>"] = len(word2id) + 1
word2id["<end>"] = len(word2id) + 1
word2id["<pad>"] = 0

id2word = {value: word for word, value in word2id.items()}


train_image_paths = []
train_image_captions = []
train_caption_lens = []

val_image_paths = []
val_image_captions = []
val_caption_lens = []

test_image_paths = []
test_image_captions = []
test_caption_lens = []

for img in data["images"]:
    captions = []
    for c in img["sentences"]:
        if len(c["tokens"]) <= max_len:
            captions.append(c["tokens"])

    if len(captions) == 0:
        continue

    if len(captions) < captions_per_image:
        captions += [
            random.choice(captions) for _ in range(captions_per_image - len(captions))
        ]
    else:
        captions = random.sample(captions, k=captions_per_image)

    assert len(captions) == captions_per_image

    enc_captions = []
    caption_lens = []
    for idx, caption in enumerate(captions):
        enc_caption = (
            [word2id["<start>"]]
            + [word2id.get(word, word2id["<unk>"]) for word in caption]
            + [word2id["<end>"]]
            + [word2id["<pad>"]] * (max_len - len(caption))
        )

        caption_len = len(caption) + 2

        enc_captions.append(enc_caption)
        caption_lens.append(caption_len)

    path = os.path.join(image_folder, img["filename"])

    if img["split"] in {"train", "restval"}:
        train_image_paths.append(path)
        train_image_captions.append(enc_captions)
        train_caption_lens.append(caption_lens)
    elif img["split"] in {"val"}:
        val_image_paths.append(path)
        val_image_captions.append(enc_captions)
        val_caption_lens.append(caption_lens)
    elif img["split"] in {"test"}:
        test_image_paths.append(path)
        test_image_captions.append(enc_captions)
        test_caption_lens.append(caption_lens)

assert len(train_image_paths) == len(train_image_captions)
assert len(val_image_paths) == len(val_image_captions)
assert len(test_image_paths) == len(test_image_captions)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CaptionDataset(Dataset):
    def __init__(
        self,
        image_paths,
        captions,
        caption_lens,
        split,
        captions_per_image=5,
        transform=None,
    ):
        self.split = split
        assert self.split in {"TRAIN", "VAL", "TEST"}

        self.image_paths = image_paths
        self.captions = captions
        self.caption_lens = caption_lens
        self.captions_per_image = captions_per_image

        self.transform = transform

    def __len__(self):
        return len(self.image_paths) * self.captions_per_image

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx // self.captions_per_image]).convert(
            "RGB"
        )
        img = np.array(img)
        img = cv2.resize(img, (256, 256))

        assert img.shape == (256, 256, 3)
        assert np.max(img) <= 255.0

        if self.transform is not None:
            img = self.transform(img)

        img = np.array(img)
        img = np.transpose(img, (2, 0, 1))
        img = torch.FloatTensor(img / 255.0)

        caption = torch.LongTensor(
            self.captions[idx // self.captions_per_image][idx % self.captions_per_image]
        )
        caption_len = torch.LongTensor(
            [
                self.caption_lens[idx // self.captions_per_image][
                    idx % self.captions_per_image
                ]
            ]
        )

        if self.split == "TRAIN":
            return img, caption, caption_len
        else:
            all_captions = torch.LongTensor(
                self.captions[idx // self.captions_per_image]
            )
            return img, caption, caption_len, all_captions


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
train_dataset = CaptionDataset(
    train_image_paths, train_image_captions, train_caption_lens, "TRAIN"
)
val_dataset = CaptionDataset(
    val_image_paths, val_image_captions, val_caption_lens, "VAL"
)
test_dataset = CaptionDataset(
    test_image_paths, test_image_captions, test_caption_lens, "TEST"
)


def decode_caption(enc_caption):
    dec_caption = [
        id2word[id]
        for id in caption.numpy()
        if id2word[id] not in ["<start>", "<end>", "<unk>", "<pad>"]
    ]
    return " ".join(dec_caption)


batch_size = 32
workers = 0

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
    pin_memory=True,
)

images, captions, caption_lens = next(iter(train_loader))


class ImageEncoder(nn.Module):
    def __init__(self, enc_image_size=14):
        super(ImageEncoder, self).__init__()
        self.enc_image_size = enc_image_size

        resnet = torchvision.models.resnet101(pretrained=True)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((enc_image_size, enc_image_size))

        self.fine_tune()

    def fine_tune(self, fine_tune=True):
        for param in self.resnet.parameters():
            param.requires_grad = False
        for child in list(self.resnet.children())[5:]:
            for param in child.parameters():
                param.requires_grad = fine_tune

    def forward(self, images):
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_attn = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attn = nn.Linear(decoder_dim, attention_dim)
        self.full_attn = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        attn1 = self.encoder_attn(encoder_out)
        attn2 = self.decoder_attn(decoder_hidden)
        attn = self.full_attn(F.relu(attn1 + attn2.unsqueeze(1)))

        alpha = F.softmax(attn, dim=1)
        attn_weighted_encoding = (encoder_out * alpha).sum(dim=1)
        alpha = alpha.squeeze(2)
        return attn_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        encoder_dim=2048,
        dropout=0.5,
    ):
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lens):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        caption_lens, sort_idx = caption_lens.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_idx]
        encoded_captions = encoded_captions[sort_idx]

        embeddings = self.embedding(encoded_captions)

        h, c = self.init_hidden_state(encoder_out)

        decode_lens = (caption_lens - 1).tolist()

        predictions = torch.zeros(batch_size, max(decode_lens), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lens), num_pixels).to(device)

        for t in range(max(decode_lens)):
            batch_size_t = sum([l > t for l in decode_lens])

            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t]
            )

            gate = F.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            h, c = self.decode_step(
                torch.cat(
                    [embeddings[:batch_size_t, t, :], attention_weighted_encoding],
                    dim=1,
                ),
                (h[:batch_size_t], c[:batch_size_t]),
            )

            preds = self.fc(self.dropout(h))

            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lens, alphas, sort_idx


def accuracy(scores, targets, k):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(
    epoch,
    epochs_since_improvement,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    bleu4,
    is_best,
):
    state = {
        "epoch": epoch,
        "epochs_since_improvement": epochs_since_improvement,
        "bleu-4": bleu4,
        "encoder": encoder,
        "decoder": decoder,
        "encoder_optimizer": encoder_optimizer,
        "decoder_optimizer": decoder_optimizer,
    }
    filename = "checkpoint_" + str(epoch) + ".pth.tar"
    torch.save(state, filename)

    if is_best:
        torch.save(state, "BEST_" + filename)


def train_epoch(
    train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer
):
    losses = []
    top5accs = []

    decoder.train()
    encoder.train()

    for i, (imgs, caps, cap_lens) in enumerate(
        tqdm(train_loader, total=len(train_loader))
    ):
        imgs = imgs.to(device)
        caps = caps.to(device)
        cap_lens = cap_lens.to(device)

        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
            imgs, caps, cap_lens
        )

        targets = caps_sorted[:, 1:]

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = (
            criterion(scores, targets)
            + alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        )

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        top5 = accuracy(scores, targets, 5)
        losses.append(loss.item())
        top5accs.append(top5)

    return np.mean(losses), np.mean(top5accs)


def val_epoch(val_loader, encoder, decoder, criterion):
    losses = []
    top5accs = []

    decoder.eval()
    if encoder is not None:
        encoder.eval()

    references = list()
    hypotheses = list()

    with torch.no_grad():
        for i, (imgs, caps, cap_lens, all_caps) in enumerate(
            tqdm(val_loader, total=len(val_loader))
        ):
            imgs = imgs.to(device)
            caps = caps.to(device)
            cap_lens = cap_lens.to(device)

            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                imgs, caps, cap_lens
            )
            sort_ind = sort_ind.to(all_caps.device)

            targets = caps_sorted[:, 1:]

            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True
            ).data

            loss = (
                criterion(scores, targets)
                + alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
            )

            top5 = accuracy(scores, targets, 5)
            losses.append(loss.item())
            top5accs.append(top5)

            all_caps = all_caps[sort_ind]
            all_caps = all_caps.to(device)

            for j in range(all_caps.shape[0]):
                img_caps = all_caps[j].tolist()
                img_captions = list(
                    map(
                        lambda caption: [
                            word
                            for word in caption
                            if word not in {word2id["<start>"], word2id["<pad>"]}
                        ],
                        img_caps,
                    )
                )
                references.append(img_captions)

            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, pred in enumerate(preds):
                temp_preds.append(preds[j][: decode_lengths[j]])
            hypotheses.extend(temp_preds)

            assert len(references) == len(hypotheses)

        bleu4 = corpus_bleu(references, hypotheses)

    return np.mean(losses), np.mean(top5accs), bleu4


embed_dim = 512
attention_dim = 512
decoder_dim = 512
encoder_dim = 2048
encoder_lr = 1e-4
decoder_lr = 4e-4
grad_clip = 5.0
alpha_c = 1.0
vocab_size = len(word2id)
best_bleu4 = 0.0
lr_decay_factor = 0.8
lr_decay_patience = 8
best_bleu4 = 0

start_epoch = 1
num_epochs = 20
epochs_since_improvement = 0

fine_tune_encoder = False
checkpoint = None
cudnn.benchmark = True


if checkpoint is None:
    decoder = DecoderWithAttention(
        attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim
    )
    decoder_optimizer = optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr
    )

    encoder = ImageEncoder()
    encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = (
        optim.Adam(
            params=filter(lambda p: p.requires_grad, encoder.parameters()),
            lr=encoder_lr,
        )
        if fine_tune_encoder
        else None
    )

else:
    checkpoint = torch.load(checkpoint)

    start_epoch = checkpoint["epoch"] + 1
    best_bleu4 = checkpoint["bleu-4"]
    encoder = checkpoint["encoder"]
    decoder = checkpoint["decoder"]
    encoder_optimizer = checkpoint["encoder_optimizer"]
    decoder_optimizer = checkpoint["decoder_optimizer"]

    if fine_tune_encoder is True and encoder_optimizer is None:
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad, encoder.parameters()),
            lr=encoder_lr,
        )


encoder = encoder.to(device)
decoder = decoder.to(device)

encoder_lr_scheduler = (
    optim.lr_scheduler.ReduceLROnPlateau(
        encoder_optimizer,
        mode="max",
        factor=lr_decay_factor,
        patience=lr_decay_patience,
    )
    if fine_tune_encoder
    else None
)
decoder_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    decoder_optimizer, mode="max", factor=lr_decay_factor, patience=lr_decay_patience
)

criterion = nn.CrossEntropyLoss().to(device)

for epoch in range(start_epoch, num_epochs + 1):
    loss_train, acc_train = train_epoch(
        train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer
    )
    loss_val, acc_val, bleu4_val = val_epoch(val_loader, encoder, decoder, criterion)

    decoder_lr_scheduler.step(bleu4_val)
    if fine_tune_encoder:
        encoder_lr_scheduler.step(bleu4_val)

    is_best = bleu4_val > best_bleu4
    best_bleu4 = max(bleu4_val, best_bleu4)
    if not is_best:
        epochs_since_improvement += 1
    else:
        epochs_since_improvement = 0

    print("-" * 40)
    print(
        f"epoch: {epoch}, train loss: {loss_train:.4f}, train acc: {acc_train:.2f}%, valid loss: {loss_val:.4f}, valid acc: {acc_val:.2f}%, best BLEU-4: {best_bleu4:.4f}"
    )
    print("-" * 40)

    save_checkpoint(
        epoch,
        epochs_since_improvement,
        encoder,
        decoder,
        encoder_optimizer,
        decoder_optimizer,
        bleu4_val,
        is_best,
    )


def generate_image_caption(encoder, decoder, image_path, word_map, beam_size=5):
    k = beam_size

    rev_word_map = {id: word for word, id in word_map.items()}

    img = np.array(Image.open(image_path).convert("RGB"))
    img = np.array(Image.open(image_path).convert("RGB"))
    img = cv2.resize(img, (256, 256))

    assert img.shape == (256, 256, 3)
    assert np.max(img) <= 255

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = transform(img)

    encoder_out = encoder(img.unsqueeze(0).to(device))
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)

    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

    top_k_prev_words = torch.tensor([[word_map["<start>"]]] * k, dtype=torch.long).to(
        device
    )

    top_k_seqs = top_k_prev_words

    top_k_scores = torch.zeros(k, 1).to(device)

    top_k_seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)

    complete_seqs = []
    complete_seqs_alpha = []
    complete_seqs_scores = []

    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    while True:
        embeddings = decoder.embedding(top_k_prev_words).squeeze(1)

        attention_weighted_encoding, alpha = decoder.attention(encoder_out, h)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)

        gate = F.sigmoid(decoder.f_beta(h))
        attention_weighted_encoding = gate * attention_weighted_encoding

        h, c = decoder.decode_step(
            torch.cat([embeddings, attention_weighted_encoding], dim=1), (h, c)
        )

        scores = decoder.fc(h)
        scores = F.log_softmax(scores, dim=1)

        scores = top_k_scores.expand_as(scores) + scores

        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        prev_word_inds = top_k_words // vocab_size
        next_word_inds = top_k_words % vocab_size

        top_k_seqs = torch.cat(
            [top_k_seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1
        )
        top_k_seqs_alpha = torch.cat(
            [top_k_seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
            dim=1,
        )

        incomplete_inds = [
            ind
            for ind, next_word in enumerate(next_word_inds)
            if next_word != word_map["<end>"]
        ]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            complete_seqs.extend(top_k_seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(top_k_seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)

        if k == 0:
            break

        top_k_seqs = top_k_seqs[incomplete_inds]
        top_k_seqs_alpha = top_k_seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        top_k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    caption = [rev_word_map[ind] for ind in seq]

    img = Image.open(image_path).convert("RGB")
    img = img.resize([14 * 24, 14 * 24], Image.LANCZOS)

    fig = plt.figure(figsize=(20, 8))
    for t in range(len(caption)):
        plt.subplot(int(np.ceil(len(caption) / 5.0)), 5, t + 1)
        plt.text(
            0,
            1,
            "%s" % (caption[t]),
            color="black",
            backgroundcolor="white",
            fontsize=12,
        )
        plt.imshow(img)

        current_alpha = alphas[t]
        current_alpha = np.array(current_alpha)
        alpha = skimage.transform.pyramid_expand(current_alpha, upscale=24, sigma=8)
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)

        plt.set_cmap("gray")
        plt.axis("off")

    plt.show()


image_url = "https://cdn.analyticsvidhya.com/wp-content/uploads/2020/10/2398605966_1d0c9e6a20-300x200.jpg"
beam_size = 5

checkpoint = torch.load("./BEST_checkpoint_13.pth.tar", map_location=device)

encoder = checkpoint["encoder"]
decoder = checkpoint["decoder"]

vocab_size = len(word2id)
image_path = "./data/val2014/COCO_val2014_000000000397.jpg"

# from urllib.request import urlretrieve
# urlretrieve(image_url, image_path)

generate_image_caption(encoder, decoder, image_path, word2id, beam_size)
