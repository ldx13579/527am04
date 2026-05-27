import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from datasets import load_dataset
from PIL import Image


class Flickr8kDataset(Dataset):
    def __init__(self, hf_dataset, transform, tokenizer, max_len=64):
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.images = []
        self.captions = []

        for item in hf_dataset:
            img = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            caps = item["caption"]
            if isinstance(caps, str):
                caps = [caps]
            for cap in caps:
                self.images.append(img)
                self.captions.append(cap)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        caption = self.captions[idx]
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return image, tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0)


class Flickr8kEvalDataset:
    def __init__(self, hf_dataset, transform, tokenizer, max_len=64):
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.images = []
        self.captions = []
        self.img2txt = {}
        self.txt2img = {}

        txt_idx = 0
        for img_idx, item in enumerate(hf_dataset):
            img = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            self.images.append(img)
            caps = item["caption"]
            if isinstance(caps, str):
                caps = [caps]
            self.img2txt[img_idx] = []
            for cap in caps:
                self.captions.append(cap)
                self.img2txt[img_idx].append(txt_idx)
                self.txt2img[txt_idx] = img_idx
                txt_idx += 1

    def get_image_tensors(self):
        return torch.stack([self.transform(img) for img in self.images])

    def get_text_tokens(self):
        tokens = self.tokenizer(
            self.captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return tokens["input_ids"], tokens["attention_mask"]


def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def load_flickr8k(config):
    print("Loading Flickr8k dataset...")
    ds = load_dataset("nlphuji/flickr_8k")

    if "test" in ds:
        train_split = ds["train"]
        test_split = ds["test"]
    else:
        full = ds["train"]
        split = full.train_test_split(test_size=0.125, seed=config.seed)
        train_split = split["train"]
        test_split = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    train_transform, val_transform = get_transforms(config.img_size)

    train_dataset = Flickr8kDataset(train_split, train_transform, tokenizer, config.max_text_len)
    eval_dataset = Flickr8kEvalDataset(test_split, val_transform, tokenizer, config.max_text_len)

    print(f"Train: {len(train_dataset)} pairs, Eval: {len(eval_dataset.images)} images / {len(eval_dataset.captions)} captions")
    return train_dataset, eval_dataset, tokenizer


def infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def get_train_loader(dataset, config):
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return infinite_loader(loader)
