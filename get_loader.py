from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class Vocabulary:
	def __init__(self, frequency_threshold=5):
		self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
		self.stoi = {token: index for index, token in self.itos.items()}
		self.frequency_threshold = frequency_threshold

	def __len__(self):
		return len(self.itos)

	def tokenizer(self, text):
		return text.lower().split()

	def build_vocabulary(self, sentences):
		frequencies = Counter()
		for sentence in sentences:
			frequencies.update(self.tokenizer(sentence))

		for word, frequency in frequencies.items():
			if frequency >= self.frequency_threshold and word not in self.stoi:
				index = len(self.itos)
				self.stoi[word] = index
				self.itos[index] = word

	def numericalize(self, text):
		return [
			self.stoi.get(token, self.stoi["<UNK>"])
			for token in self.tokenizer(text)
		]


class Flickr8kDataset(Dataset):
	def __init__(self, root_folder, annotation_file, transform=None, frequency_threshold=5):
		self.root_folder = Path(root_folder)
		self.transform = transform
		self.images = []
		self.captions = []

		with open(annotation_file, encoding="utf-8") as file:
			next(file, None)
			for line in file:
				image_name, caption = line.rstrip("\n").split(",", 1)
				image_name = image_name.split("#")[0]
				if image_name.rsplit(".", 1)[-1].isdigit():
					image_name = image_name.rsplit(".", 1)[0]
				if not (self.root_folder / image_name).is_file():
					continue
				self.images.append(image_name)
				self.captions.append(caption)

		self.vocab = Vocabulary(frequency_threshold)
		self.vocab.build_vocabulary(self.captions)

	def __len__(self):
		return len(self.captions)

	def __getitem__(self, index):
		image = Image.open(self.root_folder / self.images[index]).convert("RGB")
		if self.transform:
			image = self.transform(image)

		tokens = [
			self.vocab.stoi["<SOS>"],
			*self.vocab.numericalize(self.captions[index]),
			self.vocab.stoi["<EOS>"],
		]
		return image, torch.tensor(tokens, dtype=torch.long)


class CaptionCollate:
	def __init__(self, pad_index):
		self.pad_index = pad_index

	def __call__(self, batch):
		images, captions = zip(*batch)
		images = torch.stack(images)
		captions = pad_sequence(captions, batch_first=False, padding_value=self.pad_index)
		return images, captions


def get_loader(
	root_folder,
	annotation_file,
	transform=None,
	batch_size=32,
	num_workers=2,
	shuffle=True,
):
	dataset = Flickr8kDataset(root_folder, annotation_file, transform=transform)
	pad_index = dataset.vocab.stoi["<PAD>"]
	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=CaptionCollate(pad_index),
		pin_memory=True,
		persistent_workers=num_workers > 0,
		prefetch_factor=4 if num_workers > 0 else None,
	)
	return loader, dataset
