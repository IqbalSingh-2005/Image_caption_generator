import torch


def save_checkpoint(checkpoint, filename="my_checkpoint.pth.tar"):
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("step", 0)


def print_examples(model, dataset, device, writer=None, step=0):
    if len(dataset) == 0:
        return

    image, _ = dataset[0]
    caption = model.caption_image(image.to(device), dataset.vocab)
    text = " ".join(caption)
    print(f"Example caption: {text}")
    if writer is not None:
        writer.add_text("Example caption", text, global_step=step)