import pathlib
import sys
from datetime import datetime

import hydra
import omegaconf
import torch
from hydra_slayer import Registry
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append("../../")

import src
import src.datasets
import src.models


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    load_dir = pathlib.Path(cfg.data.load_dir)

    train_data = src.datasets.DefaultDataset(annotation_file=load_dir / "train.csv")
    valid_data = src.datasets.DefaultDataset(annotation_file=load_dir / "valid.csv")

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=64, shuffle=True)

    cfg_dct = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    registry = Registry()
    registry.add_from_module(src.models, prefix="src.models.")

    model = registry.get_from_params(**cfg_dct["model"])
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.0
        last_loss = 0.0

        for i, data in enumerate(train_dataloader):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                last_loss = running_loss / 1000
                logger.info(f"  batch {i + 1} loss: {last_loss}")
                tb_x = epoch_index * len(train_dataloader) + i + 1
                tb_writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.0
        return last_loss

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"runs/trainer_{timestamp}")
    epoch_number = 0
    best_vloss = 1000000.0
    for epoch in range(cfg.training_params.num_epochs):
        logger.info(f"EPOCH {epoch_number + 1}:")

        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)
        running_vloss = 0.0

        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(valid_dataloader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        logger.info(f"LOSS train {avg_loss} valid {avg_vloss}")

        writer.add_scalars(
            "Training vs. Validation Loss", {"Training": avg_loss, "Validation": avg_vloss}, epoch_number + 1
        )
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f"model_{timestamp}_{epoch_number}"
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


if __name__ == "__main__":
    main()
