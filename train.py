import argparse
import math
import torch
import torchvision
from torchvision import transforms

import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule
from functools import partial

from utils import linear_warmup_cosine_decay, ssl_quantize
from vit import VisionTransformer


class LM(pl.LightningModule):
    def __init__(
        self,
        steps,
        learning_rate=1e-4,
        weight_decay=0.0001,
        image_size=32,
        num_classes=10,
        patch_size=4,
        dim=256,
        layers=12,
        heads=8,
        dropout_p=0.0,
        linear_warmup_ratio=0.05,
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
        **_,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps = steps
        self.linear_warmup_ratio = linear_warmup_ratio

        self.model = VisionTransformer(
            image_size=image_size,
            num_classes=num_classes,
            patch_size=patch_size,
            dim=dim,
            layers=layers,
            heads=heads,
            dropout_p=dropout_p,
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_accuracy = pl.metrics.Accuracy()

    def configure_optimizers(self):

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )

        warmup_steps = int(self.linear_warmup_ratio * self.steps)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, linear_warmup_cosine_decay(warmup_steps, self.steps)
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.val_accuracy(y_hat, y)
        return loss

    def validation_epoch_end(self, losses):
        self.log("valid_loss", torch.stack(losses).mean(), prog_bar=True)
        self.log("valid_acc", self.val_accuracy.compute(), prog_bar=True)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args(args)


def get_datamodule():
    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    dm = CIFAR10DataModule(data_dir="data", batch_size=args.batch_size, num_workers=8)
    dm.train_transforms = train_transforms
    dm.test_transforms = test_transforms
    dm.val_transforms = test_transforms
    return dm


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    pl.seed_everything(42)
    args = parse_args()
    dm = get_datamodule()
    
    # Add some dm attributes to args Namespace
    args.image_size = dm.size(-1)  # 32 for CIFAR
    args.num_classes = dm.num_classes  # 10 for CIFAR

    # compute total number of steps
    batch_size = args.batch_size * args.gpus if args.gpus > 0 else args.batch_size
    args.steps = dm.num_samples // batch_size * args.max_epochs
    
    # Init Lightning Module
    lm = LM(**vars(args))
    
    # Set up Trainer
    trainer = pl.Trainer.from_argparse_args(args)  # (max_epochs=args.maxepochs, gpus=args.gpus, precision=args.precision)
    
    # Train!
    trainer.fit(lm, dm)
