import main
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    m = main.Model()
    trainer = pl.Trainer(
        gpus=0,
        accumulate_grad_batches=8,
        checkpoint_callback=ModelCheckpoint(),)
    trainer.fit(m)
    print("DONE!")