'''
Utilities adapted from: https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning
'''

import os
import random
import pprint
import pdb
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import utils.logger as logging
from utils.parser import parse_args, load_config
from datasets import VideoAlignmentLoader
from losses import all_loss_otuprel_exp
from utils import get_model, get_optimizer, save_checkpoint


logger = logging.get_logger(__name__)


def main(cfg):
    random.seed(cfg.TCC.RANDOM_STATE)
    os.environ['PYTHONHASHSEED'] = str(cfg.TCC.RANDOM_STATE)
    np.random.seed(cfg.TCC.RANDOM_STATE)
    torch.manual_seed(cfg.TCC.RANDOM_STATE)

    # logs
    if cfg.LOG.DIR is not None:
        logging.setup_logging(
            output_dir=cfg.LOG.DIR,
            level=cfg.LOG.LEVEL.lower()
        )
        logger.critical(f"CONFIG:\n{pprint.pformat(cfg)}")
        writer = SummaryWriter(cfg.LOG.DIR)
    else:
        writer = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # data
    """
    It is recommended to keep num_workers as 0 as in this case (when we are
    using h5 files for loading the data and the way in which the data loader
    has been designed), when we use mulitple workers, the time taken is
    actually more as we are loading same amount of files in both the cases
    the overhead of handling multiple pocesses is more.
    """
    data_loader = DataLoader(
        VideoAlignmentLoader(cfg),
        batch_size=1,
        num_workers=0
    )
    scaler = GradScaler()

    # model
    model = get_model(cfg)
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)
    optimizer = get_optimizer(model, cfg)

    # train
    iter_i = 0
    old_loss = np.inf
    gradient_accumulations = 4
    while iter_i <= cfg.TCC.TRAIN_EPOCHS:
        iter_i += 1
        frames, steps, seq_lens = next(iter(data_loader))
        frames = frames.squeeze().permute(0, 1, 4, 2, 3).to(device)
        steps = steps.squeeze()
        seq_lens = seq_lens.squeeze()

        with autocast(enabled=False):
            embeddings = model(frames)
            # print(embeddings.shape)  # torch.Size([2, 32, 128])
            loss = all_loss_otuprel_exp(
                embeddings[0],
                embeddings[1],
                epoch = iter_i,
                norm_embeds= cfg.TCC.norm_train_embeds,
                c= cfg.TCC.c,
                b_laplace= cfg.TCC.b_laplace
            )
        loss = loss.to('cuda')
        scaler.scale(loss / gradient_accumulations).backward()
        if iter_i % gradient_accumulations == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        new_loss = loss

        if cfg.LOG.DIR is not None:
            if (iter_i % cfg.TCC.CHECKPOINT_FREQ) == 0 or \
                (old_loss > new_loss):
                if old_loss > new_loss:
                    logger.critical(f'Iter count {iter_i} Loss decreased from '
                                f'{old_loss} to {new_loss}. Saving the model.')
                    old_loss = new_loss
                state_dict = {
                    'iter_i': iter_i,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss/train': loss.item(),
                }
                filename = f'checkpoint_{iter_i:05}_loss-{loss.item():.4f}.pt'
                if new_loss<.25:
                    save_checkpoint(state_dict, cfg.LOG.DIR, filename)
            logger.critical(f'Loss/Train: {loss.item()} Iter: {iter_i}')
            # writer.add_scalar('Loss/Train', loss.item(), iter_i)
        if (iter_i % 10) == 0:
            logger.critical('Iter {} Loss {}'.format(
                iter_i,
                loss.item()
            ))


if __name__ == '__main__':
    main(load_config(parse_args()))
