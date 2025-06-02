
from torch.utils.data import DataLoader

from datasets.ProceL import ProceL
from datasets.MECCANO import MECCANO
from datasets.CrossTask import CrossTask
from datasets.EPIC_Tents import EPIC_Tents
from datasets.PC_Assembly import PC_Assembly
from datasets.EGTEA_GazeP import EGTEA_GazeP, BaconAndEggs, Cheeseburger, ContinentalBreakfast, GreekSalad, PastaSalad, Pizza, TurkeySandwich, Salad_ego, Sandwich_ego, Brownie_ego, Pizza_ego, Eggs_ego
from datasets.CMU_Kitchens import CMU_Kitchens
from datasets.PC_Disassembly import PC_Disassembly


def get_loader(cfg, mode, transforms=None):
    assert mode in ['train', 'test', 'all', 'val']
    if cfg.DATA_LOADER.NAME == 'CMU_Kitchens':
        dataset = CMU_Kitchens(cfg, mode=mode, transforms=transforms)
    elif cfg.DATA_LOADER.NAME == 'EGTEAGAZEP':
        dataset = EGTEA_GazeP(cfg, mode=mode, transforms=transforms)
    elif cfg.DATA_LOADER.NAME == 'MECCANO':
        dataset = MECCANO(cfg, mode=mode, transforms=transforms)
    elif cfg.DATA_LOADER.NAME == 'EPIC-Tents':
        dataset = EPIC_Tents(cfg, mode=mode, transforms=transforms)
    elif cfg.DATA_LOADER.NAME == 'ProceL':
        dataset = ProceL(cfg, mode=mode, transforms=transforms)
    elif cfg.DATA_LOADER.NAME == 'CrossTask':
        dataset = CrossTask(cfg, mode=mode, transforms=transforms)
    elif cfg.DATA_LOADER.NAME == 'pc_assembly':
        dataset = PC_Assembly(cfg, mode=mode, transforms=transforms)
    elif cfg.DATA_LOADER.NAME == 'pc_disassembly':
        dataset = PC_Disassembly(cfg, mode=mode, transforms=transforms)

    elif cfg.DATA_LOADER.NAME == 'BaconAndEggs':
        dataset = BaconAndEggs(cfg, mode=mode, transforms=transforms)

    elif cfg.DATA_LOADER.NAME == 'Cheeseburger':
        dataset = Cheeseburger(cfg, mode=mode, transforms=transforms)

    elif cfg.DATA_LOADER.NAME == 'ContinentalBreakfast':
        dataset = ContinentalBreakfast(cfg, mode=mode, transforms=transforms)
    elif cfg.DATA_LOADER.NAME == 'Pizza':
        dataset = Pizza(cfg, mode=mode, transforms=transforms)
    elif cfg.DATA_LOADER.NAME == 'GreekSalad':
        dataset = GreekSalad(cfg, mode=mode, transforms=transforms)
    elif cfg.DATA_LOADER.NAME == 'TurkeySandwich':
        dataset = TurkeySandwich(cfg, mode=mode, transforms=transforms)
    elif cfg.DATA_LOADER.NAME == 'PastaSalad':
        dataset = PastaSalad(cfg, mode=mode, transforms=transforms)
    
    elif cfg.DATA_LOADER.NAME == 'Brownie_ego':
        dataset = Brownie_ego(cfg, mode=mode, transforms=transforms)
    elif cfg.DATA_LOADER.NAME == 'Eggs_ego':
        dataset = Eggs_ego(cfg, mode=mode, transforms=transforms)
    elif cfg.DATA_LOADER.NAME == 'Pizza_ego':
        dataset = Pizza_ego(cfg, mode=mode, transforms=transforms)
    elif cfg.DATA_LOADER.NAME == 'Salad_ego':
        dataset = Salad_ego(cfg, mode=mode, transforms=transforms)
    elif cfg.DATA_LOADER.NAME == 'Sandwich_ego':
        dataset = Sandwich_ego(cfg, mode=mode, transforms=transforms)

    else:
        raise NotImplementedError

    if mode == 'train' or mode == 'all':
        batch_size = cfg.TRAIN.BATCH_SIZE
    elif mode == 'test':
        batch_size = cfg.TEST.BATCH_SIZE
    elif mode == 'val':
        batch_size = cfg.VALIDATION.BATCH_SIZE

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=cfg.DATA_LOADER.SHUFFLE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY
    )

    return loader
