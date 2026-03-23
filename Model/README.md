Model code and training

Contents:
- `Models_fullnew.py`: model architectures used for interface prediction.
- `Dataloader_itf.py`, `Dataloader_pred.py`: dataset and dataloader implementations.
- `run_model.py`: script for training and/or inference.
- `isicisiParamodelnewesm_l1_g10_i5_do0.30_dpr0.25_lr0.0001_fold10.pth`: example pretrained weights.
- `PT_MG_TV_reg.py`, `Dynamic_Mask.py`: training utilities and regularization helpers.

Usage:
- Check `run_model.py` for available command-line options. Ensure dependencies from `data_preprocess/requirements.txt` or the repo's `requirements.txt` are installed.
