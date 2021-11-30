import os 
import time
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from utils.options import parse_args, load_param_dict
from utils.data_loader import WMDataset, split_train_val_test
from models import seq2seq, unet
from utils.evaluation import rmse_error, r2_error


def test(model, data_loader, num_test, device):
    
    model.eval()
    history, ground_truth, prediction = [], [], []
    
    with torch.no_grad():
        
        for it, data in enumerate(data_loader):
                
            batch_x = data[0].to(device)
            batch_y = data[1].to(device) 
            out = model(batch_x, batch_y) 
            history.append(batch_x.cpu().data.numpy())
            ground_truth.append(batch_y.cpu().data.numpy())
            prediction.append(out.cpu().data.numpy())
            
            print(it)
            if it == num_test - 1:
                break

    history = np.concatenate(history)
    ground_truth = np.concatenate(ground_truth)
    prediction = np.concatenate(prediction)
    
    rmse = rmse_error(ground_truth, prediction)
    r2 = r2_error(ground_truth, prediction)
    logging.info('TEST - RMSE = {:.6f}, R2 = {:.6f}'.format(rmse, r2))
    
    return history, ground_truth, prediction

    
def main():
    
    args = parse_args()
    params = load_param_dict(args, mode='test')
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(device)

    dataset = WMDataset(data_path=params['data_path'], 
                        sample_path=params['sample_path'],
                        region_id=params['region_id'], 
                        source_vars=['temperature'],
                        target_vars=['temperature'],
                        seq_len=params['seq_len'],
                        horizon=params['horizon'],
                        use_static=params['use_static'])
    
    train_dataset, _, test_dataset = split_train_val_test(dataset)
    
    test_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        
    if 'seq2seq' in params['model_name']:
        kernel_size = int(params['model_name'][params['model_name'].index('kernel') + 6])
        h_dim = int(params['model_name'][params['model_name'].index('hdim') + 4])
        model = seq2seq.EncoderDecoderConvLSTM(in_channels=len(dataset.source_vars),
                                               h_channels=h_dim, 
                                               out_channels=len(dataset.target_vars),
                                               kernel_size=(kernel_size, kernel_size)).to(device)
    elif 'unet' in params['model_name']:
        model = unet.UNet(in_channels=len(dataset.source_vars) * params['seq_len'],
                          out_channels=len(dataset.target_vars) * params['horizon'],).to(device)
    else:
        raise NotImplementedError

        
    model.load_state_dict(torch.load(params['model_path'], map_location=device)['state_dict'], strict=False)
    history, ground_truth, prediction = test(model, test_loader, num_test=3, device=device)
    
    np.savez_compressed(
        os.path.join(params['result_path'], 'prediction.npz'),
        history=v,
        ground_truth=ground_truth,
        prediction=prediction)

    
if __name__ == "__main__":
    main()
