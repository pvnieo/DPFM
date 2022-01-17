import argparse
import yaml
import os
import torch
from shrec_partial_dataset import ShrecPartialDataset, shape_to_device
from model import DPFMNet


def eval_net(args, model_path, predictions_name):
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")

    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = os.path.join(base_path, cfg["dataset"]["cache_dir"])
    dataset_path = os.path.join(base_path, cfg["dataset"]["root_test"])

    # create dataset
    if cfg["dataset"]["name"] == "shrec16":
        test_dataset = ShrecPartialDataset(dataset_path, name=cfg["dataset"]["subset"], k_eig=cfg["fmap"]["k_eig"],
                                           n_fmap=cfg["fmap"]["n_fmap"], use_cache=True, op_cache_dir=op_cache_dir)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None, shuffle=True)
    else:
        raise NotImplementedError("dataset not implemented!")

    # define model
    dpfm_net = DPFMNet(cfg).to(device)
    dpfm_net.load_state_dict(torch.load(model_path, map_location=device))
    dpfm_net.eval()

    to_save_list = []
    for i, data in enumerate(test_loader):

        data = shape_to_device(data, device)

        # prepare iteration data
        C_gt = data["C_gt"].unsqueeze(0)
        gt_partiality_mask12, gt_partiality_mask21 = data["gt_partiality_mask12"], data["gt_partiality_mask21"]

        # do iteration
        C_pred, overlap_score12, overlap_score21, use_feat1, use_feat2 = dpfm_net(data)

        name1, name2 = data["shape1"]["name"], data["shape2"]["name"]
        to_save_list.append((name1, name2, C_pred.detach().cpu().squeeze(0), C_gt.detach().cpu().squeeze(0),
                             gt_partiality_mask12.detach().cpu().squeeze(0), gt_partiality_mask21.detach().cpu().squeeze(0)))

    torch.save(to_save_list, predictions_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the training of DPFM model.")

    parser.add_argument("--config", type=str, default="shrec16_cuts", help="Config file name")

    parser.add_argument("--model_path", type=str, help="path to saved model")
    parser.add_argument("--predictions_name", type=str, help="name of the prediction file")

    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"./config/{args.config}.yaml", "r"))
    eval_net(cfg, args.model_path, args.predictions_name)
