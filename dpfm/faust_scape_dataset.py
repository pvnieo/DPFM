import os
# import sys
from itertools import permutations, combinations
import numpy as np

import torch
from torch.utils.data import Dataset

import potpourri3d as pp3d
# from utils import normalize_area_scale
from utils import farthest_point_sample, square_distance, normalize_area_scale

# sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net


class FaustScapeDataset(Dataset):
    def __init__(self, root_dir, name="faust", train=True, k_eig=128, n_fmap=30, use_cache=True, op_cache_dir=None):

        self.train = train  # bool
        self.k_eig = k_eig
        self.n_fmap = n_fmap
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, name, "cache")
        self.op_cache_dir = op_cache_dir

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.vts_list = []
        self.names_list = []
        self.sample_list = []

        # set combinations
        n_train = 80 if name == "faust" else 51
        if self.train:
            self.combinations = list(permutations(range(n_train), 2))
        else:
            self.combinations = list(combinations(range(n_train, n_train + 20), 2))

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            load_cache = train_cache
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                (
                    self.verts_list,
                    self.faces_list,
                    self.frames_list,
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    self.hks_list,
                    self.vts_list,
                    self.names_list,
                    self.sample_list
                ) = torch.load(load_cache)
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes & labels

        # Get all the files
        mesh_files = []
        vts_files = []

        # load faust data
        mesh_dirpath = os.path.join(self.root_dir, name, "shapes")
        vts_dirpath = os.path.join(self.root_dir, name, "correspondences")
        for fname in os.listdir(mesh_dirpath):
            mesh_fullpath = os.path.join(mesh_dirpath, fname)
            vts_fullpath = os.path.join(vts_dirpath, fname[:-4] + ".vts")
            mesh_files.append(mesh_fullpath)
            vts_files.append(vts_fullpath)

        print("loading {} meshes".format(len(mesh_files)))

        # TODO verify that they are sorted correctly
        mesh_files, vts_files = sorted(mesh_files), sorted(vts_files)

        # Load the actual files
        for iFile in range(len(mesh_files)):

            print("loading mesh " + str(mesh_files[iFile]))

            verts, faces = pp3d.read_mesh(mesh_files[iFile])
            vts_file = np.loadtxt(vts_files[iFile]).astype(int)

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            vts_file = torch.tensor(np.ascontiguousarray(vts_file))

            # center and unit scale
            verts = diffusion_net.geometry.normalize_positions(verts)

            # normalize area
            verts = normalize_area_scale(verts, faces)

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.vts_list.append(vts_file)
            self.names_list.append(os.path.basename(mesh_files[iFile]).split(".")[0])
            idx0 = farthest_point_sample(verts.t(), ratio=0.9)
            dists, idx1 = square_distance(verts.unsqueeze(0), verts[idx0].unsqueeze(0)).sort(dim=-1)
            dists, idx1 = dists[:, :, :130].clone(), idx1[:, :, :130].clone()
            self.sample_list.append((idx0, idx1, dists))

        for ind, labels in enumerate(self.vts_list):
            self.vts_list[ind] = labels

        # Precompute operators
        (
            self.frames_list,
            self.massvec_list,
            self.L_list,
            self.evals_list,
            self.evecs_list,
            self.gradX_list,
            self.gradY_list,
        ) = diffusion_net.geometry.get_all_operators(
            self.verts_list,
            self.faces_list,
            k_eig=self.k_eig,
            op_cache_dir=self.op_cache_dir,
        )

        self.hks_list = [diffusion_net.geometry.compute_hks_autoscale(self.evals_list[i], self.evecs_list[i], 16)
                         for i in range(len(self.L_list))]

        # save to cache
        if use_cache:
            diffusion_net.utils.ensure_dir_exists(self.cache_dir)
            torch.save(
                (
                    self.verts_list,
                    self.faces_list,
                    self.frames_list,
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    self.hks_list,
                    self.vts_list,
                    self.names_list,
                    self.sample_list
                ),
                load_cache,
            )

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        idx1, idx2 = self.combinations[idx]

        # shape1 = [
        #     self.verts_list[idx1],
        #     self.faces_list[idx1],
        #     self.frames_list[idx1],
        #     self.massvec_list[idx1],
        #     self.L_list[idx1],
        #     self.evals_list[idx1],
        #     self.evecs_list[idx1],
        #     self.gradX_list[idx1],
        #     self.gradY_list[idx1],
        #     self.hks_list[idx1],
        #     self.vts_list[idx1],
        #     self.names_list[idx1],
        # ]

        shape1 = {
            "xyz": self.verts_list[idx1],
            "faces": self.faces_list[idx1],
            "frames": self.frames_list[idx1],
            "mass": self.massvec_list[idx1],
            "L": self.L_list[idx1],
            "evals": self.evals_list[idx1],
            "evecs": self.evecs_list[idx1],
            "gradX": self.gradX_list[idx1],
            "gradY": self.gradY_list[idx1],
            "name": self.names_list[idx1],
            "vts": self.vts_list[idx1],
            "sample_idx": self.sample_list[idx1],
        }

        # shape2 = [
        #     self.verts_list[idx2],
        #     self.faces_list[idx2],
        #     self.frames_list[idx2],
        #     self.massvec_list[idx2],
        #     self.L_list[idx2],
        #     self.evals_list[idx2],
        #     self.evecs_list[idx2],
        #     self.gradX_list[idx2],
        #     self.gradY_list[idx2],
        #     self.hks_list[idx2],
        #     self.vts_list[idx2],
        #     self.names_list[idx2],
        # ]

        shape2 = {
            "xyz": self.verts_list[idx2],
            "faces": self.faces_list[idx2],
            "frames": self.frames_list[idx2],
            "mass": self.massvec_list[idx2],
            "L": self.L_list[idx2],
            "evals": self.evals_list[idx2],
            "evecs": self.evecs_list[idx2],
            "gradX": self.gradX_list[idx2],
            "gradY": self.gradY_list[idx2],
            "name": self.names_list[idx2],
            "vts": self.vts_list[idx2],
            "sample_idx": self.sample_list[idx2],
        }

        # Compute fmap
        vts1, vts2 = shape1["vts"], shape2["vts"]
        evec_1, evec_2 = shape1["evecs"][:, :self.n_fmap], shape2["evecs"][:, :self.n_fmap]
        evec_1_a, evec_2_a = evec_1[vts1], evec_2[vts2]
        C_gt = torch.lstsq(evec_2_a, evec_1_a)[0][:evec_1_a.size(-1)].t()

        # compute region labels
        gt_partiality_mask12 = torch.ones(shape1["xyz"].size(0)).long().detach()
        gt_partiality_mask21 = torch.ones(shape2["xyz"].size(0)).long().detach()

        # create map21
        map21 = torch.ones(shape2["xyz"].size(0)).long().detach() * -1
        map21[vts2] = vts1

        return {"shape1": shape1, "shape2": shape2, "C_gt": C_gt,
                "map1": vts1, "map2": vts2, "map21": map21, "gt_partiality_mask12": gt_partiality_mask12, "gt_partiality_mask21": gt_partiality_mask21}

        return (shape1, shape2, C_gt)


def shape_to_device(dict_shape, device):
    names_to_device = ["xyz", "faces", "mass", "evals", "evecs", "gradX", "gradY"]
    for k, v in dict_shape.items():
        if "shape" in k:
            for name in names_to_device:
                v[name] = v[name].to(device)
            dict_shape[k] = v
        else:
            dict_shape[k] = v.to(device)

    return dict_shape
