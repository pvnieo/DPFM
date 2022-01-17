import os
from pathlib import Path
import numpy as np
import potpourri3d as pp3d
import torch
from torch.utils.data import Dataset
import diffusion_net as dfn
from utils import farthest_point_sample, square_distance


class ShrecPartialDataset(Dataset):
    def __init__(self, root_dir, name="cuts", k_eig=128, n_fmap=30, use_cache=True, op_cache_dir=None):

        self.k_eig = k_eig
        self.n_fmap = n_fmap
        self.root_dir = root_dir
        self.cache_dir = root_dir
        self.op_cache_dir = op_cache_dir

        # check the cache
        if use_cache:
            load_cache = os.path.join(self.cache_dir, f"cache_{name}_train.pt")
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
                    self.used_shapes,
                    self.corres_dict,
                    self.sample_list,
                ) = torch.load(load_cache)
                self.combinations = list(self.corres_dict.keys())
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes & labels
        # define files and order
        self.used_shapes = sorted([x.stem for x in (Path(root_dir) / "shapes").iterdir() if name in x.stem])
        corres_path = Path(root_dir) / "maps"
        all_combs = [x.stem for x in corres_path.iterdir() if name in x.stem]
        self.corres_dict = {}
        for x, y in map(lambda x: (x[:x.rfind("_")], x[x.rfind("_") + 1:]), all_combs):
            if x in self.used_shapes and y in self.used_shapes:
                map_ = torch.from_numpy(np.loadtxt(corres_path / f"{x}_{y}.map", dtype=np.int32)).long()
                self.corres_dict[(self.used_shapes.index(y), self.used_shapes.index(x))] = map_

        # set combinations
        self.combinations = list(self.corres_dict.keys())
        mesh_dirpath = Path(root_dir) / "shapes"

        # Get all the files
        self.verts_list = []
        self.faces_list = []
        self.sample_list = []

        # Load the actual files
        for shape_name in self.used_shapes:
            print("loading mesh " + str(shape_name))

            verts, faces = pp3d.read_mesh(str(mesh_dirpath / f"{shape_name}.off"))

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            self.verts_list.append(verts)
            self.faces_list.append(faces)
            idx0 = farthest_point_sample(verts.t(), ratio=0.9)
            dists, idx1 = square_distance(verts.unsqueeze(0), verts[idx0].unsqueeze(0)).sort(dim=-1)
            dists, idx1 = dists[:, :, :130].clone(), idx1[:, :, :130].clone()
            self.sample_list.append((idx0, idx1, dists))
        # Precompute operators
        (
            self.frames_list,
            self.massvec_list,
            self.L_list,
            self.evals_list,
            self.evecs_list,
            self.gradX_list,
            self.gradY_list,
        ) = dfn.geometry.get_all_operators(
            self.verts_list,
            self.faces_list,
            k_eig=self.k_eig,
            op_cache_dir=self.op_cache_dir,
        )

        # save to cache
        if use_cache:
            dfn.utils.ensure_dir_exists(self.cache_dir)
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
                    self.used_shapes,
                    self.corres_dict,
                    self.sample_list,
                ),
                load_cache,
            )

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        idx1, idx2 = self.combinations[idx]

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
            "name": self.used_shapes[idx1],
            "sample_idx": self.sample_list[idx1],
        }

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
            "name": self.used_shapes[idx2],
            "sample_idx": self.sample_list[idx2],
        }

        # Compute fmap
        map21 = self.corres_dict[(idx1, idx2)]

        evec_1, evec_2, mass2 = shape1["evecs"][:, :self.n_fmap], shape2["evecs"][:, :self.n_fmap], shape2["mass"]
        trans_evec2 = evec_2.t() @ torch.diag(mass2)

        P = torch.zeros(evec_2.size(0), evec_1.size(0))
        P[range(evec_2.size(0)), map21.flatten()] = 1
        C_gt = trans_evec2 @ P @ evec_1

        # compute region labels
        gt_partiality_mask12 = torch.zeros(shape1["xyz"].size(0)).long().detach()
        gt_partiality_mask12[map21[map21 != -1]] = 1
        gt_partiality_mask21 = torch.zeros(shape2["xyz"].size(0)).long().detach()
        gt_partiality_mask21[map21 != -1] = 1

        return {"shape1": shape1, "shape2": shape2, "C_gt": C_gt,
                "map21": map21, "gt_partiality_mask12": gt_partiality_mask12, "gt_partiality_mask21": gt_partiality_mask21}


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
