import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np

class CO3DDataset(torch.utils.data.Dataset):
    """
    CO3D dataset (CITE HERE)
    """

    def __init__(
        self, path, dataset, stage="train", image_size=256, world_scale=1.0, category="plant", z_near=1.0, z_far=10.0,
    ):
        """
        :param stage train | val | test
        :param dataset co3d Dataset
        :param image_size result image size (resizes if different)
        :param world_scale amount to scale entire world by
        :param category class from CO3D Dataset
        """
        super().__init__()
        # path would be ./temp/CO3D
        self.path = path
        self.dataset_name = os.path.basename(self.path) # CO3D

        self.z_near = z_near
        self.z_far = z_far

        self.category = category
        print("Loading CO3D dataset", self.path, "name:", self.dataset_name, "category:", self.category)
        self.stage = stage
        assert os.path.exists(self.path)

        self.image_size = image_size
        self.world_scale = world_scale
        self._coord_trans = torch.diag(
            torch.tensor([-1, 1, 1, 1], dtype=torch.float32) # flip pytorch axis (-x, y, z) to (x, y, z)
        )

        frame_file = os.path.join(self.path, self.category, "frame_annotations.jgz")
        sequence_file = os.path.join(self.path, self.category, "sequence_annotations.jgz")
        self.image_size = image_size
        self.lindisp = False

        # self.dataset = Co3dDataset(
        #     frame_annotations_file=frame_file,
        #     sequence_annotations_file=sequence_file,
        #     dataset_root=self.path,
        #     image_height=self.image_size,
        #     image_width=self.image_size,
        #     box_crop=True,
        #     load_point_clouds=False,
        #     remove_empty_masks=False,
        # )

        self.dataset = dataset

        if stage == "train":
            self.all_objs = []
            p = os.path.join(self.path, self.category, "train.lst")
            with open(p, "r") as file:
                for line in file: 
                    line = line.strip()
                    if line != "":
                        self.all_objs.append(line)
        elif stage == "val":
            self.all_objs = []
            p = os.path.join(self.path, self.category, "val.lst")
            with open(p, "r") as file:
                for line in file: 
                    line = line.strip()
                    if line != "":
                        self.all_objs.append(line)
        elif stage == "test":
            self.all_objs = []
            p = os.path.join(self.path, self.category, "val.lst")
            with open(p, "r") as file:
                for line in file: 
                    line = line.strip()
                    if line != "":
                        self.all_objs.append(line)

    def __len__(self):
        return len(self.all_objs)

    def __getitem__(self, index):
        
        seq_idx = self.dataset.seq_to_idx[ self.all_objs[index] ]
        dataset_index = torch.utils.data.Subset(self.dataset, seq_idx)
        
        all_imgs = []
        all_poses = []
        all_masks = []
        all_depth = []
        focal = None

        # Prepare to average intrinsics over images
        fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0

        for d in dataset_index:
            
            all_imgs.append(d.image_rgb)
            all_masks.append(d.depth_mask)
            all_depth.append(d.depth_map)
            
            p = d.camera.principal_point.squeeze(0)
            f = d.camera.focal_length.squeeze(0)
            h = d.image_size_hw[0]
            w = d.image_size_hw[1]

            K = torch.eye(3)
            s = (torch.min(h, w) - 1) / 2
            K[0, 0] = f[0] * (w - 1) / 2
            K[1, 1] = f[1] * (h - 1) / 2
            K[0, 2] = -p[0] * s + (w - 1) / 2
            K[1, 2] = -p[1] * s + (h - 1) / 2

            pose = torch.eye(4)
            pose[:3, :3] = d.camera.R.squeeze(0).T
            pose[:3, 3] = d.camera.T.squeeze(0)

            pose = self._coord_trans @ pose # flip pytorch axis (-x, y, z) to (x, y, z)

            fx += K[0, 0]
            fy += K[1, 1]
            cx += K[0, 2]
            cy += K[1, 2]

            all_poses.append(pose)

        fx /= len(all_imgs)
        fy /= len(all_imgs)
        cx /= len(all_imgs)
        cy /= len(all_imgs)
        focal = torch.tensor((fx, fy), dtype=torch.float32)
        c = torch.tensor((cx, cy), dtype=torch.float32)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_depth = torch.stack(all_depth)

        result = {
            "path": self.path,
            "img_id": index,
            "focal": focal,
            "c": c,
            "images": all_imgs,
            "masks": all_masks,
            "poses": all_poses,
        }
        return result
        
