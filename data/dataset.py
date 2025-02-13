import torch
from torch.utils import data
import numpy as np
import json
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from pathlib import Path
import spacy

from torch.utils.data._utils.collate import default_collate
from data.utils import smpl_fk, resample_motion_fps, HUMANOID2SMPLX, get_handpose, load_arctic_data, get_humanoid_data, get_obj_data, aztoay_hoi
import data.matrix as matrix
from data.hml3d.utils import convert_motion_to_hmlvec263_original
from data.smplx_utils import make_smplx

# import spacy

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text':text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass


        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if self.opt.is_train:
            if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
                if self.opt.unit_length < 10:
                    coin2 = np.random.choice(['single', 'single', 'double'])
                else:
                    coin2 = 'single'
                if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                    m_length = self.max_length
                    idx = random.randint(0, m_length - self.max_length)
                    motion = motion[idx:idx+self.max_length]
                else:
                    if coin2 == 'single':
                        n_m_length = self.max_length + self.opt.unit_length * len_gap
                    else:
                        n_m_length = self.max_length + self.opt.unit_length * (len_gap - 1)
                    idx = random.randint(0, m_length - n_m_length)
                    motion = motion[idx:idx + self.max_length]
                    m_length = n_m_length
                # print(len_gap, idx, coin2)
        else:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'

            if coin2 == 'double':
                m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
            elif coin2 == 'single':
                m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length


'''For use of training text motion matching model, and evaluations'''
class Text2MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)



def load_motion_and_text(base_motion_path, base_text_path):
    data = {}
    
    # 遍历motion数据的子集文件夹
    for subset in os.listdir(base_motion_path):
        motion_subset_path = os.path.join(base_motion_path, subset)
        text_subset_path = os.path.join(base_text_path, subset)

        if os.path.isdir(motion_subset_path) and os.path.isdir(text_subset_path):
            # 遍历每个subset中的.npy文件
            for file in os.listdir(motion_subset_path):
                if file.endswith('.npy'):
                    motion_file_path = os.path.join(motion_subset_path, file)
                    text_file_path = os.path.join(text_subset_path, file.replace('.npy', '.txt'))

                    # 读取motion和text
                    if os.path.exists(text_file_path):
                        motion = np.load(motion_file_path)
                        with open(text_file_path, 'r') as f:
                            text = f.read().strip()
                        
                        # 存储在字典中
                        data[subset+ "/" + file[:-4]] = {"motion": motion, "text": text}

    return data


def load_mixed(base_motion_path, base_text_path, mixed_text_path="../Motion-2Dto3D/inputs/motionx/motionx_seq_text_v1.1/mixed_train_seq_names.json"):
    with open(mixed_text_path, "r") as file:
        test_seq_names = json.load(file)
    data = {}
    for k in test_seq_names:
        motion_path = os.path.join(base_motion_path, k + ".npy")
        text_path = os.path.join(base_text_path, k + ".txt")
        motion = np.load(motion_path)
        with open(text_path, 'r') as f:
            text = f.read().strip()
        data[k] = {"motion": motion, "text": text}

    return data


# MotionX training
class Text2MotionDatasetV3(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 

        self.token_model = spacy.load("en_core_web_sm")

        if opt.dataset_name == "mixed":
            data_dict = load_mixed(opt.motion_dir, opt.text_dir)
        elif opt.dataset_name == "idea400":
            if "train" in split_file:
                mixed_text_path = "../Motion-2Dto3D/inputs/motionx/motionx_seq_text_v1.1/idea400_train_seq_names.json"
            else:
                mixed_text_path = "../Motion-2Dto3D/inputs/motionx/motionx_seq_text_v1.1/idea400_test_seq_names.json"
            data_dict = load_mixed(opt.motion_dir, opt.text_dir, mixed_text_path)
        else:
            data_dict = load_motion_and_text(opt.motion_dir, opt.text_dir)

        data_dict, length_list, name_list = self.prepare_meta(data_dict)
        
        if False:
            # add hml3d into training
            if "val.txt" in split_file:
                data_dict, length_list, name_list = {}, [], []
            else:
                data_dict, length_list, name_list = self.prepare_meta(data_dict)

            hml3d_data_dict = {}
            hml3d_data = torch.load("../Motion-2Dto3D/inputs/hml3d/joints3d.pth")
        
            txt_path = f"../Motion-2Dto3D/inputs/hml3d/texts"

            max_motion_len = 200
            min_motion_len = 40

            with cs.open(split_file, "r") as f:
                for line in f.readlines():
                    seq_name = line.strip()
                    if seq_name + ".npy" in hml3d_data.keys():
                        motion = hml3d_data[seq_name + ".npy"]["joints3d"]
                        motion_len = motion.shape[0]
                        # Follow MDM, only uses [2s ~ 10s]
                        if motion_len < min_motion_len or motion_len > max_motion_len:
                            continue
                        with cs.open(os.path.join(txt_path, seq_name + ".txt")) as text_f:
                            for text_line in text_f.readlines():
                                text_dict = {}
                                line_split = text_line.strip().split("#")
                                caption = line_split[0]
                                tokens = line_split[1].split(" ")
                                f_tag = float(line_split[2])
                                to_tag = float(line_split[3])
                                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                                text_dict["caption"] = caption
                                text_dict["tokens"] = tokens
                                if f_tag == 0.0 and to_tag == 0.0:
                                    if seq_name in hml3d_data_dict:
                                        seq_name += "_1"
                                    hml3d_data_dict[seq_name] = {"motion": motion, "text": caption, "length": motion.shape[0], "is_hml3d": True}
                                    name_list.append(seq_name)
                                else:
                                    start_frame = int(f_tag * 20)
                                    end_frame = int(to_tag * 20)
                                    n_motion = motion[start_frame:end_frame]
                                    if (len(n_motion)) < min_motion_len or (len(n_motion) > max_motion_len):
                                        continue
                                    if seq_name in hml3d_data_dict:
                                        seq_name += "_1"
                                    hml3d_data_dict[seq_name] = {"motion": n_motion, "text": caption, "length": n_motion.shape[0], "is_hml3d": True}
                                    name_list.append(seq_name)

            data_dict.update(hml3d_data_dict)
                            
        body_models = {
            "male": make_smplx("rich-smplx", gender="male"),
            "neutral": make_smplx("rich-smplx", gender="neutral"),
            "female": make_smplx("rich-smplx", gender="female"),
        }
        self.smpl = body_models

        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        self.name_list = name_list

    def prepare_meta(self, data_dict):
        new_data_dict = {}
        length_list = []
        name_list = []
        for k in data_dict.keys():
            motion = data_dict[k]["motion"]
            text = data_dict[k]["text"]
            L = motion.shape[0]
            if L < 2 * 30:
                continue
            if L > 10 * 30:
                continue
            new_data_dict[k] = {"motion": motion, "text": text, "length": L, "is_hml3d": False}
            length_list.append(L)
            name_list.append(k)
        
        return new_data_dict, length_list, name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text = data['motion'], data['length'], data['text']
        is_hml3d = data['is_hml3d']
        if is_hml3d:
            joints = data["motion"]
            m_length -= 1
            if isinstance(joints, torch.Tensor):
                joints = joints.numpy()
        else:
            m_length = m_length // 3 * 2
            m_length = min(m_length, self.max_motion_length)

            motion = torch.tensor(motion, dtype=torch.float32)
            smpl_params = {"global_orient": motion[:, :3],
                      "body_pose": motion[:, 3:66],
                      "transl": motion[:, 309:312],
                      "betas": None,
                      }

            joints = smpl_fk(self.smpl["neutral"], **smpl_params)  # (F, 22, 3)

            joints = resample_motion_fps(joints, m_length + 1) # original convert will remove 1 frame
            joints = joints.numpy()

        motion, _, _, _ = convert_motion_to_hmlvec263_original(joints)

        # Randomly select a caption
        caption = text
        caption = caption.replace('/', ' ')
        tokens = self.token_model(caption)
        token_format = " ".join([f"{token.text}/{token.pos_}" for token in tokens])
        tokens = token_format.split(" ")

        filter_tokens = []
        for token in tokens:
            try: 
                word_emb, pos_oh = self.w_vectorizer[token]
            except Exception as e:
                continue
            filter_tokens.append(token)
        tokens = filter_tokens

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        if len(motion) > m_length:
            idx = random.randint(0, len(motion) - m_length)
        else:
            idx = 0
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)



# ARCTIC training
class Text2MotionDatasetV4(data.Dataset):
    def __init__(self, opt, mean, std, split, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 

        self.token_model = spacy.load("en_core_web_sm")
        self.split = split

        self.test_sbj = ["s01", "s10"]

        self.mean = np.array([-2.00921501e-03,  9.38767988e-01, -1.35749395e-03,  5.57863477e-02,
        8.46087177e-01, -2.66990821e-02, -6.45776550e-02,  8.34755720e-01,
       -2.10076671e-02, -5.52574401e-03,  1.04779603e+00, -2.96834077e-02,
        1.22112615e-01,  4.76183925e-01, -4.08915014e-02, -1.18454424e-01,
        4.81293256e-01, -5.35632505e-02,  6.58911289e-03,  1.18043456e+00,
       -3.68465921e-02,  9.06854641e-02,  7.97914026e-02, -5.97199721e-02,
       -1.14727457e-01,  7.90358819e-02, -6.70341853e-02, -7.33095065e-03,
        1.23020187e+00, -6.99912577e-03,  1.53111839e-01,  2.50787011e-02,
        4.82791799e-02, -1.80402382e-01,  1.86145559e-02,  3.46453367e-02,
       -1.87492025e-02,  1.39302999e+00, -3.28376327e-02,  3.76353352e-02,
        1.31385855e+00, -8.94564136e-03, -5.36809784e-02,  1.31328374e+00,
       -1.87594897e-02,  6.02411601e-03,  1.54308990e+00,  1.07701186e-02,
        1.62161752e-01,  1.33996654e+00,  7.62200249e-04, -1.60303445e-01,
        1.34216936e+00, -2.25472424e-02,  1.76987845e-01,  1.10333012e+00,
        7.16321369e-02, -2.32868821e-01,  1.12495910e+00,  7.93349011e-02,
        1.05841381e-01,  1.10609981e+00,  2.82065803e-01, -1.54303136e-01,
        1.13092894e+00,  2.86620282e-01,  8.19814386e-02,  1.12789863e+00,
        3.63838497e-01,  7.33279645e-02,  1.12605136e+00,  3.85989208e-01,
        6.15930307e-02,  1.12074782e+00,  3.94685430e-01,  8.52002617e-02,
        1.11427051e+00,  3.71523942e-01,  7.26450821e-02,  1.11089495e+00,
        3.88034600e-01,  5.80314661e-02,  1.10697305e+00,  3.95380325e-01,
        8.81782620e-02,  1.08523397e+00,  3.52657417e-01,  7.87351089e-02,
        1.08506296e+00,  3.59472136e-01,  6.76119051e-02,  1.08439094e+00,
        3.65589472e-01,  8.78073806e-02,  1.09779732e+00,  3.62940402e-01,
        7.39290426e-02,  1.09753210e+00,  3.77427401e-01,  5.89925402e-02,
        1.09508809e+00,  3.84532806e-01,  8.43066229e-02,  1.12413342e+00,
        3.12690045e-01,  7.17577604e-02,  1.12789959e+00,  3.29200315e-01,
        6.30262227e-02,  1.13045198e+00,  3.45964060e-01, -1.31593978e-01,
        1.15721003e+00,  3.65367057e-01, -1.24180354e-01,  1.15346829e+00,
        3.88417522e-01, -1.13973937e-01,  1.14660549e+00,  3.98932640e-01,
       -1.37222154e-01,  1.14480321e+00,  3.74867759e-01, -1.25825768e-01,
        1.13839424e+00,  3.92142001e-01, -1.13078559e-01,  1.13234632e+00,
        3.99791464e-01, -1.43043248e-01,  1.11418363e+00,  3.60641450e-01,
       -1.34431589e-01,  1.11325803e+00,  3.68146499e-01, -1.25266354e-01,
        1.11029347e+00,  3.74617746e-01, -1.41565170e-01,  1.12777746e+00,
        3.68787792e-01, -1.28078165e-01,  1.12526356e+00,  3.83317556e-01,
       -1.15242222e-01,  1.12059805e+00,  3.91484839e-01, -1.30919017e-01,
        1.14836194e+00,  3.15020016e-01, -1.17695177e-01,  1.15096292e+00,
        3.31969604e-01, -1.08899542e-01,  1.15511180e+00,  3.49366090e-01,
        4.80002526e-02,  4.13842687e-02, -8.72942231e-02,  3.63049039e-02,
        5.14553552e-02, -1.33574210e-01,  3.00841461e-02,  6.43677381e-02,
       -9.23038757e-02,  7.28571079e-02,  2.82818228e-02, -6.14001728e-02,
       -2.99747666e-02,  9.87667763e-02, -2.51690906e-01, -3.20306631e-02,
        1.23286241e-01, -1.65348762e-01,  9.81889141e-02,  8.28447205e-03,
       -3.36583391e-02, -1.01042786e-01,  1.59704677e-01, -3.41633657e-01,
       -1.01802974e-01,  1.81891728e-01, -2.69396610e-01,  1.02559437e-01,
       -1.95154595e-04, -1.39923038e-02, -1.27043411e-01,  1.53044096e-01,
       -3.67784109e-01, -1.30274679e-01,  1.89499231e-01, -2.52535566e-01,
        1.37124202e-01, -2.07793721e-02,  2.77999626e-02,  1.19751647e-01,
       -1.64648359e-02, -9.63349626e-03,  1.19021719e-01, -6.94440276e-03,
        2.18346044e-02,  1.60226605e-01, -4.80423739e-02,  5.93684335e-02,
        1.25341863e-01, -3.18054483e-02, -4.62863764e-02,  1.21650187e-01,
       -1.03831607e-03,  6.59321185e-02,  6.47095546e-02, -1.49338147e-03,
       -1.01419726e-01,  5.18656183e-02,  2.38495127e-02,  5.05316481e-02,
        3.31106290e-02, -9.02134865e-03, -5.70000144e-02,  2.23272772e-02,
        4.88113206e-03,  3.52227304e-02,  2.50031803e-02, -1.60322349e-02,
       -3.62076541e-02,  2.03572980e-02, -1.59882120e-02, -3.10303075e-02,
        1.56069128e-02, -1.28112232e-02, -2.76658534e-02,  2.01523716e-02,
       -1.55724742e-02, -3.83931238e-02,  1.42209210e-02, -1.36048109e-02,
       -3.26317883e-02,  9.79526603e-03, -9.97006240e-03, -2.81623950e-02,
        1.54362646e-02, -1.06690706e-02, -4.66526283e-02,  1.34255755e-02,
       -9.64512364e-03, -4.33555143e-02,  1.09443596e-02, -7.36077482e-03,
       -3.94336527e-02,  1.71605182e-02, -1.31235985e-02, -4.30456991e-02,
        1.26276678e-02, -1.17441106e-02, -3.68953725e-02,  8.64855247e-03,
       -8.25295453e-03, -3.19402206e-02,  3.24049330e-02, -1.10379106e-02,
       -4.42133301e-02,  3.06485347e-02, -1.17329332e-02, -3.86134609e-02,
        2.89249862e-02, -1.26690860e-02, -3.45311176e-02,  1.57968402e-02,
       -6.73118901e-03,  3.81294427e-02,  1.06709689e-02, -8.32752202e-03,
        3.49820026e-02,  6.15654424e-03, -7.37946338e-03,  2.93712362e-02,
        1.12942573e-02, -5.55740030e-03,  3.76375712e-02,  5.25521057e-03,
       -4.54152876e-03,  3.11037796e-02,  5.12056968e-04, -2.40166856e-03,
        2.42830435e-02,  6.29049444e-03,  1.41419477e-03,  3.01107096e-02,
        3.59715886e-03,  2.43579843e-03,  2.61553648e-02,  5.13374818e-04,
        4.20803290e-03,  2.11543506e-02,  8.21810998e-03, -1.90591562e-03,
        3.40842411e-02,  3.31802549e-03, -9.26411196e-04,  2.74143016e-02,
       -1.11688771e-03,  1.04432876e-03,  2.06328234e-02,  2.18617201e-02,
       -1.18390782e-04,  3.17342755e-02,  2.04183589e-02, -2.32791600e-03,
        2.85721952e-02,  1.87597685e-02, -4.73756618e-03,  2.74504998e-02])
        self.std = np.array([0.02780309, 0.05796335, 0.01587407, 0.02872579, 0.05361869,
       0.01896513, 0.02944513, 0.05404601, 0.01901566, 0.02910634,
       0.06324   , 0.01828851, 0.02820635, 0.02896972, 0.03299947,
       0.02762931, 0.03106783, 0.03217027, 0.03122959, 0.07277743,
       0.02285246, 0.03408906, 0.00522432, 0.02632153, 0.03255344,
       0.00600583, 0.02971304, 0.03085778, 0.07620697, 0.02469245,
       0.03518368, 0.00443974, 0.03468419, 0.02940659, 0.00320628,
       0.03512695, 0.03832338, 0.08453548, 0.03221419, 0.03400938,
       0.08097612, 0.02892018, 0.03439043, 0.0797054 , 0.02799257,
       0.04310803, 0.08781063, 0.04465688, 0.03755655, 0.07661285,
       0.0458821 , 0.03982052, 0.0715544 , 0.04133033, 0.05387375,
       0.07860749, 0.0900869 , 0.05632668, 0.08492889, 0.09116944,
       0.09517223, 0.1057756 , 0.10905123, 0.09550031, 0.10182305,
       0.10622003, 0.11525763, 0.12830104, 0.11918138, 0.12160176,
       0.13552495, 0.12295611, 0.12503036, 0.13870439, 0.12591504,
       0.11721246, 0.13035336, 0.12224127, 0.12228798, 0.1354586 ,
       0.12671084, 0.12580046, 0.13822463, 0.13025617, 0.11528701,
       0.12498573, 0.12467785, 0.11768465, 0.12730784, 0.12757245,
       0.12034354, 0.12950082, 0.13055141, 0.11618661, 0.12777986,
       0.12366022, 0.12057885, 0.13197157, 0.12787927, 0.12405325,
       0.13447801, 0.13144724, 0.10286344, 0.11445781, 0.11132792,
       0.10867899, 0.12071883, 0.11268368, 0.11515709, 0.12692796,
       0.11498764, 0.11764144, 0.11460819, 0.11666655, 0.12443239,
       0.11973362, 0.12014527, 0.12733206, 0.12135731, 0.12272081,
       0.11846047, 0.11799386, 0.11861991, 0.12366485, 0.12109918,
       0.12247545, 0.12584012, 0.12138209, 0.12599693, 0.11347538,
       0.1165509 , 0.11909909, 0.11552067, 0.11705335, 0.1223982 ,
       0.11753579, 0.11747561, 0.12575843, 0.11564653, 0.11770728,
       0.11888442, 0.11972949, 0.11936485, 0.12314387, 0.12229695,
       0.11991024, 0.12720919, 0.10383369, 0.10397437, 0.10921388,
       0.10978797, 0.10679016, 0.11046376, 0.11600688, 0.10976909,
       0.11273549, 0.29993349, 0.2693656 , 0.24988976, 0.34408125,
       0.30768426, 0.27252528, 0.34028362, 0.30262356, 0.28467739,
       0.29085077, 0.2715455 , 0.25615854, 0.49800957, 0.47012163,
       0.36864506, 0.49052703, 0.46332297, 0.40788026, 0.27812952,
       0.28576715, 0.26394837, 0.68173239, 0.6857559 , 0.51995122,
       0.67765775, 0.68428499, 0.55945267, 0.25898458, 0.27900039,
       0.25345598, 0.68116473, 0.70865835, 0.50987028, 0.67581685,
       0.7107759 , 0.57596319, 0.2831374 , 0.34364102, 0.29435499,
       0.26393464, 0.30713945, 0.27036835, 0.26839763, 0.30690806,
       0.27126562, 0.29447835, 0.39175791, 0.31724872, 0.27705134,
       0.32787135, 0.2925191 , 0.28412289, 0.32090177, 0.28488316,
       0.26062183, 0.24776048, 0.23173174, 0.25327449, 0.22854754,
       0.24458897, 0.15966626, 0.14898206, 0.14328117, 0.14035566,
       0.12949902, 0.13384487, 0.14278984, 0.12889628, 0.13278479,
       0.14276064, 0.12630959, 0.13240632, 0.14145473, 0.12399198,
       0.13097648, 0.1468899 , 0.13163205, 0.13516346, 0.14550001,
       0.12898311, 0.13398125, 0.14357475, 0.1261729 , 0.13269718,
       0.15380039, 0.13847274, 0.1379018 , 0.15115507, 0.1359042 ,
       0.13664776, 0.14883633, 0.13347778, 0.13585625, 0.15062135,
       0.13540775, 0.13669318, 0.14745805, 0.13170161, 0.13477917,
       0.14511485, 0.12864755, 0.13370709, 0.14359904, 0.13454773,
       0.13285629, 0.13721567, 0.12822158, 0.12895414, 0.13424001,
       0.1241582 , 0.12747572, 0.11941536, 0.11478436, 0.10947583,
       0.11724196, 0.1124292 , 0.10706047, 0.1140705 , 0.10953586,
       0.10543521, 0.12159064, 0.11559413, 0.11132208, 0.11814637,
       0.1130787 , 0.10909277, 0.1146469 , 0.10942958, 0.10719518,
       0.12706253, 0.11664327, 0.11766986, 0.1239506 , 0.11370859,
       0.11514092, 0.12162471, 0.11125737, 0.11342825, 0.12430385,
       0.11610631, 0.11461329, 0.11949686, 0.11224608, 0.1108412 ,
       0.11622194, 0.10920688, 0.10909386, 0.12352401, 0.11749619,
       0.11697663, 0.11636023, 0.11043399, 0.10976336, 0.11195368,
       0.10686915, 0.10528832])

        self.smplx = make_smplx(type="wholebody")

        self.motion_files = {}
        dataset_path = "./inputs/arctic_neutral"
        # ./inputs/arctic_neutral/s01/xx.pt
        max_length = -1
        for path in tqdm(Path(dataset_path).glob("**/*.pt")):
            if self.split == "train":
                if path.parent.name in self.test_sbj:
                    continue
            else:
                if path.parent.name not in self.test_sbj:
                    continue

            vid_name = path.parent.name + "_" + path.name
            # print(f"Loading {vid_name}")
            motion_data = load_arctic_data(path)
            humanoid_localmat, humanoid_globalmat = get_humanoid_data(motion_data)
            obj_localmat, obj_globalmat = get_obj_data(motion_data)
            contact = motion_data["contact"]
            angles = motion_data["obj"]["angles"]
            beta = motion_data["humanoid"]["betas"]
            self.motion_files[vid_name] = {
                "humanoid_localmat": humanoid_localmat,
                "humanoid_globalmat": humanoid_globalmat,
                "obj_localmat": obj_localmat,
                "obj_globalmat": obj_globalmat,
                "contact": contact,
                "angles": angles,
                "beta": beta,
            }
            # print(f"seq {vid_name} length: {humanoid_localmat.shape[0]}")
            max_length = max(max_length, humanoid_localmat.shape[0])
        print(f"max_length: {max_length}")
        self.idx2meta = []
        for k, v in self.motion_files.items():
            L = v["beta"].shape[0]
            for _ in range(max(L // self.max_motion_length, 1)):
                self.idx2meta.append(k)
        self.bps_data = torch.load("./inputs/arctic_bps.pth")
        print(f"Motion dataset size: {len(self.idx2meta)}")
    
    def _load_data(self, idx):
        vid = self.idx2meta[idx]
        humanoid_localmat = self.motion_files[vid]["humanoid_localmat"]
        humanoid_globalmat = self.motion_files[vid]["humanoid_globalmat"]
        obj_localmat = self.motion_files[vid]["obj_localmat"]
        obj_globalmat = self.motion_files[vid]["obj_globalmat"]

        start_id = 0

        raw_len = humanoid_globalmat.shape[0] - start_id
        # Get {tgt_len} frames from data
        # Random select a subset with speed augmentation  [start, end)
        tgt_len = 300
        # raw_subset_len = np.random.randint(int(tgt_len / self.l_factor), int(tgt_len * self.l_factor))
        raw_subset_len = tgt_len
        if raw_subset_len <= raw_len:
            start = np.random.randint(0, raw_len - raw_subset_len + 1)
            end = start + raw_subset_len
        else:  # interpolation will use all possible frames (results in a slow motion)
            start = 0
            end = raw_len


        humanoid_localmat = humanoid_localmat[start:end].clone()
        humanoid_globalmat = humanoid_globalmat[start:end].clone()
        obj_localmat = obj_localmat[start:end].clone()
        obj_globalmat = obj_globalmat[start:end].clone()

        humanoid_globalmat, obj_globalmat = aztoay_hoi(humanoid_globalmat, obj_globalmat)
        # modify root position
        humanoid_localmat[:, 0] = humanoid_globalmat[:, 0]
        obj_localmat[:, 0] = obj_globalmat[:, 0]

        angles = self.motion_files[vid]["angles"]
        seq_angles = angles[start:end].clone()
        contact = self.motion_files[vid]["contact"]
        seq_contact = []
        for k in [
            "L_Index3",
            "L_Middle3",
            "L_Pinky3",
            "L_Ring3",
            "L_Thumb3",
            "R_Index3",
            "R_Middle3",
            "R_Pinky3",
            "R_Ring3",
            "R_Thumb3",
        ]:
            c = contact[k][start:end].clone()
            seq_contact.append(c)
        seq_contact = torch.stack(seq_contact, dim=-1)  # (N, 10)

        beta = self.motion_files[vid]["beta"]
        seq_beta = beta[start:end].clone()

        # all seqs larger than 300 frames, no need to pad

        data = {
            "humanoid_localmat": humanoid_localmat,
            "humanoid_globalmat": humanoid_globalmat,
            "obj_localmat": obj_localmat,
            "obj_globalmat": obj_globalmat,
            "contact": seq_contact,
            "angles": seq_angles,
            "beta": seq_beta,
        }

        if vid in self.bps_data.keys():
            basis_point = self.bps_data["basis_point"]
            bps_top = self.bps_data[vid]["top_bps"]
            bps_bottom = self.bps_data[vid]["bottom_bps"]
            finger_dist = self.bps_data[vid]["finger_dist"]
            scale = self.bps_data[vid]["scale"]
            center = self.bps_data[vid]["center"]
            data["basis_point"] = basis_point
            data["bps_top"] = bps_top[start:end].clone()
            data["bps_bottom"] = bps_bottom[start:end].clone()
            data["finger_dist"] = finger_dist[start:end].clone()
            data["scale"] = scale
            data["center"] = center

        return data

    def _process_data(self, data, idx):
        length = data["humanoid_globalmat"].shape[0]
        beta = data["beta"]
        mask = torch.ones((length, 1), dtype=torch.bool)  # always supervise full
        humanoid_localmat = data["humanoid_localmat"]
        smplx_localmat = humanoid_localmat[:, HUMANOID2SMPLX]

        local_skeleton = self.smplx.get_local_skeleton_with_finger(beta[:1])
        root_0 = local_skeleton[:, 0]  # (1, 3)
        root_transl = matrix.get_position(smplx_localmat[:, 0])  # (N, 3)
        transl = root_transl - root_0

        global_orient_rotmat = matrix.get_rotation(smplx_localmat[:, 0])  # (N, 3, 3)
        global_orient = matrix.matrix_to_axis_angle(global_orient_rotmat)  # (N, 3)

        body_pose_rotmat = matrix.get_rotation(smplx_localmat[:, 1:22])  # (N, 21, 3, 3)
        body_pose = matrix.matrix_to_axis_angle(body_pose_rotmat)  # (N, 21, 3)
        left_hand_rotmat = matrix.get_rotation(smplx_localmat[:, 22:37])  # (N, 15, 3, 3)
        left_hand_pose = matrix.matrix_to_axis_angle(left_hand_rotmat)  # (N, 15, 3)
        right_hand_rotmat = matrix.get_rotation(smplx_localmat[:, 37:52])  # (N, 15, 3, 3)
        right_hand_pose = matrix.matrix_to_axis_angle(right_hand_rotmat)  # (N, 15, 3)

        local_pos = matrix.get_position(smplx_localmat)  # (N, 52, 3)
        local_skeleton = local_pos[:1, 1:]  # (1, 51, 3)
        local_skeleton = torch.cat([root_0[None], local_skeleton], dim=1)  # (1, 52, 3)
        return_data = {
            "length": length,
            "mask": mask,
        }
        return_data["gt_global_pos"] = matrix.get_position(data["humanoid_globalmat"])[:, HUMANOID2SMPLX]
        return_data["transl"] = transl
        return_data["global_orient"] = global_orient
        return_data["body_pose"] = body_pose.flatten(-2)
        return_data["left_hand_pose"] = left_hand_pose.flatten(-2)
        return_data["right_hand_pose"] = right_hand_pose.flatten(-2)
        return_data["skeleton"] = local_skeleton
        return_data["beta"] = beta

        left_handpose = get_handpose(humanoid_localmat, is_right=False)
        left_base_rotmat = matrix.get_rotation(data["humanoid_globalmat"][:, 17])
        left_base_pos = matrix.get_position(data["humanoid_globalmat"][:, 17])
        left_hand_localpos = matrix.get_position(left_handpose)
        return_data["left_base_rotmat"] = left_base_rotmat
        return_data["left_base_pos"] = left_base_pos
        return_data["left_handpose"] = left_handpose
        return_data["left_hand_pose"] = matrix.matrix_to_axis_angle(matrix.get_rotation(left_handpose)).flatten(-2)
        return_data["left_hand_localpos"] = left_hand_localpos
        handpose_tip = get_handpose(humanoid_localmat, is_right=False, istip=True)
        handtip_localpos = matrix.get_position(handpose_tip)
        return_data["left_handtip_localpos"] = handtip_localpos

        right_handpose = get_handpose(humanoid_localmat, is_right=True)
        right_base_rotmat = matrix.get_rotation(data["humanoid_globalmat"][:, 41])
        right_base_pos = matrix.get_position(data["humanoid_globalmat"][:, 41])
        righthand_localpos = matrix.get_position(right_handpose)
        return_data["right_base_rotmat"] = right_base_rotmat
        return_data["right_base_pos"] = right_base_pos
        return_data["right_handpose"] = right_handpose
        return_data["right_hand_pose"] = matrix.matrix_to_axis_angle(matrix.get_rotation(right_handpose)).flatten(-2)
        return_data["right_hand_localpos"] = righthand_localpos
        handpose_tip = get_handpose(humanoid_localmat, is_right=True, istip=True)
        handtip_localpos = matrix.get_position(handpose_tip)
        return_data["right_handtip_localpos"] = handtip_localpos

        return_data["obj_angles"] = data["angles"]
        obj_transl = matrix.get_position(data["obj_globalmat"][:, 0])
        obj_global_orient = matrix.matrix_to_axis_angle(matrix.get_rotation(data["obj_globalmat"][:, 0]))
        return_data["obj_transl"] = obj_transl
        return_data["obj_global_orient"] = obj_global_orient

        return_data["humanoid"] = data["humanoid_globalmat"]
        return_data["obj"] = data["obj_globalmat"]
        return_data["contact"] = data["contact"]
        return_data["angles"] = data["angles"]

        return_data["basis_point"] = data["basis_point"]
        return_data["bps"] = torch.cat([data["bps_top"], data["bps_bottom"]], dim=-1)
        return_data["finger_dist"] = data["finger_dist"]
        return_data["scale"] = data["scale"].reshape(1)
        return_data["center"] = data["center"]
        return_data["global_obj_center"] = matrix.get_position_from(data["center"][None], data["obj_globalmat"][:, 0]) # (N, 3)

        vid = self.idx2meta[idx]
        obj_name = vid.split("_")[1]
        action = vid.split("_")[2]
        caption = f"a person {action}s a {obj_name}"
        # caption = ""
        return_data["caption"] = caption
        return return_data

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.idx2meta)

    def __getitem__(self, item):
        idx = item
        data = self._load_data(idx)
        processed_data = self._process_data(data, idx)

        human_motion = processed_data["gt_global_pos"]
        obj_motion = processed_data["global_obj_center"]
        obj_rot = matrix.matrix_to_rotation_6d(matrix.get_rotation(processed_data["obj"]))
        obj_center_mat = matrix.get_TRS(matrix.get_rotation(processed_data["obj"])[..., 0, :, :], obj_motion)
        obj_name = self.idx2meta[idx].split("_")[1]
        m_length = processed_data["length"]
        text = processed_data["caption"]

        obj_invhumanmotion = matrix.get_relative_position_to(human_motion, obj_center_mat)
        motion = torch.cat([human_motion.flatten(-2), obj_invhumanmotion.flatten(-2)], dim=-1)
        motion = motion.numpy()

        m_length = min(m_length, self.max_motion_length)

        # Randomly select a caption
        caption = text
        caption = caption.replace('/', ' ')
        tokens = self.token_model(caption)
        token_format = " ".join([f"{token.text}/{token.pos_}" for token in tokens])
        tokens = token_format.split(" ")

        filter_tokens = []
        for token in tokens:
            try: 
                word_emb, pos_oh = self.w_vectorizer[token]
            except Exception as e:
                continue
            filter_tokens.append(token)
        tokens = filter_tokens

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        if len(motion) > m_length:
            idx = random.randint(0, len(motion) - m_length)
        else:
            idx = 0
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


'''For use of training baseline'''
class Text2MotionDatasetBaseline(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'
            if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                m_length = self.max_length
                s_idx = random.randint(0, m_length - self.max_length)
            else:
                if coin2 == 'single':
                    n_m_length = self.max_length + self.opt.unit_length * len_gap
                else:
                    n_m_length = self.max_length + self.opt.unit_length * (len_gap - 1)
                s_idx = random.randint(0, m_length - n_m_length)
                m_length = n_m_length
        else:
            s_idx = 0

        src_motion = motion[s_idx: s_idx + m_length]
        tgt_motion = motion[s_idx: s_idx + self.max_length]

        "Z Normalization"
        src_motion = (src_motion - self.mean) / self.std
        tgt_motion = (tgt_motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            src_motion = np.concatenate([src_motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(m_length, src_motion.shape, tgt_motion.shape)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, caption, sent_len, src_motion, tgt_motion, m_length


class MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx+self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class RawTextDataset(data.Dataset):
    def __init__(self, opt, mean, std, text_file, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.nlp = spacy.load('en_core_web_sm')

        with cs.open(text_file) as f:
            for line in f.readlines():
                word_list, pos_list = self.process_text(line.strip())
                tokens = ['%s/%s'%(word_list[i], pos_list[i]) for i in range(len(word_list))]
                self.data_dict.append({'caption':line.strip(), "tokens":tokens})

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))


    def process_text(self, sentence):
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens = data['caption'], data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len