import os
import pandas as pd
import numpy as np
import pickle
import cv2
import tqdm


def train_valid_split(data_dir: str='/content/camera_data', meta_path: str='train_meta.csv', valid_type: int=1, full_train: bool=False) -> Tuple[List[str]]:
    """
    Example:
        train_input_paths, train_label_paths, valid_input_paths, valid_label_paths = \
            train_valid_split(data_dir='/content/', meta_path='train_meta.csv', valid_type=1, full_train=True)
    Args:
        data_dir (str): 학습 데이터 디렉토리 경로
        meta_path (str): 메타파일 경로. Defaults to 'train_meta.csv'.
        valid_type (int):
            - 1: 스프레드시트를 통해 결정한 validation set
            - 0: 기존 validation set(10000~10059 이미지로 검증)
        full_train (bool):
            - True: valid_type에 관계 없이 모든 데이터를 학습에 활용
    Returns:
        Tuple[List[str]]: [학습IMG경로], [학습GT경로], [검증IMG경로], [검증GT경로]
    """
    assert os.path.isfile(meta_path), f"'{meta_path}' not found"
    assert os.path.isdir(data_dir), f"'{data_dir}' is not a directory"
    meta = pd.read_csv(meta_path)

    # align data path
    meta['input_img'] = meta['input_img'].apply(lambda x: os.path.join(data_dir, 'train_input_img', x))
    meta['label_img'] = meta['label_img'].apply(lambda x: os.path.join(data_dir, 'train_label_img', x))

    # split train & valid
    if full_train:
        train_input_paths = meta['input_img'].tolist()
        train_label_paths = meta['label_img'].tolist()
    else:
        train_input_paths = meta[meta[f'valid_type{valid_type}']=='train']['input_img'].tolist()
        train_label_paths = meta[meta[f'valid_type{valid_type}']=='train']['label_img'].tolist()
    valid_input_paths = meta[meta[f'valid_type{valid_type}']=='valid']['input_img'].tolist()
    valid_label_paths = meta[meta[f'valid_type{valid_type}']=='valid']['label_img'].tolist()
    return train_input_paths, train_label_paths, valid_input_paths, valid_label_paths


def save_pickle(path: str, f: object) -> None:
    with open(path, "wb") as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
def make_img_square(img: np.array):
    h, w, _ = img.shape

    # padding to make image square
    if h < w and (h - w) % 2 == 0:
        margin = (w - h) // 2
        lower_pad = img[::-1, :, :][:margin]
        upper_pad = img[::-1, :, :][h-margin:]
        img = np.vstack([upper_pad, img, lower_pad])
        
    return img

LARGE = 3264
SMALL = 1632

def preprocess_pixel_shift(img_path_list, save_path, img_size):
    os.makedirs(f'{save_path}{img_size}', exist_ok=True)
    num = 0
    for path in tqdm(img_path_list):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = make_img_square(img)
        #recover = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        shift_factor = 6 if img.shape[0] == LARGE else 3
        shifts = [(i, j) for i in range(shift_factor) for j in range(shift_factor)]

        for w, h in shifts:
            tmp = img[:, w::shift_factor, :]
            tmp = tmp[h::shift_factor, :, :]
            #recover[h::shift_factor, w::shift_factor,:] = tmp
            pkl_save_path = f'{save_path}{img_size}/{num}.pickle'
            save_pickle(pkl_save_path, tmp)
            num+=1
        
        # Recover image
        #recover_h = (img.shape[0]//4)*3 
        #recover = recover[(img.shape[0]-recover_h)//2:(img.shape[0]+recover_h)//2, :, :]
        
    return num


def preprocess_sliding_windows(img_path_list, save_path, stride, img_size):
    os.makedirs(f'{save_path}{img_size}', exist_ok=True)
    num = 0
    for path in tqdm(img_path_list):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for top in range(0, img.shape[0], stride):
            for left in range(0, img.shape[1], stride):
                piece = np.zeros([img_size, img_size, 3], np.uint8)
                temp = img[top:top+img_size, left:left+img_size, :]
                if temp.shape[0] < img_size or temp.shape[1] < img_size:
                    y_gap = (img_size-temp.shape[0])
                    x_gap = (img_size-temp.shape[1])
                    temp = img[top-y_gap:top-y_gap+img_size,left-x_gap:left-x_gap+img_size,:]
                piece[:temp.shape[0], :temp.shape[1], :] = temp
                np.save(f'{save_path}{img_size}/{num}.npy', piece)
                num+=1


