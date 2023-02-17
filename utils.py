import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt


# use extract_patches_2d of sklearn to extract image patches
def extract_patches(noisy_img: np.ndarray, patch_size: int) -> np.ndarray:
    
    dataset = extract_patches_2d(noisy_img, (patch_size, patch_size))
    dataset = dataset.reshape(-1, patch_size*patch_size).T
    
    return dataset

# write a function to reconstruct the denoised image from its vectorizedd patches
def reconstruct_image(final_dictionary, X, noisy_img, p: int, m: float):
    
    O = np.zeros(noisy_img.shape)
    C = np.zeros(noisy_img.shape)
    Y_hat = final_dictionary @ X
    
    k = 0
    
    r, c = noisy_img.shape
    
    for i in range(r-p+1):
        for j in range(c-p+1):
            O[i:i+p, j:j+p] += Y_hat[:, k].reshape(p, p)
            C[i:i+p, j:j+p] += np.ones((p, p))
            k += 1
        
    denoised_img = (O + m * noisy_img) / (C + m)
    
    return denoised_img


def visualize_dict(dictionary: np.ndarray):
    M = dictionary.copy()
    n, k = dictionary.shape
    # normalize dictionary columns
    for i in range(k):
        M[:, i] -= np.min(M[:, i])
        if np.max(M[:, i]):
            M[:, i] /= np.max(dictionary[:, i])
    
        n_r = int(np.sqrt(n))
        k_r = int(np.sqrt(k))

        dim = n_r * k_r + k_r + 1
        V = np.ones((dim, dim)) * np.min(dictionary)

        # compute the patches
        patches = [np.reshape(M[:, i], (n_r, n_r)) for i in range(k)]

        # place patches
        for i in range(k_r):
            for j in range(k_r):
                V[j * n_r + 1 + j:(j + 1) * n_r + 1 + j, i * n_r + 1 + i:(i + 1) * n_r + 1 + i] = patches[
                    i * k_r + j]
            
    return V


def calculate_PSNR(image1, image2):
    mse = np.sum((image1-image2)**2)/np.prod(image1.shape)
    return 10*np.log10((255**2)/mse)


# train the dictionary
def train(Y, initial_dict, num_iter=10, C=1.5, sigma=20, sparse_code=None, dict_update=None):
    num_iter = num_iter
    Y = Y.copy()
    D = initial_dict.copy()
    for _ in tqdm(range(num_iter)):
        X = sparse_code(D, Y, C*sigma)
        D = dict_update(Y, X)
    
    return D, X


# Evaluate denoising algorithm
def evaluate(clean_img, noisy_img, denoised_img):
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(denoised_img, cmap='gray')
    psnr = calculate_PSNR(clean_img, denoised_img)
    plt.title(f'Denoised Image, PSNR: {np.round(psnr, 2)}', size=12)
    plt.subplot(1, 3, 2)
    plt.imshow(noisy_img, cmap='gray')
    psnr = calculate_PSNR(clean_img, noisy_img)
    plt.title(f'Noisy Image, PSNR: {np.round(psnr, 2)}', size=12);
    plt.subplot(1, 3, 3)
    plt.imshow(clean_img, cmap='gray')
    plt.title('Original Image');
