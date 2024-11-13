import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

class GrayscaleImageCompressor:
    def __init__(self, img_matrix):
        self.img_matrix = img_matrix
        
    def compress_image(self, k):
        U, S, VT = np.linalg.svd(self.img_matrix, full_matrices=False)
        U_k = U[:, :k]
        S_k = np.diag(S[:k])
        VT_k = VT[:k, :]
        compressed_img = np.dot(U_k, np.dot(S_k, VT_k))
        return compressed_img

class ColorImageCompressor:
    def __init__(self, img_matrix):
        self.img_matrix = img_matrix
        
    def compress_channel(self, channel_matrix, k):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        channel_tensor = torch.tensor(channel_matrix, dtype=torch.float32, device=device)
        U, S, V = torch.svd(channel_tensor)
        U_k = U[:, :k]
        S_k = torch.diag(S[:k])
        V_k = V[:, :k]
        compressed_channel_tensor = torch.mm(U_k, torch.mm(S_k, V_k.t()))
        compressed_channel = compressed_channel_tensor.cpu().numpy()
        return compressed_channel

    def compress_image(self, k):
        Y, Cb, Cr = self.img_matrix[:, :, 0], self.img_matrix[:, :, 1], self.img_matrix[:, :, 2]
        Y_compressed = self.compress_channel(Y, k)
        Cb_compressed = self.compress_channel(Cb, k)
        Cr_compressed = self.compress_channel(Cr, k)
        compressed_img = np.stack((Y_compressed, Cb_compressed, Cr_compressed), axis=2).astype(np.uint8)
        return compressed_img

    def ycbcr_to_rgb(self, ycbcr_img):
        img = Image.fromarray(ycbcr_img, 'YCbCr')
        rgb_img = img.convert('RGB')
        return np.array(rgb_img)

def display_images(original_img, compressed_img):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_img, cmap='gray' if len(original_img.shape) == 2 else None)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(compressed_img, cmap='gray' if len(compressed_img.shape) == 2 else None)
    axes[1].set_title('Compressed Image')
    axes[1].axis('off')
    st.pyplot(fig)

def main():
    st.title("Image Compression App")

    # File upload
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        original_img = np.array(Image.open(uploaded_file))

        # Compression mode selection
        compression_mode = st.selectbox("Choose compression mode", ['Grayscale', 'Color'])

        # Compression level slider
        compression_level = st.slider("Compression Level (k)", 1, min(original_img.shape), 50)

        if st.button("Compress Image"):
            if compression_mode == 'Grayscale':
                compressor = GrayscaleImageCompressor(original_img)
                compressed_img = compressor.compress_image(compression_level)
            else:
                compressor = ColorImageCompressor(original_img)
                compressed_img = compressor.compress_image(compression_level)
            display_images(original_img, compressed_img)

            # Download button
            st.download_button(
                label="Download Compressed Image",
                data=Image.fromarray(compressed_img),
                file_name="compressed_image.png",
                mime="image/png",
            )

if __name__ == "__main__":
    main()