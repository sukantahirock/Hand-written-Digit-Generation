import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
import io

# --- Model Definition (must be identical to the one used for training) ---
latent_dim = 100
image_size = 28 * 28

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, image_size),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1, 28, 28)

# --- Function to load the model ---
@st.cache_resource
def load_generator_model(model_path):
    generator = Generator()
    generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    generator.eval() # Set to evaluation mode
    return generator

# --- Load your trained model ---
# Make sure 'generator_mnist_gan.pth' is in the same directory as this script
# Or provide the full path to your model file.
MODEL_PATH = 'generator_mnist_gan.pth'
try:
    generator_model = load_generator_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it's in the correct directory.")
    st.stop() # Stop the app if the model isn't found
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# --- Function to generate and display images ---
def generate_and_display_images(model, digit, num_images=5):
    # For a Conditional GAN, you would also pass the 'digit' here
    # For this unconditional GAN, we'll just generate random digits
    # and hope it's the one requested. A conditional GAN is strongly recommended.

    # If you implemented a Conditional GAN in training:
    # noise = torch.randn(num_images, latent_dim)
    # label_one_hot = torch.zeros(num_images, 10)
    # label_one_hot[range(num_images), digit] = 1 # Set the specific digit
    # combined_input = torch.cat((noise, label_one_hot), 1) # Concatenate noise and label

    # For now, with the unconditional GAN:
    noise = torch.randn(num_images, latent_dim)

    with torch.no_grad():
        generated_images_tensor = model(noise)

    # Normalize images to [0, 1] for display if using Tanh output
    generated_images_tensor = (generated_images_tensor + 1) / 2

    # Create a grid of images
    grid = make_grid(generated_images_tensor, nrow=num_images, padding=2, normalize=True)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr).convert('RGB')

    return im

# --- Streamlit UI ---
st.title("Handwritten Digit Generation Web App")

st.write("Welcome! This application generates images of handwritten digits.")

# User selects a digit
selected_digit = st.selectbox(
    "Select a digit (0-9) to generate:",
    options=list(range(10))
)

if st.button("Generate Images"):
    st.subheader(f"Generating 5 images of digit {selected_digit}...")

    # Call the image generation function
    # NOTE: With the current UNCONDITIONAL GAN, the 'selected_digit'
    # is not actually used to condition the generation.
    # You MUST implement a Conditional GAN for this feature to work as intended.
    generated_image_grid = generate_and_display_images(generator_model, selected_digit, num_images=5)

    # Display the generated images
    st.image(generated_image_grid, caption=f"Generated Images of Digit {selected_digit}", use_column_width=True)

st.markdown("---")
st.markdown("### How it works:")
st.markdown("- Uses a trained Generative Adversarial Network (GAN) model.")
st.markdown("- The model was trained from scratch on the MNIST dataset using PyTorch.")
st.markdown("- Images are generated on the fly when you click 'Generate Images'.")