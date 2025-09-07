import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import ipywidgets as widgets
from IPython.display import display, clear_output

# === Step 1: Upload image ===
uploaded = files.upload()
filename = list(uploaded.keys())[0]

# Load image (color + gray)
img_color = cv2.imread(filename)
img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)

# Histogram Equalization (fixed)
hist_eq = cv2.equalizeHist(img_gray)

# --- Function to update display with gamma ---
def update_dashboard(gamma):
    gamma = float(gamma)
    gamma_corrected = np.array(255 * (img_gray / 255.0) ** gamma, dtype='uint8')

    # Clean output
    clear_output(wait=True)

    # Dashboard Layout (2 rows × 4 columns)
    fig = plt.figure(figsize=(18,9))
    grid = fig.add_gridspec(2, 4, height_ratios=[2,1])  # Top = images, Bottom = histograms

    # === Row 1: Images ===
    ax1 = fig.add_subplot(grid[0,0])
    ax1.imshow(img_color)
    ax1.set_title("Original RGB", fontsize=12, fontweight="bold")
    ax1.axis('off')

    ax2 = fig.add_subplot(grid[0,1])
    ax2.imshow(img_gray, cmap='gray')
    ax2.set_title("Original Gray", fontsize=12, fontweight="bold")
    ax2.axis('off')

    ax3 = fig.add_subplot(grid[0,2])
    ax3.imshow(hist_eq, cmap='gray')
    ax3.set_title("Histogram Equalized", fontsize=12, fontweight="bold")
    ax3.axis('off')

    ax4 = fig.add_subplot(grid[0,3])
    ax4.imshow(gamma_corrected, cmap='gray')
    ax4.set_title(f"Gamma Corrected (γ={gamma:.2f})", fontsize=12, fontweight="bold")
    ax4.axis('off')

    # === Row 2: Histograms (Gray only, not RGB) ===
    ax5 = fig.add_subplot(grid[1,1])
    ax5.hist(img_gray.ravel(), bins=256, range=(0,256), color='black')
    ax5.set_title("Original Gray Histogram", fontsize=11)

    ax6 = fig.add_subplot(grid[1,2])
    ax6.hist(hist_eq.ravel(), bins=256, range=(0,256), color='black')
    ax6.set_title("Equalized Histogram", fontsize=11)

    ax7 = fig.add_subplot(grid[1,3])
    ax7.hist(gamma_corrected.ravel(), bins=256, range=(0,256), color='black')
    ax7.set_title(f"Gamma Histogram (γ={gamma:.2f})", fontsize=11)

    plt.suptitle("Underwater Image Enhancement Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Show slider + run button *after* plots
    display(control_box)

# === Step 2: Create Slider + Run Button ===
gamma_slider = widgets.FloatSlider(
    value=0.6, min=0.1, max=2.0, step=0.1,
    description="Gamma (γ):",
    continuous_update=False,
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='60%')
)

run_button = widgets.Button(description="Run Enhancement", button_style='success')

def on_button_click(b):
    update_dashboard(gamma_slider.value)

run_button.on_click(on_button_click)

# Combine slider + button in a clean box
control_box = widgets.VBox([gamma_slider, run_button])

# Show interface for the first time
display(control_box)
