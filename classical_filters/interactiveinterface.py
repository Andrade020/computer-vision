import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
#############################################################################################

# =============================================================================
# imge procss functn
# =============================================================================
######################################################################

def adjust_brightness_contrast(image, beta, k):
    """
    adjsts the bright (beta) and contrs (k) of the imge.
#########################################################################################

    parmtr:
        imge: numpy arry imge in bgr formt.
        beta: addtv vale for bright.
        k: multpl factr for contrs.
###########################################################################

    retrns:
        adjstd imge.
    """
    image_float = image.astype(np.float32)
    adjusted = np.clip(k * image_float + beta, 0, 255).astype(np.uint8)
    return adjusted
##################################################################

def convolution_filter(image, kernel):
    """
    appls a manl convlt filtr to an imge.
    if the imge is in colr, it is first convrt to graysc.
######################################################################

    parmtr:
        imge: inpt imge (bgr or graysc).
        kernl: filtr matrx.
########################################################################

    retrns:
        filtrd imge (convrt back to bgr for disply).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    img = gray.astype(np.float32) / 255.0
    height, width = img.shape
    k_height, k_width = kernel.shape
    pad_y = k_height // 2
    pad_x = k_width // 2
    result = np.zeros_like(img)
    for y in range(pad_y, height - pad_y):
        for x in range(pad_x, width - pad_x):
            region = img[y - pad_y:y + pad_y + 1, x - pad_x:x + pad_x + 1]
            result[y, x] = np.sum(region * kernel)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
##############################################################################

def add_gaussian_noise(image, std_dev):
    """
    adds gaussn nose to the imge.
###############################################################################################

    parmtr:
        imge: inpt imge.
        std_dv: standr devtn of the nose.
######################################################################################

    retrns:
        nosy imge.
    """
    noise = np.random.normal(0, std_dev, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)
######################################################################################

def kuwahara_filter(image, window_size):
    """
    appls the kuwhr filtr to the imge.
##########################################################################################

    parmtr:
        imge: inpt imge in bgr formt.
        windw_: windw size (shld be odd).
##############################################################################################

    retrns:
        filtrd imge.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].astype(float)
    result = np.zeros_like(image)
    half = window_size // 2
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            tl_x = max(x - half, 0)
            tl_y = max(y - half, 0)
            br_x = min(x + half, image.shape[1] - 1)
            br_y = min(y + half, image.shape[0] - 1)
            # defne the 4 quadrn
            q1 = brightness[tl_y:y+1, tl_x:x+1]
            q2 = brightness[tl_y:y+1, x:br_x+1]
            q3 = brightness[y:br_y+1, tl_x:x+1]
            q4 = brightness[y:br_y+1, x:br_x+1]
            quadrants = [q1, q2, q3, q4]
            stds = [np.std(q) for q in quadrants]
            idx = np.argmin(stds)
            region = image[tl_y:br_y+1, tl_x:br_x+1]
            avg_color = np.mean(region.reshape(-1, 3), axis=0)
            result[y, x] = avg_color
    return result.astype(np.uint8)
#################################################################################################

def resize_image(image, max_dim):
    """
    reszs the imge so that its largst dimnsn is max_dm pixls,
    presrv the aspct rato.
    """
    height, width = image.shape[:2]
    scaling = max_dim / float(max(height, width))
    if scaling < 1.0:
        new_size = (int(width * scaling), int(height * scaling))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image
#####################################################################################

# =============================================================================
# graphc intrfc with side panl buttns
# =============================================================================
######################################################################

class ImageProcessingInterface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Processing Interface")
        self.geometry("1000x700")
#####################################################################################
        
        self.original_image = None  # lodd imge
###############################################################################
        
        # left side panl for buttns
        self.left_frame = tk.Frame(self, width=200, bg="#e0e0e0")
        self.left_frame.pack(side="left", fill="y")
####################################################################################################
        
        # main panl for disply effcts
        self.right_frame = tk.Frame(self, bg="white")
        self.right_frame.pack(side="right", fill="both", expand=True)
######################################################################
        
        # dictnr to hold panls for each effct
        self.panels = {}
        self.create_panels()
        self.create_menu_buttons()
###############################################################################################
        
        # initll show the load panl
        self.show_panel("load")
#########################################################################################
    
    def create_menu_buttons(self):
        """crts side buttns to selct effcts."""
        buttons = [
            ("Load Image", "load"),
            ("Brightness/Contrast", "adjust"),
            ("Convolution", "conv"),
            ("Gaussian Noise", "noise"),
            ("Kuwahara Filter", "kuwahara")
        ]
        for text, key in buttons:
            btn = tk.Button(self.left_frame, text=text, command=lambda k=key: self.show_panel(k),
                            relief="raised", padx=10, pady=5)
            btn.pack(fill="x", padx=10, pady=10)
########################################################################
    
    def create_panels(self):
        """crts panls for each effct and strs them in self.panls."""
        # panl for lodng imge
        panel_load = tk.Frame(self.right_frame, bg="white")
        btn_select = tk.Button(panel_load, text="Select Image", command=self.load_image)
        btn_select.pack(pady=20)
        self.label_image_load = tk.Label(panel_load, bg="white")
        self.label_image_load.pack()
        self.panels["load"] = panel_load
#############################################################
        
        # panl for bright/contrs
        panel_adjust = tk.Frame(self.right_frame, bg="white")
        tk.Label(panel_adjust, text="Brightness/Contrast Adjustment", font=("Arial", 14), bg="white").pack(pady=5)
        frm_adjust = tk.Frame(panel_adjust, bg="white")
        frm_adjust.pack(pady=10)
        tk.Label(frm_adjust, text="Beta (Brightness):", bg="white").grid(row=0, column=0, padx=5, pady=5)
        self.scale_beta = tk.Scale(frm_adjust, from_=-100, to=100, orient="horizontal", bg="white")
        self.scale_beta.set(0)
        self.scale_beta.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(frm_adjust, text="K (Contrast):", bg="white").grid(row=1, column=0, padx=5, pady=5)
        self.scale_k = tk.Scale(frm_adjust, from_=0.0, to=3.0, resolution=0.1, orient="horizontal", bg="white")
        self.scale_k.set(1.0)
        self.scale_k.grid(row=1, column=1, padx=5, pady=5)
        tk.Button(panel_adjust, text="Apply", command=self.apply_adjust).pack(pady=10)
        self.label_adjust_result = tk.Label(panel_adjust, bg="white")
        self.label_adjust_result.pack(pady=10)
        self.panels["adjust"] = panel_adjust
###################################################################################
        
        # panl for convlt
        panel_conv = tk.Frame(self.right_frame, bg="white")
        tk.Label(panel_conv, text="Convolution Filter", font=("Arial", 14), bg="white").pack(pady=5)
        frm_conv = tk.Frame(panel_conv, bg="white")
        frm_conv.pack(pady=10)
        tk.Label(frm_conv, text="Select Filter:", bg="white").grid(row=0, column=0, padx=5, pady=5)
        self.combo_kernel = ttk.Combobox(frm_conv, values=[
            "Blur 3x3", 
            "Horizontal Derivative", 
            "Vertical Derivative", 
            "Sobel Horizontal", 
            "Sobel Vertical"
        ])
        self.combo_kernel.current(0)
        self.combo_kernel.grid(row=0, column=1, padx=5, pady=5)
        tk.Button(panel_conv, text="Apply", command=self.apply_convolution).pack(pady=10)
        self.label_conv_result = tk.Label(panel_conv, bg="white")
        self.label_conv_result.pack(pady=10)
        self.panels["conv"] = panel_conv
#########################################################################
        
        # panl for gaussn nose
        panel_noise = tk.Frame(self.right_frame, bg="white")
        tk.Label(panel_noise, text="Add Gaussian Noise", font=("Arial", 14), bg="white").pack(pady=5)
        frm_noise = tk.Frame(panel_noise, bg="white")
        frm_noise.pack(pady=10)
        tk.Label(frm_noise, text="Standard Deviation:", bg="white").grid(row=0, column=0, padx=5, pady=5)
        self.scale_noise = tk.Scale(frm_noise, from_=0, to=100, orient="horizontal", bg="white")
        self.scale_noise.set(20)
        self.scale_noise.grid(row=0, column=1, padx=5, pady=5)
        tk.Button(panel_noise, text="Apply", command=self.apply_noise).pack(pady=10)
        self.label_noise_result = tk.Label(panel_noise, bg="white")
        self.label_noise_result.pack(pady=10)
        self.panels["noise"] = panel_noise
###################################################################################
        
        # panl for kuwhr filtr
        panel_kuwahara = tk.Frame(self.right_frame, bg="white")
        tk.Label(panel_kuwahara, text="Kuwahara Filter", font=("Arial", 14), bg="white").pack(pady=5)
        frm_kuwahara = tk.Frame(panel_kuwahara, bg="white")
        frm_kuwahara.pack(pady=10)
        tk.Label(frm_kuwahara, text="Window Size:", bg="white").grid(row=0, column=0, padx=5, pady=5)
        self.scale_kuwahara = tk.Scale(frm_kuwahara, from_=3, to=15, orient="horizontal", bg="white",
                                       command=self.check_kuwahara_value)
        self.scale_kuwahara.set(3)
        self.scale_kuwahara.grid(row=0, column=1, padx=5, pady=5)
        # warnng labl (initll empty)
        self.label_warning = tk.Label(frm_kuwahara, text="", fg="red", bg="white")
        self.label_warning.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        tk.Button(panel_kuwahara, text="Apply", command=self.apply_kuwahara).pack(pady=10)
        self.label_kuwahara_result = tk.Label(panel_kuwahara, bg="white")
        self.label_kuwahara_result.pack(pady=10)
        self.panels["kuwahara"] = panel_kuwahara
###############################################################################################
        
        # plce all panls in the same area and hide them
        for panel in self.panels.values():
            panel.place(relx=0, rely=0, relwidth=1, relheight=1)
##########################################################################################
    
    def show_panel(self, key):
        """rass the panl corrsp to the givn key."""
        panel = self.panels.get(key)
        if panel:
            panel.tkraise()
##################################################################################
    
    def load_image(self):
        """opns a file dilg to selct an imge and disply the lodd imge."""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.original_image = cv2.imread(file_path)
            # resze imge to sped up procss
            self.original_image = resize_image(self.original_image, 400)
            self.display_image(self.label_image_load, self.original_image)
#######################################################################
    
    def display_image(self, label, image):
        """
        convrt imge from bgr to rgb and disply it in a labl.
########################################################################

        parmtr:
            labl: widgt whre the imge will be shwn.
            imge: imge to be disply.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(image_rgb)
        im_pil.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(im_pil)
        label.config(image=photo)
        label.image = photo
############################################################################################
    
    def apply_adjust(self):
        """appls bright/contrs adjstm and disply the reslt."""
        if self.original_image is None:
            return
        beta = self.scale_beta.get()
        k = self.scale_k.get()
        adjusted = adjust_brightness_contrast(self.original_image, beta, k)
        self.display_image(self.label_adjust_result, adjusted)
#######################################################################
    
    def apply_convolution(self):
        """appls the selctd convlt filtr and disply the reslt."""
        if self.original_image is None:
            return
        selection = self.combo_kernel.get()
        if selection == "Blur 3x3":
            kernel = np.ones((3, 3), dtype=np.float32) / 9.0
        elif selection == "Horizontal Derivative":
            kernel = np.array([[1, 0, -1]], dtype=np.float32)
        elif selection == "Vertical Derivative":
            kernel = np.array([[1], [0], [-1]], dtype=np.float32)
        elif selection == "Sobel Horizontal":
            kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float32)
        elif selection == "Sobel Vertical":
            kernel = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]], dtype=np.float32)
        else:
            kernel = np.ones((3, 3), dtype=np.float32) / 9.0
        result = convolution_filter(self.original_image, kernel)
        self.display_image(self.label_conv_result, result)
################################################################################
    
    def apply_noise(self):
        """adds gaussn nose and disply the reslt."""
        if self.original_image is None:
            return
        std_dev = self.scale_noise.get()
        noisy = add_gaussian_noise(self.original_image, std_dev)
        self.display_image(self.label_noise_result, noisy)
##############################################################################
    
    def apply_kuwahara(self):
        """appls the kuwhr filtr and disply the reslt."""
        if self.original_image is None:
            return
        window_size = int(self.scale_kuwahara.get())
        # ensre the windw size is odd
        if window_size % 2 == 0:
            window_size += 1
        result = kuwahara_filter(self.original_image, window_size)
        self.display_image(self.label_kuwahara_result, result)
###########################################################################################
    
    def check_kuwahara_value(self, val):
        """
        chcks the sldr vale for the kuwhr filtr and disply a warnng if it is high.
##################################################################################################

        parmtr:
            val: currnt vale of the sldr.
        """
        try:
            value = int(float(val))
            threshold = 9
            if value > threshold:
                self.label_warning.config(text="This value may cause slow processing!")
            else:
                self.label_warning.config(text="")
        except Exception:
            self.label_warning.config(text="")
#################################################################

# =============================================================================
# run the intrfc
# =============================================================================
#############################################################################################

if __name__ == "__main__":
    app = ImageProcessingInterface()
    app.mainloop()