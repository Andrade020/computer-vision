"""Vectorized classical image-filter building blocks used by both the CLI
(imgfilter.py) and the GUI (filters_gui.py) in this project.

Submodules:
    io           -- load_image / save_image, with real error handling.
    pointops     -- brightness/contrast, Gaussian noise, aspect-preserving
                    resize (already vectorized in the original app).
    convolution  -- 2D convolution via cv2.filter2D + the 5 kernel presets
                    from the original combobox.
    kuwahara     -- vectorized Kuwahara edge-preserving filter.
"""
