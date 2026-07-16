"""Vectorized classical image-filter building blocks used by both the CLI
(imgfilter.py) and the GUI (filters_gui.py) in this project.

Submodules:
    io           -- load_image / save_image, with real error handling.
    pointops     -- brightness/contrast, Gaussian noise, aspect-preserving
                    resize (already vectorized in the original app).
    convolution  -- 2D convolution via cv2.filter2D + the 5 kernel presets
                    from the original combobox.
    kuwahara     -- vectorized Kuwahara edge-preserving filter.
    frequency    -- 2D FFT filtering (low/high/band-pass, notch) + magnitude
                    spectrum visualization.
    edges        -- Canny (with an optional stage-by-stage breakdown),
                    Laplacian/LoG, Sobel gradient magnitude.
    morphology   -- erosion/dilation/opening/closing/tophat/blackhat.
"""
