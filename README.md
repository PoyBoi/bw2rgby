<h1 align="center">
Black and White Image/Video to Color
</h1>
<p align="center">
Denoises a Black and White image or live camera stream and then restores it to RGB using OpenCV and GAN's
</p>
<hr/>

### Description:

<hr/>

### Resources:

#### OpenCV Based:

- [OpenCV Implementation of Colorization](https://github.com/dhananjayan-r/Colorizer)
- [Image enhancement + restoration](https://www.tome01.com/nhance-restore-images-with-opencv-and-python#image-enhancement-techniques)
- [Working with inpainting in openCV](https://pyimagesearch.com/2020/05/18/image-inpainting-with-opencv-and-python/)
- [SO Issue on masking in openCV](https://stackoverflow.com/questions/10469235/opencv-apply-mask-to-a-color-image)

#### GAN Based:

- [Refusion model based restoration](https://github.com/Algolzw/image-restoration-sde)
- [GAN usage for colorization](https://github.com/emilwallner/Coloring-greyscale-images)
- [Using GAN's for inpainting](https://github.com/Nirvan101/Image-Restoration-deep-learning)

#### Open Source solutions:

- [For Image "restoration" (realESR-GAN)](https://github.com/xinntao/Real-ESRGAN)
- [For recontruction of faces(GFP-GAN)](https://github.com/TencentARC/GFPGAN) 
- [MS img restoration (VAE, mapping networks)](https://github.com/topics/old-photo-restoration)

#### Sci-Kit Image Based Masking:

- [Making masks using sci-kit image](https://campus.datacamp.com/courses/image-processing-in-python/image-restoration-noise-segmentation-and-contours?ex=1)

<hr/>

### To-do's

- Masking worked, need a way to get it to be right, edge detection is weak
  - need to find out where to place the edge detector
    - see if it is better w/ or w/o blur and/or processing
- NS is superior to telea based on blurriness

### References

- [Real-Time User-Guided Image Colorization with Learned Deep Priors](https://github.com/richzhang/colorization)
- [Aforementioned Research Paper](https://richzhang.github.io/colorization/)
