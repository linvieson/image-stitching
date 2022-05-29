# Image Stitching
### Linear Algebra Coursework Project

### Aim of the project
The aim is to implement an algorithm that composites multiple overlapping images captured from different viewing positions into one natural-looking panorama, using linear algebra principles and algorithms. It will help obtain a full-range view in a single photo, excluding the necessity of manual stitching in photo editors. The final goal is to create a program that receives some images of an object/surface/scene as an input and outputs a single picture of that

### Implementation

There are three main steps in implementing the image stitching algorithm:

- Keypoint detection and local invariant descriptors extraction
- Feature matching
- Homography estimation with RANSAC algorithm

Each step is described more meticulously in the [final report](). It contains description of Linear Algebra methods and algorithms' pseudocodes.

### Data and implementatino

Several sources were used to execute the algorithm, including own photo materials and open-source images on the Internet. The image stitching algorithm gets a set of two or more images in _jpg_ or _png_ format (specified by the user when entering a path) and outputs one single wide-view panorama stitched from those pictures. Examples of data input and output are presented in the __Results__ section.

Images can be stitched either horizontally or vertically. In the latter case, each image is rotated by 90 degrees clockwise, stitched, and rotated back. Therefore, it is essential for the user to specify the type of stitching they want to perform. It can be done by typing **H** for horizontal stitching and **V** for vertical stitching, respectively, in the particular input field.

### Usage
Firstly, `git clone https://github.com/linvieson/image-stitching` to your computer. Open the module `stitching.py` and run it. The following will appeat in the terminal:

```
Input H if you want to stitch images horizintally, V if vertically:
```

Input **H** if you want to stitch the images horizontally, or **V** if you want them to be stitched vertically. Press enter. The following text will appear:

```
Path to image:
```

Enter the path to the image, press enter, input the next path, and so on. When you finish inputting the images, press enter one more time. The program will start to work. In some time, the final stitched picture will be displayed on the screen, and saved on your computer by path `picture{i}.jpg`, with i being the largest index.


### Results

The result of the project is an image stitching algorithm. By inputting the paths to the images the user wants to stitch, they can get one full image obtained from cropped ones. More specific results of different input options are displayed in the [final report]().


### Contributors

- [Alina Voronina](https://github.com/linvieson)
- [Viktoriia Maksymiuk](https://github.com/Vihtoriaaa)
- [Yulia Maksymiuk](https://github.com/juliaaz)


### Lisence

[MIT](https://github.com/linvieson/image-stitching/blob/main/LICENSE)
