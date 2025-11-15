# JHU-605-617 Final Project: Particle Simulator

https://github.com/user-attachments/assets/6db1206b-564c-4fab-a7dd-222cd69b8748

## Overview

For my final project in this course, I am implementing a basic particle simulator using CUDA. The simulation spawns particles in a two-dimensional environment. The particles' positions and velocities are initialized with random values using cuRAND. Currently, the particles are affected by gravity, and by collisons with the boundaries of the environment. The program generates a series of bitmap images, which can then be stitched together into a video using ffmpeg.

## Build and Run

The project relies on CUDA and cuRAND. For final stitching of the video, the user can manually run ffmpeg. All testing has been done on Linux using an Nvidia Quadro P620. 

To build, run `make`. The executable will be `./simulator`, which can be run on its own or with additional arguments (see below). Images will be written to `/img` in the bitmap (`.bmp`) format. To clear all intermediate files (and everything from `/img`), run `make clean`.

To encode a `.mp4` video from the generated images, run the following command:
```bash
ffmpeg -f image2 -framerate 50 -i ./img/image%04d.bmp -c:v libx264 -pix_fmt yuv420p ./img/simulation.mp4
```
The video will be generated at `./img/simulation.mp4`. 

## Program Parameters

These are the available parameters, which can also be found with `./simulator --help`. 

```bash
  -h, --help
      Show this mesasge and exit.
  -g, --debug
      Print additional information. Useful for debugging.
  -b, --block-size
      Threads per block. Default: 512
  -c, --boundary
      Size of the 2D world environment. Default: 10.0
  -d, --image-dim
      Output image resolution (D by D). Default dimensions: 256 by 256
  -e, --elasticity
      The coefficient of restitution (COR); controls elasticity of collisons. Default: 0.50
  -n, --particles
      The number of particles in the simulation. Default: 1000
  -t, --duration
      The length of time to run the simulation, in seconds. Default: 5.0
```

## Future Work

There are several avenues I would like to explore for improvement and expansion of this project.

* Timing and performance compared to CPU-only implementation
  * Consider using GPU for additional work, such as creating the bitmap image buffers.
  * Make use of CUDA streams for better performance on the large memory transfers.
  * If possible: directly encode the video using the GPU.
* Simulate additional forces and/or types of particles.
  * Adding additional sources of gravity, modeling gaseous particles, simulating particle-particle interaction, or many other possibilities.

## References

* https://en.wikipedia.org/wiki/BMP_file_format
* https://www.samaterials.com/content/coefficient-of-restitution.html
* https://stackoverflow.com/questions/31598021/cuda-cudamemcpy-struct-of-arrays
* https://forums.developer.nvidia.com/t/dynamic-array-inside-struct/10455
* https://www.gnu.org/software/libc/manual/html_node/Getopt-Long-Option-Example.html
* https://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/
