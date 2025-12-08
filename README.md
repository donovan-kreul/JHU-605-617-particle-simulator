# JHU-605-617 Final Project: Particle Simulator

https://github.com/user-attachments/assets/6db1206b-564c-4fab-a7dd-222cd69b8748

## Overview

For my final project in this course, I implemented a basic particle simulator using CUDA. The simulation spawns particles in a two-dimensional environment. The particles' positions and velocities are initialized with random values using cuRAND. Currently, the particles are affected by gravity, and by collisons with the boundaries of the environment. The program generates a series of bitmap images which are then stitched together by ffmpeg. 

## Build and Run

The project relies on CUDA (including cuRAND) and ffmpeg. All testing has been done on Linux using an Nvidia Quadro P620. 

To build, run `make`. The executable will be `./simulator`, which can be run on its own or with additional arguments (see below). The output video will be written as `/img/simulation.mp4`. If the `--debug` flag is used, images will be written to `/img` in the bitmap (`.bmp`) format. To clear all intermediate files (and everything from `/img`), run `make clean`.

## Program Parameters

These are the available parameters, which can also be found with `./simulator --help`. 

```bash
  -h, --help
      Show this mesasge and exit.
  -g, --debug
      Print additional information and output bitmap image files. Useful for debugging.
  -b, --block-size
      Threads per block. Default: 512
  -c, --boundary
      Size of the 2D world environment. Default: 10.0
  -d, --image-dim
      Output image resolution (D by D). Default dimensions: 256 by 256
  -e, --elasticity
      The coefficient of restitution (COR); controls elasticity of collisons. Default: 0.70
  -n, --particles
      The number of particles in the simulation. Default: 1000
  -t, --duration
      The length of time to run the simulation, in seconds. Default: 5.0
```

## References

* https://en.wikipedia.org/wiki/BMP_file_format
* https://www.samaterials.com/content/coefficient-of-restitution.html
* https://stackoverflow.com/questions/31598021/cuda-cudamemcpy-struct-of-arrays
* https://forums.developer.nvidia.com/t/dynamic-array-inside-struct/10455
* https://www.gnu.org/software/libc/manual/html_node/Getopt-Long-Option-Example.html
* https://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/
* https://batchloaf.wordpress.com/2017/02/12/a-simple-way-to-read-and-write-audio-and-video-files-in-c-using-ffmpeg-part-2-video/
* https://www.gnu.org/software/libc/manual/html_node/Rounding-Functions.html
