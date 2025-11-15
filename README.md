# JHU-605-617 Final Project: Particle Simulator

## Summary

For my final project in this course, I am implementing a basic particle simulator using CUDA. The simulation spawns particles in a two-dimensional environment. The particles' positions and velocities are initialized with random values using cuRAND. Currently, the particles are affected by gravity, and by collisons with the boundaries of the environment. The program generates a series of bitmap images, which can then be stitched together into a video using ffmpeg.

## Build and Run

The project relies on CUDA and cuRAND. For final stitching of the video, the user can manually run ffmpeg. All testing has been done on Linux using an Nvidia Quadro P620. 

To build, run `make`. The executable will be `./simulator`, which can be run without any additional arguments. See below for optional arguments. Images will be sent to `/img` in the bitmap (`.bmp`) format. To clear all intermediate files (and everything from `/img`), run `make clean`.

To encode a `.mp4` video from the generated images, run the following command:
```bash
ffmpeg -f image2 -framerate 50 -i ./img/image%04d.bmp -c:v libx264 -pix_fmt yuv420p ./img/simulation.mp4
```
The video will be generated at `./img/simulation.mp4`. 

## Program Parameters

These are the available parameters, which can also be found with `./simulator --help`. 

## References

// TODO: write up explanation of why we need a (non-pointer) struct to hold structure of arrays (attach reference(s))
// https://stackoverflow.com/questions/31598021/cuda-cudamemcpy-struct-of-arrays

https://forums.developer.nvidia.com/t/dynamic-array-inside-struct/10455

ffmpeg taken from google AI with modifications:
