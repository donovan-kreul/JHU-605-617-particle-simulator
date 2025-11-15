# JHU-605-617-particle-simulator

Project setup and Makefile derived from https://github.com/TravisWThompson1/Makefile_Example_CUDA_CPP_To_Executable

// TODO: add /bin
// TODO: write up explanation of why we need a (non-pointer) struct to hold structure of arrays (attach reference(s))
// https://stackoverflow.com/questions/31598021/cuda-cudamemcpy-struct-of-arrays

ffmpeg taken from google AI with modifications:
`ffmpeg -f image2 -framerate 60 -i ./img/image%03d.bmp -c:v libx264 -pix_fmt yuv420p ./img/simulation.mp4`