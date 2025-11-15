#ifndef ARGS_H
#define ARGS_H

#include <cstddef>

// Argument defaults.
#define BLOCK_SIZE_DEFAULT 512
#define NUM_PARTICLES_DEFAULT 1000
#define N_STEPS_DEFAULT 500
#define PRINT_INTERVAL_DEFAULT 100
#define IMG_HEIGHT_DEFAULT 256
#define IMG_WIDTH_DEFAULT 256

typedef struct {
  unsigned int block_size;
  unsigned int n_steps;
  unsigned int print_interval;
  unsigned int img_height;
  unsigned int img_width;
  size_t n_particles;
} args_t;

args_t *get_arguments(int argc, char **argv);

#endif // ARGS_H