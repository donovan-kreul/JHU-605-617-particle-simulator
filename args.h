#ifndef ARGS_H
#define ARGS_H

#include <cstddef>

// Argument defaults.
#define BLOCK_SIZE_DEFAULT 512
#define NUM_PARTICLES_DEFAULT 1000
#define N_STEPS_DEFAULT 2500
#define IMG_HEIGHT_DEFAULT 256
#define IMG_WIDTH_DEFAULT 256
#define BOUNDARY_DEFAULT 10.0
#define ELASTICITY_DEFAULT 0.5
#define DURATION_DEFAULT 5.0

typedef struct {
  unsigned int block_size;
  unsigned int n_steps;
  unsigned int img_height;
  unsigned int img_width;
  double duration;
  double boundary;
  double elasticity;
  size_t n_particles;
  bool debug;
} args_t;

int get_arguments(int argc, char **argv, args_t *args);

#endif // ARGS_H