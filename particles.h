#ifndef PARTICLES_H
#define PARTICLES_H

// Use SoA for better coalescing on GPU.
typedef struct {
  double *x;
  double *y;
  double *vx;
  double *vy;
} particle_grid_t;

#endif // PARTICLES_H