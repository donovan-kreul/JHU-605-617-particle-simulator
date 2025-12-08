#ifndef BITMAP_H
#define BITMAP_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#include "particles.h"

#define pad_to_four(n) (((n + 3) / 4) * 4)

void generate_bitmap(size_t idx, unsigned int img_height, unsigned int img_width,
                      double scale, particle_grid_t p, size_t num_particles, 
                      FILE *pipeout, bool debug);

#endif // BITMAP_H