#ifndef BITMAP_H
#define BITMAP_H

#include <stdint.h>
#include <stddef.h>

#include "particles.h"

#define pad_to_four(n) (((n + 3) / 4) * 4)

void generate_bitmap(unsigned int img_length, unsigned int img_width, particle_grid_t p, size_t num_particles, const uint8_t *bg_color, const uint8_t *pt_color, char *file_name);

#endif // BITMAP_H