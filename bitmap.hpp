#ifndef BITMAP_H
#define BITMAP_H

#include <stdint.h>
#include <stddef.h>

#define bmp_length 2
#define bmp_width 2
#define bmp_size (bmp_length * bmp_width)
#define pad_to_four(n) (((n + 3) / 4) * 4)

#define bmp_pixel_array_size (size_t)(pad_to_four(3 * bmp_width) * bmp_length)

// void generate_bitmap();
void generate_bitmap(size_t img_length, size_t img_width, float *particles, size_t num_particles, const uint8_t *bg_color, const uint8_t *pt_color, char *file_name);

#endif // BITMAP_H