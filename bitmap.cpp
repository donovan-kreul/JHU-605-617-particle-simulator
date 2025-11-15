#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>

#include "bitmap.h"
#include "particles.h"

// Bitmap header size. 
#define BMP_HDR_SIZE 54

// HELPER FUNCTIONS
size_t min(size_t a, size_t b) {
  return ((a < b) ? a : b);
}

size_t max(size_t a, size_t b) {
  return ((a > b) ? a : b);
}

size_t clamp(size_t val, size_t low, size_t high) {
  return max(low, min(val, high));
}

int min(int a, int b) {
  return ((a < b) ? a : b);
}

int max(int a, int b) {
  return ((a > b) ? a : b);
}

int clamp(int val, int low, int high) {
  return max(low, min(val, high));
}

// Write the given buffer to a file. 
// NOTE: Doesn't validate that buffer is in proper bitmap format.
void write_bitmap_to_file(uint8_t *buffer, size_t num_bytes, char *file_name) {
  FILE *write_ptr;
  write_ptr = fopen(file_name,"wb");
  fwrite(buffer, sizeof(uint8_t), num_bytes, write_ptr);
  fclose(write_ptr);
}

// Set up an image buffer of the given size.
uint8_t *allocate_image_buffer(size_t header_size, size_t grid_size) {
  uint8_t *buffer = (uint8_t *)calloc(header_size + grid_size, sizeof(uint8_t));
  return buffer;
}

// Writes the low four bytes of val as little-endian in buffer.
void set_four_bytes_LE(uint8_t *buffer, size_t idx, size_t val) {
  buffer[idx]   =  val        & 0xFF;
  buffer[idx+1] = (val >>  8) & 0xFF;
  buffer[idx+2] = (val >> 16) & 0xFF;
  buffer[idx+3] = (val >> 24) & 0xFF;
}

// Writes the (mostly-boilerplate) BMP header to the buffer.
void write_header_to_buffer(uint8_t *buffer, unsigned int img_height, 
                            unsigned int img_width, size_t num_bytes) {
  uint8_t bmp_header[BMP_HDR_SIZE] = {	
    // BMP header
    0x42, 0x4D,
    0x46, 0x00, 0x00, 0x00, // file size (54 bytes of header)
    0x00, 0x00,
    0x00, 0x00,
    0x36, 0x00, 0x00, 0x00,
    // DIB header
    0x28, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00, // image length
    0x02, 0x00, 0x00, 0x00, // image width
    0x01, 0x00,
    0x18, 0x00,
    0x00, 0x00, 0x00, 0x00,
    0x10, 0x00, 0x00, 0x00,
    0x13, 0x0B, 0x00, 0x00,
    0x13, 0x0B, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00
  };

  set_four_bytes_LE(bmp_header, 2, num_bytes);
  set_four_bytes_LE(bmp_header, 18, img_height);
  set_four_bytes_LE(bmp_header, 22, img_width);
  
  for (int i = 0; i < BMP_HDR_SIZE; i++) {
    buffer[i] = bmp_header[i];
  }
}

// Returns an index into the grid based on particle's position.
size_t get_grid_index(double scale, unsigned int img_height, unsigned int img_width, 
                      double pt_x, double pt_y) {
  int out_x = (pt_x * (0.5 * img_width / scale) + (0.5 * img_width));
  int out_y = (pt_y * (0.5 * img_height / scale) + (0.5 * img_height));
  out_x = clamp(out_x, 0, (int)img_width - 1);
  out_y = clamp(out_y, 0, (int)img_height - 1);
  size_t idx = BMP_HDR_SIZE + 3 * (img_width * out_y + out_x);
  // printf("%f %f -> %d %d %lu\n", pt_x, pt_y, out_x, out_y, idx); fflush(stdout);
  return idx;
}

// Color a single pixel in the buffer, representing a particle's position.
void write_particle_to_buffer(uint8_t *buffer, double scale, size_t img_height, 
                              size_t img_width, double pt_x, double pt_y, 
                              const uint8_t *pt_color) {
  size_t idx = get_grid_index(scale, img_height, img_width, pt_x, pt_y);
  buffer[idx]   = pt_color[2];
  buffer[idx+1] = pt_color[1];
  buffer[idx+2] = pt_color[0];
}

// Generate a bitmap image file based on the given particle grid. 
// NOTE: Currently, bg_color is ignored.
void generate_bitmap(unsigned int img_height, unsigned int img_width, double scale,
                      particle_grid_t p, size_t num_particles, const uint8_t *bg_color, 
                      const uint8_t *pt_color, char *file_name) {
  size_t img_buffer_width = pad_to_four(3 * img_width);
  size_t img_grid_size = img_height * img_buffer_width;
  size_t num_bytes = BMP_HDR_SIZE + img_grid_size;
  uint8_t *img_buffer = allocate_image_buffer(BMP_HDR_SIZE, img_grid_size);

  write_header_to_buffer(img_buffer, img_height, img_width, num_bytes);

  for (int i = 0; i < num_particles; i++) {
    double pt_x = p.x[i];
    double pt_y = p.y[i];
    write_particle_to_buffer(img_buffer, scale, img_height, img_width, pt_x, pt_y, pt_color);
  }

  write_bitmap_to_file(img_buffer, num_bytes, file_name);

  free(img_buffer);
}