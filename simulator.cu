#include <cstddef>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#include "args.h"
#include "particles.h"
#include "bitmap.h"

// Simulation renders one image per TIME_STEP * PRINT_INTERVAL seconds,
// which in this case is 50 frames per second.
#define PRINT_INTERVAL 20
#define TIME_STEP 0.001

// Gravity.
#define G ((float)-9.8)

// Controls the (random) distribution of initial positions and velocities.
#define P_SCALE 1.0
#define V_SCALE 5.0

// Standard CUDA error-check macro.
// Taken from Robert Crovella on stackexchange.
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess) 
  {
    fprintf(stderr,"CUDA ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

// Print a particle's index, position, and velocity. Useful for debugging.
void printParticle(particle_grid_t p, int idx) {
  double x = p.x[idx];
  double y = p.y[idx];
  double vx = p.vx[idx];
  double vy = p.vy[idx];
	printf("Particle %d:\n", idx);
	printf("  pos: (%.3lf, %.3lf)\n", x, y);
	printf("  vel: (%.3lf, %.3lf)\n", vx, vy);
}

// Add a particle with the given position and location to a particle array.
__device__ __host__ 
void add_particle(particle_grid_t p, int idx, double x, double y, double vx, double vy) {
  p.x[idx] = x;
  p.y[idx] = y;
  p.vx[idx] = vx;
  p.vy[idx] = vy;
}

// [Taken from class example code]
// Initialize random states, one for each particle.
__global__ 
void curand_init_kernel(unsigned int seed, curandState_t *states, size_t n_particles) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (thread_idx < n_particles) {
      curand_init(seed, thread_idx, 0, &states[thread_idx]);
  }
}

// Allocate space for particle grid on host.
particle_grid_t create_host_particle_grid(size_t num_particles) {
  particle_grid_t p;
  size_t num_bytes = sizeof(double) * num_particles;
  p.x = (double *)malloc(num_bytes);
  p.y = (double *)malloc(num_bytes);
  p.vx = (double *)malloc(num_bytes);
  p.vy = (double *)malloc(num_bytes);
  return p;
}

// Allocate space for particle grid on device.
particle_grid_t create_device_particle_grid(size_t num_particles) {
  particle_grid_t p;  size_t num_bytes = sizeof(double) * num_particles;
  double *x, *y, *vx, *vy;
  gpuErrChk(cudaMalloc(&x, num_bytes));
  gpuErrChk(cudaMalloc(&y, num_bytes));
  gpuErrChk(cudaMalloc(&vx, num_bytes));
  gpuErrChk(cudaMalloc(&vy, num_bytes));
  p.x = x;
  p.y = y;
  p.vx = vx;
  p.vy = vy;
  return p;
}

// Free host particle grid.
void destroy_host_particle_grid(particle_grid_t p) {
  free(p.x);
  free(p.y);
  free(p.vx);
  free(p.vy);
}

// Free device particle grid.
void destroy_device_particle_grid(particle_grid_t p) {
  cudaFree(p.x);
  cudaFree(p.y);
  cudaFree(p.vx);
  cudaFree(p.vy);
}

// Fill particle grid with random positions and velocities.
__global__ 
void initialize_device_particle_grid(curandState_t *states, particle_grid_t particles, 
                                      size_t n_particles) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (thread_idx < n_particles) {
    double x, y, vx, vy;
    x = P_SCALE * curand_normal_double(&states[thread_idx]);
    y = P_SCALE * curand_normal_double(&states[thread_idx]);
    vx = V_SCALE * curand_normal_double(&states[thread_idx]);
    vy = V_SCALE * curand_normal_double(&states[thread_idx]);

    add_particle(particles, thread_idx, x, y, vx, vy);
  }
}

// Simulate dt seconds of time for a single particle. 
__device__
void update_particle(particle_grid_t p, size_t idx, double dt, double bdry, double e) {
  double x = p.x[idx];
  double y = p.y[idx];
  double vx = p.vx[idx];
  double vy = p.vy[idx];

  // Compute change in position and velocity.
  x = x + dt * vx;
  y = y + dt * vy;
  vy = vy + dt * G;

  // Check for collisions with boundary box of [-bdry, bdry] x [bdry, bdry].
  if (x < -bdry) {
    x = -bdry;
    vx *= -1.0 * e;
    vy *= e;
  }
  else if (x > bdry) {
    x = bdry;
    vx *= -1.0 * e;
    vy *= e;
  }
  if (y < -bdry) {
    y = -bdry;
    vy *= -1.0 * e;
    vx *= e;
  }
  else if (y > bdry) {
    y = bdry;
    vy *= -1.0 * e;
    vx *= e;
  }

  // Update particle with new values.
  p.x[idx] = x;
  p.y[idx] = y;
  p.vx[idx] = vx;
  p.vy[idx] = vy;
}

// Simulate time_step seconds of time for all particles in the grid.
__global__
void update_device_particle_grid(particle_grid_t particles, size_t num_particles, 
                                  double time_step, double boundary, double elasticity) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < num_particles) {
    update_particle(particles, tid, time_step, boundary, elasticity);
  }
}

// Copy device particle matrix to host.
void copy_particle_grid(particle_grid_t p_d, particle_grid_t p_h, size_t num_particles) {
  size_t num_bytes = sizeof(double) * num_particles;
  gpuErrChk(cudaMemcpy(p_h.x, p_d.x, num_bytes, cudaMemcpyDeviceToHost));
  gpuErrChk(cudaMemcpy(p_h.y, p_d.y, num_bytes, cudaMemcpyDeviceToHost));
  gpuErrChk(cudaMemcpy(p_h.vx, p_d.vx, num_bytes, cudaMemcpyDeviceToHost));
  gpuErrChk(cudaMemcpy(p_h.vy, p_d.vy, num_bytes, cudaMemcpyDeviceToHost));
}

// Print the first and last particles at a given step.
void debug_print(bool debug, particle_grid_t p_h, size_t num_particles, int step) {
  if (debug) {
    printf("\n=== Time step %d: ===\n", step);
    printParticle(p_h, 0);
    printParticle(p_h, num_particles - 1);
  }
}

// Simulate the motion of particles initialized with random values.
int main(int argc, char** argv)
{	
  // Get command-line arguments.
  args_t *args = (args_t *)malloc(sizeof(args_t));
  int result;
  if ((result = get_arguments(argc, argv, args)) != 0) {
    free(args);
    exit(result);
  }
  
  // Set up particle and background colors for generated bitmap images.
  char file_name[64];
  uint8_t bg_color[3] = {0x00, 0x00, 0x00};
  uint8_t pt_color[3] = {0xFF, 0xA5, 0x00};
  
  // Assign values to local variables for better readability.
  size_t n_particles = args->n_particles;
  unsigned int block_size = args->block_size;
  unsigned int grid_size = (n_particles + block_size - 1) / block_size;
  unsigned int img_width = args->img_width;
  unsigned int img_height = args->img_height;
  double duration = args->duration;
  double boundary = args->boundary;
  double elasticity = args->elasticity;
  bool debug = args->debug;
  
  // Compute number of steps needed for desired duration.
  unsigned int print_interval = PRINT_INTERVAL;
  unsigned int n_steps = (unsigned int)(duration / TIME_STEP);
  
	// [Taken from class example code] Set up cuRAND states.
	curandState_t* states;
	gpuErrChk(cudaMalloc((void **)&states, n_particles * sizeof(curandState_t)));
	curand_init_kernel<<<grid_size, block_size>>>
    (time(0), states, n_particles);
	gpuErrChk(cudaGetLastError());
  
	// Create particle grid on device, and initialize with random values.
  printf("Creating particle grids...\n");
  particle_grid_t particles_d = create_device_particle_grid(n_particles);
	initialize_device_particle_grid<<<grid_size, block_size>>>
    (states, particles_d, n_particles);
	gpuErrChk(cudaGetLastError());
  
  // Allocate space for host particle grid.
  particle_grid_t particles_h = create_host_particle_grid(n_particles);
  
	// Run simulation.
  printf("Running %u steps of simulation...\n", n_steps);
  for (int step = 0; step < n_steps; step++) {
    // Generate bitmap image on every print_interval'th step.
    if (step % print_interval == 0) {
      copy_particle_grid(particles_d, particles_h, n_particles);
      debug_print(debug, particles_h, n_particles, step);
      sprintf(file_name, "./img/image%04d.bmp", step / print_interval);
      generate_bitmap(img_width, img_height, boundary, particles_h,
                      n_particles, bg_color, pt_color, file_name);
    }
    // Compute update to particle grid.
    update_device_particle_grid<<<grid_size, block_size>>>
      (particles_d, n_particles, TIME_STEP, boundary, elasticity);
  }

	// Generate image of final result.
  copy_particle_grid(particles_d, particles_h, n_particles);
	debug_print(debug, particles_h, n_particles, n_steps);
  sprintf(file_name, "./img/image%04u.bmp", n_steps / print_interval);
  generate_bitmap(img_width, img_height, boundary, particles_h,
                  n_particles, bg_color, pt_color, file_name);

  // Clean up memory allocations.
	gpuErrChk(cudaFree(states));
  destroy_device_particle_grid(particles_d);
  destroy_host_particle_grid(particles_h);
  free(args);

  printf("Simulation complete! See /img for output.\n");
	return EXIT_SUCCESS;
}