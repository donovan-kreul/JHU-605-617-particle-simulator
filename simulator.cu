#include <cstddef>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#include "bitmap.hpp"

#define BLOCK_SIZE 512

// For 2D case (D = 2):
//   Particle matrix is 3 by 2*N
//   Simulator matrix is 3 by 3
#define D 2
#define H 3
#define W (D * N_PARTICLES)

// Simulation constants.
// TODO: make these into user-provided arguments
#define N_PARTICLES 10000
#define N_STEPS 500
#define PRINT_INTERVAL 100
#define TIME_STEP 0.001
#define G ((float)-9.8)

// Size of particle matrix.
#define MATRIX_SIZE (N_PARTICLES * H * D)

// Controls the (random) distribution of initial positions and velocities.
#define P_SCALE 0.8
#define V_SCALE 2.0

// Use SoA for better coalescing on GPU
typedef struct {
  double *x;
  double *y;
  double *vx;
  double *vy;
} particle_grid_t;

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

// Print a particle's index, position, and velocity.
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
void curand_init_kernel(unsigned int seed, curandState_t *states) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (thread_idx < N_PARTICLES) {
      curand_init(seed, thread_idx, 0, &states[thread_idx]);
  }
}

// Allocate space for particle grid on host.
particle_grid_t create_host_particle_grid(size_t num_particles) {
  // particle_grid_t p = (particle_grid_t *)malloc(sizeof(particle_grid_t));
  particle_grid_t p;
  size_t num_bytes = sizeof(double) * num_particles;
  p.x = (double *)malloc(num_bytes);
  p.y = (double *)malloc(num_bytes);
  p.vx = (double *)malloc(num_bytes);
  p.vy = (double *)malloc(num_bytes);
  return p;
}

// TODO: explain; give reference to https://forums.developer.nvidia.com/t/dynamic-array-inside-struct/10455 and google ai
// Allocate space for particle grid on device.
particle_grid_t create_device_particle_grid(size_t num_particles) {
  particle_grid_t p;
  // gpuErrChk(cudaMalloc(&p, sizeof(particle_grid_t)));
  size_t num_bytes = sizeof(double) * num_particles;
  double *x, *y, *vx, *vy;
  gpuErrChk(cudaMalloc(&x, num_bytes));
  gpuErrChk(cudaMalloc(&y, num_bytes));
  gpuErrChk(cudaMalloc(&vx, num_bytes));
  gpuErrChk(cudaMalloc(&vy, num_bytes));
  // particle_grid_t p_dummy;
  // p_dummy.x = x;
  // p_dummy.y = y;
  // p_dummy.vx = vx;
  // p_dummy.vy = vy;
  p.x = x;
  p.y = y;
  p.vx = vx;
  p.vy = vy;
  // gpuErrChk(cudaMemcpy(p, &p_dummy, sizeof(particle_grid_t), cudaMemcpyHostToDevice));
  return p;
}

// Free host particle grid.
void destroy_host_particle_grid(particle_grid_t p) {
  free(p.x);
  free(p.y);
  free(p.vx);
  free(p.vy);
  // free(p);
}

// TODO: add cuda error checking here?
// Free device particle grid.
void destroy_device_particle_grid(particle_grid_t p) {
  cudaFree(p.x);
  cudaFree(p.y);
  cudaFree(p.vx);
  cudaFree(p.vy);
  // cudaFree(p);
}

// Fill particle grid with random positions and velocities.
__global__ 
void initialize_device_particle_grid(curandState_t *states, particle_grid_t particles) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (thread_idx < N_PARTICLES) {
    double x, y, vx, vy;
    x = P_SCALE * curand_normal_double(&states[thread_idx]);
    y = P_SCALE * curand_normal_double(&states[thread_idx]);
    vx = V_SCALE * curand_normal_double(&states[thread_idx]);
    vy = V_SCALE * curand_normal_double(&states[thread_idx]);

    add_particle(particles, thread_idx, x, y, vx, vy);
  }
}

// TODO: may need to break this out into member arrays (x, y, etc.) for better performance
// Simulate dt seconds of time. Adjust vy according to gravity.
__device__
void update_particle(particle_grid_t p, size_t idx, double dt) {
  double x = p.x[idx];
  double y = p.y[idx];
  double vx = p.vx[idx];
  double vy = p.vy[idx];

  p.x[idx] = x + dt * vx;
  p.y[idx] = y + dt * vy;
  p.vy[idx] = vy + dt * G;
}

// TODO: may need to break this out into member arrays (x, y, etc.) for better performance
__global__
void update_device_particle_grid(particle_grid_t particles, size_t num_particles, double time_step) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < num_particles) {
    update_particle(particles, tid, time_step);
  }
}

// TODO: consider asynchronous copy here
// Copy device particle matrix to host.
void copy_particle_grid(particle_grid_t p_d, particle_grid_t p_h, size_t num_particles) {
  size_t num_bytes = sizeof(double) * num_particles;
  gpuErrChk(cudaMemcpy(p_h.x, p_d.x, num_bytes, cudaMemcpyDeviceToHost));
  gpuErrChk(cudaMemcpy(p_h.y, p_d.y, num_bytes, cudaMemcpyDeviceToHost));
  gpuErrChk(cudaMemcpy(p_h.vx, p_d.vx, num_bytes, cudaMemcpyDeviceToHost));
  gpuErrChk(cudaMemcpy(p_h.vy, p_d.vy, num_bytes, cudaMemcpyDeviceToHost));
}

// Copy device particle matrix to host, and print the first and last particles.
void copy_and_print(particle_grid_t p_d, particle_grid_t p_h, size_t num_particles, int step) {
  copy_particle_grid(p_d, p_h, num_particles);
	printf("\n=== Time step %d: ===\n", step);
	printParticle(p_h, 0);
	printParticle(p_h, num_particles - 1);
}

/* Simulate the motion of particles initialized with random values. */
int main(int argc, char** argv)
{	
  // Compute kernel arguments. [gridSize suggestion from LLM]
  // TODO: change to snake_case?
  int gridSize = (N_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int blockSize = BLOCK_SIZE;

	// [Taken from class example code] Set up cuRAND states.
	curandState_t* states;
	gpuErrChk(cudaMalloc((void **)&states, N_PARTICLES * sizeof(curandState_t)));
	curand_init_kernel<<<gridSize, blockSize>>>(time(0), states);
	gpuErrChk(cudaGetLastError());

	// Create particle grid on device, initialize with random values.
  particle_grid_t particles_d = create_device_particle_grid(N_PARTICLES);
	initialize_device_particle_grid<<<gridSize, blockSize>>>(states, particles_d);
	gpuErrChk(cudaGetLastError());

  // Copy device particle grid to host.
  particle_grid_t particles_h = create_host_particle_grid(N_PARTICLES);

  // uint8_t bg_color[3] = {0x00, 0x00, 0x00};
  // uint8_t pt_color[3] = {0xFF, 0xA5, 0x00};
  // size_t img_width = 256;
  // size_t img_height = 256;
  // cudaDeviceSynchronize();
  // generate_bitmap(img_width,img_height,particles_h,N_PARTICLES,bg_color,pt_color,"./img/image000.bmp");

	// Run simulation.
  for (int step = 0; step < N_STEPS; step++) {
    // Print status every PRINT_INTERVAL'th step.
    if (step % PRINT_INTERVAL == 0) {
      copy_and_print(particles_d, particles_h, N_PARTICLES, step);
      //     //TODO: make this asynchronous
      //     cudaDeviceSynchronize();
      //     char file_name[20];
      //     sprintf(file_name, "./img/image%03d.bmp", step / PRINT_INTERVAL);
      //     // generate_bitmap(img_width,img_height,particles_h,N_PARTICLES,bg_color,pt_color,file_name);
    }
    // Compute update to particle grid.
    update_device_particle_grid<<<gridSize, blockSize>>>(particles_d, N_PARTICLES, TIME_STEP);
  }

	// Final result printout.
	copy_and_print(particles_d, particles_h, N_PARTICLES, N_STEPS);
  // cudaDeviceSynchronize();
  
  // Generate final image.
  // generate_bitmap(img_width,img_height,particles_h,N_PARTICLES,bg_color,pt_color,"final.bmp");

	gpuErrChk(cudaFree(states));
  destroy_device_particle_grid(particles_d);
  destroy_host_particle_grid(particles_h);

	return EXIT_SUCCESS;
}