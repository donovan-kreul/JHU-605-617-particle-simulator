#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <cublas.h>
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
#define N_PARTICLES 100
#define N_STEPS 2500
#define PRINT_INTERVAL 10
#define TIME_STEP 0.001
#define G ((float)-9.8)

// Size of particle matrix.
#define MATRIX_SIZE (N_PARTICLES * H * D)

// Controls the (random) distribution of initial positions and velocities.
#define P_SCALE 0.8
#define V_SCALE 2.0

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

// Similar error-check macro for cuBLAS.
#define cbErrChk(ans) { cublasGpuAssert((ans), __FILE__, __LINE__); }
inline void cublasGpuAssert(cublasStatus_t status, const char *file, int line)
{
   if (status != CUBLAS_STATUS_SUCCESS) 
   {
      fprintf(stderr,"cuBLAS ERROR: %s %s %d\n", cublasGetStatusString(status), file, line);
      exit(status);
   }
}

// [Taken from class example code]
// Print a matrix P from memory.
#define index(i,j,ld) (((j)*(ld))+(i))
void printMat(float*P, int uWP, int uHP){
	int i,j;
	for (i = 0; i < uHP; i++){
		printf("\n");
		for(j = 0; j < uWP; j++)
			printf("%f ",P[index(i,j,uHP)]);
	}
}

// Print a particle's index, position, and velocity.
void printParticle(float *particles, int index) {
	int i = H * D * index;
	float x =  particles[i];
	float vx = particles[i + 1];
	float y =  particles[i + 3];
	float vy = particles[i + 4];
	printf("Particle %d:\n", index);
	printf("  pos: (%.3f, %.3f)\n", x, y);
	printf("  vel: (%.3f, %.3f)\n", vx, vy);
}

// Add a particle with the given position and location to a particle array.
__device__ __host__ void add_particle(float *particle_array, 
								      int index, 
                                      float x, 
                                      float y, 
                                      float vx, 
                                      float vy) 
{
	int i = H * D * index;
	particle_array[i]     = x;
	particle_array[i + 1] = vx;
	particle_array[i + 2] = 0;
	particle_array[i + 3] = y;
	particle_array[i + 4] = vy;
	particle_array[i + 5] = 1;
}

// [Taken from class example code]
// Initialize random states, one for each particle.
__global__ void curand_init_kernel(unsigned int seed, curandState_t *states) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread_idx < N_PARTICLES) {
        curand_init(seed, thread_idx, 0, &states[thread_idx]);
    }
}

// Fill particle array with random positions and velocities.
__global__ void initialize_particle_grid(curandState_t *states, float *particles) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread_idx < N_PARTICLES) {
        float x, y, vx, vy;
        x = P_SCALE * curand_normal(&states[thread_idx]);
        y = P_SCALE * curand_normal(&states[thread_idx]);
        vx = V_SCALE * curand_normal(&states[thread_idx]);
        vy = V_SCALE * curand_normal(&states[thread_idx]);
    
        add_particle(particles, thread_idx, x, y, vx, vy);
    }
}

// Copy device particle matrix to host, and print the first and last particles.
void copy_and_print(float *particles_d, float *particles_h, int step) {
	cbErrChk(cublasGetMatrix(H, W, sizeof(float), particles_d, H, particles_h, H));
	printf("\n=== Time step %d: ===\n", step);
	printParticle(particles_h, 0);
	printParticle(particles_h, N_PARTICLES - 1);
}


/* Simulate the motion of particles initialized with random values. */
int main(int argc, char** argv)
{	
  // Compute kernel arguments. [gridSize suggestion from LLM]
  int gridSize = (N_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int blockSize = BLOCK_SIZE;

	// [Taken from class example code] Set up cuBLAS and cuRAND states.
	cublasInit();
	curandState_t* states;
	gpuErrChk(cudaMalloc((void **)&states, N_PARTICLES * sizeof(curandState_t)));
	curand_init_kernel<<<gridSize, blockSize>>>(time(0), states);
	gpuErrChk(cudaGetLastError());

	// Initialize host matrices
	float *particles_h = (float *)malloc(sizeof(float) * MATRIX_SIZE);
	// Linear function matrix used to compute the simulation .
	float simulator_h[H*H] = {1.0, 0.0, 0.0, 
                              TIME_STEP, 1.0, 0.0, 
                              0.0, TIME_STEP*G, 1.0};

	// Initialize device matrices
	float *particles_d, *buffer_d, *simulator_d;
	cbErrChk(cublasAlloc(MATRIX_SIZE, sizeof(float), (void **)&particles_d));
	cbErrChk(cublasAlloc(MATRIX_SIZE, sizeof(float), (void**)&buffer_d));
	cbErrChk(cublasAlloc(H*H, sizeof(float), (void **)&simulator_d));

	// Get initial (randomized) particle positions and velocities.
	initialize_particle_grid<<<gridSize, blockSize>>>(states, particles_d);
	gpuErrChk(cudaGetLastError());

	// Copy host simulator matrix to device.
	cbErrChk(cublasSetMatrix(H, H, sizeof(float), simulator_h, H, simulator_d, H));

  // Generate initial image.
  uint8_t bg_color[3] = {0x00, 0x00, 0x00};
  uint8_t pt_color[3] = {0xFF, 0xA5, 0x00};
  size_t img_width = 256;
  size_t img_height = 256;
  copy_and_print(particles_d, particles_h, 0);
  cudaDeviceSynchronize();
  generate_bitmap(img_width,img_height,particles_h,N_PARTICLES,bg_color,pt_color,"./img/image000.bmp");

	// Run simulation.
	for (int step = 0; step < N_STEPS; step++) {
		// Print status every PRINT_INTERVAL'th step.
		if (step > 0 && (step % PRINT_INTERVAL == 0)) {
			copy_and_print(particles_d, particles_h, step);
      //TODO: make this asynchronous
      cudaDeviceSynchronize();
      char file_name[20];
      sprintf(file_name, "./img/image%03d.bmp", step / PRINT_INTERVAL);
      generate_bitmap(img_width,img_height,particles_h,N_PARTICLES,bg_color,pt_color,file_name);
		}
		// Compute B = S * P; use pointer swap to make this P = S * P.
		cublasSgemm('n', 'n', H, W, H, 1, simulator_d, H, particles_d, H, 0, buffer_d, H);
		cbErrChk(cublasGetError());
		float *tmp = particles_d; particles_d = buffer_d; buffer_d = tmp;
	}

	// Final result printout.
	copy_and_print(particles_d, particles_h, N_STEPS);
  cudaDeviceSynchronize();
  
  // Generate final image.
  generate_bitmap(img_width,img_height,particles_h,N_PARTICLES,bg_color,pt_color,"final.bmp");

	// Free memory and shut down cuBLAS.
	cbErrChk(cublasFree(simulator_d));
	cbErrChk(cublasFree(buffer_d));
	cbErrChk(cublasFree(particles_d));
	gpuErrChk(cudaFree(states));
	cbErrChk(cublasShutdown());
	free(particles_h);

	return EXIT_SUCCESS;
}