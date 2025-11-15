#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include "args.h"

void print_help_message() {
  printf("=====================================================================================================\n");
  printf("  Particle Simulator\n");
  printf("=====================================================================================================\n");
  printf("\n");
  printf("  -h, --help\n");
  printf("      Show this mesasge and exit.\n");
  printf("  -g, --debug\n");
  printf("      Print additional information. Useful for debugging.\n");
  printf("  -b, --block-size\n");
  printf("      Threads per block. Default: %d\n", BLOCK_SIZE_DEFAULT);
  printf("  -c, --boundary\n");
  printf("      Size of the 2D world environment. Default: %.1f\n", BOUNDARY_DEFAULT);
  printf("  -d, --image-dim\n");
  printf("      Output image resolution (D by D). Default dimensions: %d by %d\n", IMG_WIDTH_DEFAULT, IMG_HEIGHT_DEFAULT);
  printf("  -e, --elasticity\n");
  printf("      The coefficient of restitution (COR); controls elasticity of collisons. Default: %.2f\n", ELASTICITY_DEFAULT);
  printf("  -n, --particles\n");
  printf("      The number of particles in the simulation. Default: %d\n", NUM_PARTICLES_DEFAULT);
  printf("  -t, --duration\n");
  printf("      The length of time to run the simulation, in seconds. Default: %.1f\n", DURATION_DEFAULT);
  printf("\n");
  printf("=====================================================================================================\n");
}

// Code adapted from getopt_long example
// https://www.gnu.org/software/libc/manual/html_node/Getopt-Long-Option-Example.html
// Returns 0 on success, 1 if -h was used, and -1 otherwise
int get_arguments(int argc, char **argv, args_t *args) {

  // Set up arguments struct with defaults.
  args->n_particles = NUM_PARTICLES_DEFAULT;
  args->block_size = BLOCK_SIZE_DEFAULT;
  args->n_steps = N_STEPS_DEFAULT;
  args->img_height = IMG_HEIGHT_DEFAULT;
  args->img_width = IMG_WIDTH_DEFAULT;
  args->duration = DURATION_DEFAULT;
  args->boundary = BOUNDARY_DEFAULT;
  args->elasticity = ELASTICITY_DEFAULT;
  args->debug = false;

  // Process arguments, checking for non-negative values as necessary.
  int c;
  int tmp_int;
  float tmp_double;
  while (1) {
    static struct option long_options[] = {
      {"help",            no_argument,       NULL, 'h'},
      {"debug",           no_argument,       NULL, 'g'},
      {"block-size",      required_argument, NULL, 'b'},
      {"boundary",        required_argument, NULL, 'c'},
      {"image-dim",       required_argument, NULL, 'd'},
      {"elasticity",      required_argument, NULL, 'e'},
      {"particles",       required_argument, NULL, 'n'},
      {"steps",           required_argument, NULL, 's'},
      {"duration",        required_argument, NULL, 't'},
      {0, 0, 0, 0}
    };

    int option_index = 0;

    c = getopt_long(argc, argv, "hgb:c:d:e:n:s:t:", long_options, &option_index);

    if (c == -1)
      break;

    switch (c) {
      case 0:
        if (long_options[option_index].flag != 0)
          break;
        printf("option %s", long_options[option_index].name);
        if (optarg)
          printf(" with arg %s", optarg);
        printf("\n");
        break;

      case 'b':
        if ((tmp_int = atoi(optarg)) > 0)
          args->block_size = tmp_int;
        break;

      case 'c':
        if ((tmp_double = atof(optarg)) > 0)
          args->boundary = tmp_double;
        break;
      
      case 'd':
        if ((tmp_int = atoi(optarg)) > 0) {
          args->img_height = tmp_int;
          args->img_width = tmp_int;
        }
        break;

      case 'e':
        if ((tmp_double = atof(optarg)) > 0)
          args->elasticity = tmp_double;
        break;

      case 'h':
        print_help_message();
        return 1;
        break;

      case 'g':
        args->debug = true;
        break;

      case 'n':
        args->n_particles = (size_t)atol(optarg);
        break;

      case 's':
        if ((tmp_int = atoi(optarg)) > 0)
          args->n_steps = tmp_int;
        break;

      case 't':
        if ((tmp_double = atof(optarg)) > 0)
          args->duration = tmp_double;
        break;

      case '?':
        return -1;
        break;
      
      default:
        return -1;
    }
  }

  if (optind < argc) {
    printf("non-option ARGV-elements: ");
    while (optind < argc)
      printf("%s ", argv[optind++]);
    printf("\n");
  }

  return 0;
}