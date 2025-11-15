#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include "args.h"

// Code adapted from getopt_long example
// https://www.gnu.org/software/libc/manual/html_node/Getopt-Long-Option-Example.html
args_t *get_arguments(int argc, char **argv) {

  // Set up arguments struct with defaults.
  args_t *args = (args_t *)malloc(sizeof(args_t));
  args->n_particles = NUM_PARTICLES_DEFAULT;
  args->block_size = BLOCK_SIZE_DEFAULT;
  args->n_steps = N_STEPS_DEFAULT;
  args->print_interval = PRINT_INTERVAL_DEFAULT;
  args->img_height = IMG_HEIGHT_DEFAULT;
  args->img_width = IMG_WIDTH_DEFAULT;
  args->boundary = BOUNDARY_DEFAULT;

  // Process arguments, checking for non-negative values as necessary.
  int c;
  int tmp;
  while (1) {
    static struct option long_options[] = {
      {"block-size",      required_argument, NULL, 'b'},
      {"boundary",        required_argument, NULL, 'c'},
      {"image-dim",       required_argument, NULL, 'd'},
      {"particles",       required_argument, NULL, 'n'},
      {"print-interval",  required_argument, NULL, 'p'},
      {"steps",           required_argument, NULL, 's'},
      {0, 0, 0, 0}
    };

    int option_index = 0;

    c = getopt_long(argc, argv, "b:c:d:n:p:s:", long_options, &option_index);

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
        if ((tmp = atoi(optarg)) > 0)
          args->block_size = tmp;
        break;

      case 'c':
        args->boundary = (double)atof(optarg);
        break;
      
      case 'd':
        if ((tmp = atoi(optarg)) > 0) {
          args->img_height = tmp;
          args->img_width = tmp;
        }
        break;

      case 'n':
        args->n_particles = (size_t)atol(optarg);
        break;

      case 'p':
        if ((tmp = atoi(optarg)) > 0)
          args->print_interval = tmp;
        break;

      case 's':
        if ((tmp = atoi(optarg)) > 0)
          args->n_steps = tmp;
        break;

      case '?':
        break;
      
      default:
        abort ();
    }
  }

  if (optind < argc) {
    printf("non-option ARGV-elements: ");
    while (optind < argc)
      printf("%s ", argv[optind++]);
    printf("\n");
  }

  return args;
}