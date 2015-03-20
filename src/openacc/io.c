#include "SimpleMOC-kernel_header.h"

// Prints program logo
void logo(int version)
{
  border_print();
  printf(
      "   __           __        ___        __   __           ___  __        ___     \n"
      "  /__` |  |\\/| |__) |    |__   |\\/| /  \\ /  ` __ |__/ |__  |__) |\\ | |__  |   \n"
      "  .__/ |  |  | |    |___ |___  |  | \\__/ \\__,    |  \\ |___ |  \\ | \\| |___ |___\n" 
      );
  printf("\n");
  border_print();
  printf("\n");

  center_print("Developed at", 79);
  center_print("The Massachusetts Institute of Technology", 79);
  center_print("and", 79);
  center_print("Argonne National Laboratory", 79);
  printf("\n");
  char v[100];
  sprintf(v, "Version: %d", version);
  center_print(v, 79);
  printf("\n");
  border_print();
}

// Prints Section titles in center of 80 char terminal
void center_print(const char *s, int width)
{
  int length = strlen(s);
  int i;
  for (i=0; i<=(width-length)/2; i++) {
    fputs(" ", stdout);
  }
  fputs(s, stdout);
  fputs("\n", stdout);
}

// Prints a border
void border_print(void)
{
  printf(
      "==================================================================="
      "=============\n");
}

// Prints comma separated integers - for ease of reading
void fancy_int( int a )
{
  if( a < 1000 )
    printf("%d\n",a);

  else if( a >= 1000 && a < 1000000 )
    printf("%d,%03d\n", a / 1000, a % 1000);

  else if( a >= 1000000 && a < 1000000000 )
    printf("%d,%03d,%03d\n", a / 1000000, (a % 1000000) / 1000, a % 1000 );

  else if( a >= 1000000000 )
    printf("%d,%03d,%03d,%03d\n",
        a / 1000000000,
        (a % 1000000000) / 1000000,
        (a % 1000000) / 1000,
        a % 1000 );
  else
    printf("%d\n",a);
}

// Prints out the summary of User input
void print_input_summary(Input * I)
{
  center_print("INPUT SUMMARY", 79);
  border_print();
#ifdef OPENMP
  printf("%-25s%d\n", "Number of Threads:", I->nthreads);
#endif
  printf("%-25s%d\n", "Energy Groups:", I->egroups);
  printf("%-25s%d\n", "Source Regions:", I->source_regions);
  printf("%-25s%d\n", "Fine Axial Intervals:", I->fine_axial_intervals);
  printf("%-25s", "Segments:"); fancy_int(I->segments);
  printf("%-25s", "Flux states:"); fancy_int(I->n_state_fluxes);
#ifdef PAPI
  if( I->papi_event_set == -1)
    printf("%-25s%s\n", "PAPI event to count:", I->event_name);
#endif
  border_print();
}

// reads command line inputs and applies options
void read_CLI( int argc, char * argv[], Input * input )
{
  // defaults to max threads on the system	
#ifdef OPENMP
  input->nthreads = omp_get_num_procs();
#else
  input->nthreads = 1;
#endif

  // Collect Raw Input
  for( int i = 1; i < argc; i++ )
  {
    char * arg = argv[i];

    // nthreads (-t)
    if( strcmp(arg, "-t") == 0 )
    {
      if( ++i < argc )
        input->nthreads = atoi(argv[i]);
      else
        print_CLI_error();
    }

    // segments (-s)
    else if( strcmp(arg, "-s") == 0 )
    {
      if( ++i < argc )
        input->segments = atoi(argv[i]);
      else
        print_CLI_error();
    }

    // egroups (-e)
    else if( strcmp(arg, "-e") == 0 )
    {
      if( ++i < argc )
        input->egroups = atoi(argv[i]);
      else
        print_CLI_error();
    }

    // number of flux states (-f)
    else if( strcmp(arg, "-f") == 0 )
    {
      if( ++i < argc )
        input->n_state_fluxes = atoi(argv[i]);
      else
        print_CLI_error();
    }

#ifdef PAPI
    // Add single PAPI event
    else if( strcmp(arg, "-p") == 0 )
    {
      if( ++i < argc ){
        input->papi_event_set = -1;
        strcpy(input->event_name, argv[i]);
      }
      else
        print_CLI_error();
    }
#endif
    else
      print_CLI_error();
  }

  // Validate Input

  // Validate nthreads
  if( input->nthreads < 1 )
    print_CLI_error();

  assert( input->segments >= input->n_state_fluxes );
}

// print error to screen, inform program options
void print_CLI_error(void)
{
  printf("Usage: ./SimpleMOC <options>\n");
  printf("Options include:\n");
  printf("  -t <threads>        Number of OpenMP threads to run\n");
  printf("  -s <segments>       Number of segments to process\n");
  printf("  -e <energy groups>  Number of energy groups\n");
  printf("  -f <flux states>    Number of flux states\n");
  printf("  -p <PAPI event>     PAPI event name to count (1 only) \n");
  printf("See readme for full description of default run values\n");
  exit(1);
}

