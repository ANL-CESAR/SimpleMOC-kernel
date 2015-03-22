#include "SimpleMOC-kernel_header.h"

int main( int argc, char * argv[] )
{
	int version = 2;

	srand(time(NULL));

	Input * I = set_default_input();
	read_CLI( argc, argv, I );

	logo(version);

	print_input_summary(I);

	// Build Source Data
  printf("Initializing source data...\n");
  float (* fine_flux_arr   )[I->fine_axial_intervals][I->egroups];
  float (* fine_source_arr )[I->fine_axial_intervals][I->egroups];
  float (* sigT_arr        )[I->egroups];
  initialize_sources(
      I->source_regions,
      I->fine_axial_intervals,
      I->egroups,
      &fine_flux_arr,
      &fine_source_arr,
      &sigT_arr
      );

  printf("Initializing state fluxes...\n");
  float ( * state_flux_arr )[I->egroups];
  initialize_state_flux( 
      I->n_state_fluxes, 
      I->egroups, 
      &state_flux_arr 
      );

  printf("Initializing random numbers...\n");
  unsigned int ( * randIdx )[3];
  initialize_randIdx( 
      I->segments, 
      &randIdx 
      );
	
	// Build Exponential Table
	//Table * table = buildExponentialTable( 0.01, 10.0 );

	center_print("SIMULATION", 79);
	border_print();
	printf("Attentuating fluxes across segments...\n");

	double start, stop;

	// Run Simulation Kernel Loop
	start = get_time();
  run_kernel(
      I->source_regions,
      I->fine_axial_intervals,
      I->segments,
      I->egroups,
      I->nthreads,
      I->n_state_fluxes,
      fine_flux_arr,
      fine_source_arr,
      sigT_arr,
      state_flux_arr,
      randIdx
      );

	stop = get_time();

	printf("Simulation Complete.\n");

	border_print();
	center_print("RESULTS SUMMARY", 79);
	border_print();

	double tpi = ((double) (stop - start) /
			(double)I->segments / (double) I->egroups) * 1.0e9;
	printf("%-25s%.3lf seconds\n", "Runtime:", stop-start);
	printf("%-25s%.3lf ns\n", "Time per Intersection:", tpi);
	border_print();

	return 0;
}
