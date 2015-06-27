#include "SimpleMOC-kernel_header.h"

int main( int argc, char * argv[] )
{
	int version = 3;
	unsigned long long vhash;

	#ifdef PAPI
	papi_serial_init();
	#endif

	#ifdef VERIFY
	srand(1);
	#else
	srand(time(NULL));
	#endif

	// Get Inputs
	Input * I = set_default_input();
	read_CLI( argc, argv, I );
	
	// Calculate Number of 3D Source Regions
	I->source_3D_regions = (int) ceil((double)I->source_2D_regions *
		I->coarse_axial_intervals / I->decomp_assemblies_ax);

	logo(version);

	#ifdef OPENMP
	omp_set_num_threads(I->nthreads); 
	#endif
	
	// Build Source Data
	Source * S = initialize_sources(I); 
	
	// Build Exponential Table
	Table * table;
	#ifdef TABLE
	table = buildExponentialTable( 0.01, 10.0, I );
	#endif
	
	print_input_summary(I);

	center_print("SIMULATION", 79);
	border_print();
	printf("Attentuating fluxes across segments...\n");

	double start, stop;

	// Run Simulation Kernel Loop
	start = get_time();
	vhash = run_kernel(I, S, table);
	stop = get_time();

	printf("Simulation Complete.\n");

	border_print();
	center_print("RESULTS SUMMARY", 79);
	border_print();

	double tpi = ((double) (stop - start) /
			(double)I->segments / (double) I->egroups) * 1.0e9;
	printf("%-25s%.3lf seconds\n", "Runtime:", stop-start);
	printf("%-25s%.3lf ns\n", "Time per Intersection:", tpi);
	#ifdef VERIFY
	printf("%-25s%llu\n", "Verification Hash:", vhash);
	#endif
	border_print();

	return 0;
}
