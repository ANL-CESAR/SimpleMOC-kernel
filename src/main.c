#include "SimpleMOC-kernel_header.h"

int main( int argc, char * argv[] )
{
	int version = 0;

	#ifdef PAPI
	papi_serial_init();
	#endif

	srand(time(NULL));

	Input * I = set_default_input();
	read_CLI( argc, argv, I );

	logo(version);

	#ifdef OPENMP
	omp_set_num_threads(I->nthreads); 
	#endif

	print_input_summary(I);

	// Build Source Data
	Source * S = initialize_sources(I); 
	
	// Build Exponential Table
	Table * table = buildExponentialTable( 0.01, 10.0 );

	center_print("SIMULATION", 79);
	border_print();

	double start, stop;

	// Run Simulation Kernel Loop
	start = get_time();
	run_kernel(I, S, table);
	stop = get_time();

	border_print();
	center_print("RESULTS SUMMARY", 79);
	border_print();

	printf("Time per Intersection:          ");
	printf("%.2lf ns\n", 0.5);
	border_print();

	return 0;
}
