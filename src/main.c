#include<SimpleMOC-kernel_header.h>

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
	omp_set_num_threads(input.nthreads); 
	#endif

	print_input_summary(input);

	// Build Source Data
	Source * sources = initialize_sources(I); 
	
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

	long tracks_per_second = input.ntracks/time_transport;

	printf("Time per Intersection:          ");
	printf("%.2lf ns\n", time_per_intersection( input, time_transport ));
	border_print();

	return 0;
}
