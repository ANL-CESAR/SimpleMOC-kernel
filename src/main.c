#include"SimpleMOC_header.h"

int main( int argc, char * argv[] )
{
	int version = 0;
	int mype = 0;

	#ifdef PAPI
	papi_serial_init();
	#endif

	srand(time(NULL) * (mype+1));

	Input input = set_default_input();
	read_CLI( argc, argv, &input );

	logo(version);

	#ifdef OPENMP
	omp_set_num_threads(input.nthreads); 
	#endif

	print_input_summary(input);

	center_print("SIMULATION", 79);
	border_print();

	double start, stop;

	start = get_time();
	for( int i = 0; i < num_iters; i++)
	{

	}
	stop = get_time();

	border_print();
	center_print("RESULTS SUMMARY", 79);
	border_print();

	long tracks_per_second = input.ntracks/time_transport;

	printf("Total Tracks per Second:        ");
	fancy_int( tracks_per_second );
	printf("Time per Intersection:          ");
	printf("%.2lf ns\n", time_per_intersection( input, time_transport ));
	border_print();

	return 0;
}
