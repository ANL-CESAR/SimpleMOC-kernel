#include "SimpleMOC-kernel_header.h"

int main( int argc, char * argv[] )
{
	int version = 2;

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
	printf("Attentuating fluxes across segments...\n");

	double start, stop;

	// Run Simulation Kernel Loop
	start = get_time();
    #ifdef OFFLOAD
    //int n_d = _Offload_number_of_devices(); 
    int n_d = 1; 
    int i;
    if(n_d < 1){
        printf("No devices available for offload\n");
        return(2);
    }

    send_structs(I, S, table); 
    for(i=0; i<n_d; i++){
        #pragma offload target(mic:i) \
        in(I[0:0], S[0:0], table[0:0])
        {
	        run_kernel(I, S, table);
        }
    }
    get_structs(I, S, table);

    #else
	run_kernel(I, S, table);
    #endif
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
