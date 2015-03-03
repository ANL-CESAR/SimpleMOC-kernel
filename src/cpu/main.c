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

    #ifdef OFFLOAD
	// Run Simulation Kernel Loop
    int n_d = _Offload_number_of_devices(); 
    int i;
    if(n_d < 1){
        printf("No devices available for offload\n");
        return(2);
    }

    char *signal = (char *) malloc(sizeof(char) * n_d);
    printf("Copying data to %d MICs...",n_d);
	start = get_time();
    send_structs(I, S, table); 
	stop = get_time();
    printf("done in %f seconds.\n",stop-start);
    #endif

    printf("Running kernel...");
	start = get_time();
    #ifdef OFFLOAD
    for(i=0; i<n_d; i++){
        #pragma offload target(mic:i) \
        in(I[0:0], S[0:0], table[0:0]) signal(&signal[i])
        {
	        run_kernel(I, S, table);
        }
    }

    for(i=0; i<n_d; i++){
        #pragma offload_wait target(mic:i) wait(&signal[i])
    }
    #else
	run_kernel(I, S, table);
    #endif
	stop = get_time();
    double kernel_time = stop-start;
    printf("done.\n");

    #ifdef OFFLOAD
    printf("Copying from %d MICs...",n_d);
	start = get_time();
    get_structs(I, S, table); 
	stop = get_time();
    printf("done in %f seconds.\n",stop-start);
    #endif

	border_print();
	center_print("RESULTS SUMMARY", 79);
	border_print();

	double tpi = ((double) (kernel_time) /
			(double)I->segments / (double) I->egroups) * 1.0e9;
	printf("%-25s%.3lf seconds\n", "Runtime:", kernel_time);
	printf("%-25s%.3lf ns\n", "Time per Intersection:", tpi);
	border_print();

	return 0;
}
