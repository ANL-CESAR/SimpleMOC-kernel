#include "SimpleMOC-kernel_header.h"

int main( int argc, char * argv[] )
{
	int version = 0;

	srand(time(NULL));

	Input I = set_default_input();
	read_CLI( argc, argv, &I );

	logo(version);

	print_input_summary(I);

	// Build Source Data
	Source_Arrays SA_h, SA_d;
	Source * sources_h = initialize_sources(I, &SA_h); 
	Source * sources_d = initialize_device_sources( I, SA_h, SA_d, sources_h); 
	
	// Build Exponential Table
	Table table = buildExponentialTable();
	Table * table_d;
	cudaMalloc((void **) &table_d, sizeof(Table));
	cudaMemcpy(table_d, &table, sizeof(Table), cudaMemcpyHostToDevice);

	center_print("SIMULATION", 79);
	border_print();
	printf("Attentuating fluxes across segments...\n");

	double start, stop;

	// Run Simulation Kernel Loop
	start = get_time();
	int block_size = 32;
	int n_blocks = I.segments/block_size + (I.segments%block_size == 0 ? 0:1);
	square_array <<< n_blocks, block_size >>> (I, sources_d, SA_d, table_d);
	stop = get_time();

	printf("Simulation Complete.\n");

	border_print();
	center_print("RESULTS SUMMARY", 79);
	border_print();

	double tpi = ((double) (stop - start) /
			(double)I.segments / (double) I.egroups) * 1.0e9;
	printf("%-25s%.3lf seconds\n", "Runtime:", stop-start);
	printf("%-25s%.3lf ns\n", "Time per Intersection:", tpi);
	border_print();

	return 0;
}
