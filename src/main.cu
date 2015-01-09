#include "SimpleMOC-kernel_header.h"

int main( int argc, char * argv[] )
{
	int version = 0;

	srand(time(NULL));

	Input I = set_default_input();
	read_CLI( argc, argv, &I );

	logo(version);

	print_input_summary(I);
	
	center_print("INITIALIZATION", 79);
	border_print();

	printf("Building Source Data Arrays...\n");
	// Build Source Data
	Source_Arrays SA_h, SA_d;
	Source * sources_h = initialize_sources(I, &SA_h); 
	Source * sources_d = initialize_device_sources( I, &SA_h, &SA_d, sources_h); 
	cudaDeviceSynchronize();
	
	printf("Building Exponential Table...\n");
	// Build Exponential Table
	Table table = buildExponentialTable();
	Table * table_d;
	CUDA_CALL( cudaMalloc((void **) &table_d, sizeof(Table)) );
	CUDA_CALL( cudaMemcpy(table_d, &table, sizeof(Table), cudaMemcpyHostToDevice) );

	// Setup CUDA blocks / threads
	int n_blocks = sqrt(I.segments);
	dim3 blocks(n_blocks, n_blocks);
	if( blocks.x * blocks.y < I.segments )
		blocks.x++;
	if( blocks.x * blocks.y < I.segments )
		blocks.y++;
	assert( blocks.x * blocks.y >= I.segments );
	
	CudaCheckError();
	printf("Setting up CUDA RNG...\n");
	// Setup CUDA RNG on Device
	curandState * RNG_states;
	CUDA_CALL( cudaMalloc((void **)&RNG_states, I.streams * sizeof(curandState)) );
	setup_kernel<<<I.streams/100 + 1, 100>>>(RNG_states, I);
	CudaCheckError();

	cudaDeviceSynchronize();

	printf("Setting up Flux State Vectors...\n");
	// Allocate Some Flux State vectors to randomly pick from
	float * flux_states;
	int N_flux_states = 10000;
	assert( I.segments >= N_flux_states );
	CUDA_CALL( cudaMalloc((void **) &flux_states, N_flux_states * I.egroups * sizeof(float)) );
	printf("CUDA ptr = %p\n", flux_states);
	init_flux_states<<< blocks, I.egroups >>> ( flux_states, N_flux_states, I, RNG_states );
	printf("CUDA ptr = %p\n", flux_states);

	printf("Initialization Complete.\n");

	border_print();
	center_print("SIMULATION", 79);
	border_print();
	cudaDeviceSynchronize();
	printf("Attentuating fluxes across segments...\n");

	// CUDA timer variables
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time = 0;

	// Run Simulation Kernel Loop
	cudaEventRecord(start, 0);
	run_kernel <<< blocks, I.egroups >>> (I, sources_d, SA_d, table_d, 
			RNG_states, flux_states, N_flux_states);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaDeviceSynchronize();

	float * host_flux_states = (float*) malloc(N_flux_states * I.egroups * sizeof(float));
	printf("CUDA ptr = %p\n", flux_states);
	CUDA_CALL( cudaMemcpy( host_flux_states, flux_states, N_flux_states * I.egroups * sizeof(float), cudaMemcpyDeviceToHost) );

	printf("Simulation Complete.\n");

	border_print();
	center_print("RESULTS SUMMARY", 79);
	border_print();

	double tpi = ((double) (time/1000.0) /
			(double)I.segments / (double) I.egroups) * 1.0e9;
	printf("%-25s%.3f seconds\n", "Runtime:", time / 1000.0);
	printf("%-25s%.3lf ns\n", "Time per Intersection:", tpi);
	border_print();

	return 0;
}
