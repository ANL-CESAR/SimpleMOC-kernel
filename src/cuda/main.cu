#include "SimpleMOC-kernel_header.h"

int main( int argc, char * argv[] )
{
	int version = 4;

	srand(time(NULL));

	Input I = set_default_input();
	read_CLI( argc, argv, &I );
	
	// Calculate Number of 3D Source Regions
	I.source_3D_regions = (int) ceil((double)I.source_2D_regions *
		I.coarse_axial_intervals / I.decomp_assemblies_ax);

	logo(version);

	print_input_summary(I);
	
	center_print("INITIALIZATION", 79);
	border_print();

	// Build Source Data
	printf("Building Source Data Arrays...\n");
	Source_Arrays SA_h, SA_d;
	Source * sources_h = initialize_sources(I, &SA_h); 
	Source * sources_d = initialize_device_sources( I, &SA_h, &SA_d, sources_h); 
	cudaDeviceSynchronize();
	
	// Build Exponential Table
	Table * table_d = NULL;
	#ifdef TABLE
	printf("Building Exponential Table...\n");
	Table table = buildExponentialTable();
	CUDA_CALL( cudaMalloc((void **) &table_d, sizeof(Table)) );
	CUDA_CALL( cudaMemcpy(table_d, &table, sizeof(Table), cudaMemcpyHostToDevice) );
	#endif

	// Setup CUDA blocks / threads
	int n_blocks = sqrt(I.segments);
	dim3 blocks(n_blocks, n_blocks);
	if( blocks.x * blocks.y < I.segments )
		blocks.x++;
	if( blocks.x * blocks.y < I.segments )
		blocks.y++;
	assert( blocks.x * blocks.y >= I.segments );

	// Setup CUDA RNG on Device
	printf("Setting up CUDA RNG...\n");
	curandState * RNG_states;
	CUDA_CALL( cudaMalloc((void **)&RNG_states, I.streams * sizeof(curandState)) );
	setup_kernel<<<I.streams/100 + 1, 100>>>(RNG_states, I);
	CudaCheckError();
	cudaDeviceSynchronize();

	// Allocate Some Flux State vectors to randomly pick from
	printf("Setting up Flux State Vectors...\n");
	float * flux_states;
	int N_flux_states = 10000;
	assert( I.segments >= N_flux_states );
	CUDA_CALL( cudaMalloc((void **) &flux_states, N_flux_states * I.egroups * sizeof(float)) );
	init_flux_states<<< blocks, I.egroups >>> ( flux_states, N_flux_states, I, RNG_states );


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
	
	// Setup kernel call block parameters
	assert( I.segments % I.seg_per_thread == 0 );
	n_blocks = sqrt(I.segments / I.seg_per_thread);
	dim3 blocks_k(n_blocks, n_blocks);
	if( blocks_k.x * blocks_k.y < I.segments / I.seg_per_thread )
		blocks_k.x++;
	if( blocks_k.x * blocks_k.y < I.segments / I.seg_per_thread )
		blocks_k.y++;
	assert( blocks_k.x * blocks_k.y >= I.segments / I.seg_per_thread );

	// Run Simulation Kernel Loop
	cudaEventRecord(start, 0);
	run_kernel <<< blocks_k, I.egroups, I.seg_per_thread * 3 *sizeof(int) >>> (I, sources_d, SA_d, table_d, 
			RNG_states, flux_states, N_flux_states);
	CudaCheckError();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaDeviceSynchronize();

	float * host_flux_states = (float*) malloc(N_flux_states * I.egroups * sizeof(float));
	CUDA_CALL( cudaMemcpy( host_flux_states, flux_states, N_flux_states * I.egroups * sizeof(float), cudaMemcpyDeviceToHost));

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
