#include "utils.h"

//wrapper code of cyclic pool based on Sanders' kernel 

__global__ void cyclic_roll(float * input, float * output, int batch_size, int num_features) {
    int x = blockIdx.x*blockDim.x + threadIdx.x; // feature dim, fastest varying index!
    int y = blockIdx.y*blockDim.y + threadIdx.y; // batch dim
    int height = 4 * batch_size;
    int width = 4 * num_features;
    if (x < num_features && y < height) {
        for (int i = 0; i < 4; i++) {
            int y_out = (y + batch_size * (4 - i)) % height;
            int x_out = x + num_features * i;
            output[y_out * width + x_out] = input[y * num_features + x];
        }
    }
}

__global__ void cyclic_roll_grad(float * input, float * output, int batch_size, int num_features) {
    int x = blockIdx.x*blockDim.x + threadIdx.x; // feature dim, fastest varying index!
    int y = blockIdx.y*blockDim.y + threadIdx.y; // batch dim
    int height = 4 * batch_size;
    int width = 4 * num_features;
    float val = 0;
    if (x < num_features && y < height) {
        for (int i = 0; i < 4; i++) {
            int y_in = (y + batch_size * (4 - i)) % height;
            int x_in = x + num_features * i;
            val += input[y_in * width + x_in];
        }
        output[y * num_features + x] = val;
    }
}

__global__ void cyclic_convroll(float * input, float * output, int batch_size, int num_channels, int map_size) {
    int x = blockIdx.x*blockDim.x + threadIdx.x; // feature dim, fastest varying index!
    int y = blockIdx.y*blockDim.y + threadIdx.y; // batch dim
    int map_size_sq = map_size * map_size;
    int example_size = num_channels * map_size_sq;
    int num_rows = 4 * batch_size; // number of rows in the input/output, seen as a 2D array
    int num_cols = 4 * example_size; // number of columns in the output, seen as a 2D array
    // feature indices (channels, height, width)
    int x_channel = x / map_size_sq;
    int x_f0 = (x % map_size_sq) / map_size;
    int x_f1 = x % map_size;
    int x_out_f0 = x_f0;
    int x_out_f1 = x_f1;
    int tmp;
    if (x < example_size && y < num_rows) {
        for (int i = 0; i < 4; i++) {
            int y_out = (y + batch_size * (4 - i)) % num_rows;
            int x_out = example_size * i + x_channel * map_size_sq + x_out_f0 * map_size + x_out_f1;
            output[y_out * num_cols + x_out] = input[y * example_size + x];
            // note that the writes to output go in reverse order for all the rotated feature maps.
            // this may slow things down a little, perhaps there is room for further optimization.
            // rotate
            tmp = x_out_f0;
            x_out_f0 = x_out_f1;
            x_out_f1 = map_size - 1 - tmp;
        }
    }
}

__global__ void cyclic_convroll_grad(float * input, float * output, int batch_size, int num_channels, int map_size) {
    int x = blockIdx.x*blockDim.x + threadIdx.x; // feature dim, fastest varying index!
    int y = blockIdx.y*blockDim.y + threadIdx.y; // batch dim
    int map_size_sq = map_size * map_size;
    int example_size = num_channels * map_size_sq;
    int num_rows = 4 * batch_size; // number of rows in the input/output, seen as a 2D array
    int num_cols = 4 * example_size; // number of columns in the input, seen as a 2D array
    // feature indices (channels, height, width)
    int x_channel = x / map_size_sq;
    int x_f0 = (x % map_size_sq) / map_size;
    int x_f1 = x % map_size;
    int x_in_f0 = x_f0;
    int x_in_f1 = x_f1;
    int tmp;
    float val;
    if (x < example_size && y < num_rows) {
        for (int i = 0; i < 4; i++) {
            int y_in = (y + batch_size * (4 - i)) % num_rows;
            int x_in = example_size * i + x_channel * map_size_sq + x_in_f0 * map_size + x_in_f1;
            val += input[y_in * num_cols + x_in];
            // rotate
            tmp = x_in_f0;
            x_in_f0 = x_in_f1;
            x_in_f1 = map_size - 1 - tmp;
        }
        output[y * example_size + x] = val;
    }
}

static int cunn_CyclicRoll_updateOutput(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

    luaL_argcheck(L, input->nDimension == 4, 2, "4D (batch) tensor expected");

    int count = THCudaTensor_nElement(state, input);
    input = THCudaTensor_newContiguous(state, input);

    float * output_data = THCudaTensor_data(state, output);
    float * input_data = THCudaTensor_data(state, input);

    int nDim = input->nDimension;
    int height = input->size[nDim-2];
    int width = input->size[nDim-1];
	int nBatch = input->size[0];
	int batch_size = nBatch/4;
	int num_channels = input->size[1];
	int map_size = height;
	int example_size = num_channels * height * width;
	int full_batch_size = nBatch;

	int x_block = 32;
	int y_block = 32;
	int x_grid = ceil(float(example_size) / x_block );
    int y_grid = ceil(float(full_batch_size) / y_block);
    dim3 blocks(x_block, y_block);
    dim3 grids(x_grid, y_grid);

    // run kernel
     cyclic_convroll<<<grids, blocks>>>(input_data, output_data, batch_size, num_channels, map_size);

    THCudaTensor_free(state, input);

    // check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in FracMaxPoolingForward.updateOutput: %s\n", cudaGetErrorString(err));
        THError("aborting");
    }
    return 1;
}

static int cunn_CyclicRoll_updateGradInput(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

    luaL_argcheck(L, gradOutput->nDimension == 4, 2, "4D (batch) tensor expected");

    float *gradInput_data;
    float *gradOutput_data;

    gradOutput = THCudaTensor_newContiguous(state, gradOutput);
    int count =  THCudaTensor_nElement(state, gradOutput);
    THCudaTensor_resizeAs(state, gradInput, input);
    THCudaTensor_zero(state, gradInput);

    gradOutput_data = THCudaTensor_data(state, gradOutput);
    gradInput_data = THCudaTensor_data(state, gradInput);

	int width, height;
    int nDim = input->nDimension;
    height = input->size[nDim-2];
    width = input->size[nDim-1];
	int nBatch = input->size[0];
	int batch_size = nBatch/4;
	int num_channels = input->size[1];
	int map_size = height;
	int example_size = num_channels * height * width;
	int full_batch_size = nBatch;

	int x_block = 32;
	int y_block = 32;
	int x_grid = ceil(float(example_size) / x_block );
    int y_grid = ceil(float(full_batch_size) / y_block);
    dim3 blocks(x_block, y_block);
    dim3 grids(x_grid, y_grid);

	cyclic_convroll_grad<<<grids, blocks>>>(gradOutput_data, gradInput_data, batch_size, num_channels, map_size);

    // check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in CyclicRoll.updateGradInput: %s\n", cudaGetErrorString(err));
        THError("aborting");
    }
    // clean
    THCudaTensor_free(state, gradOutput);
    return 1;
}

static const struct luaL_Reg cunn_CyclicRoll__ [] = {
    {"CyclicRoll_updateOutput", cunn_CyclicRoll_updateOutput},
    {"CyclicRoll_updateGradInput", cunn_CyclicRoll_updateGradInput},
    {NULL, NULL}
};

static void cunnx_CyclicRoll_init(lua_State *L)
{
    luaT_pushmetatable(L, "torch.CudaTensor");
    luaT_registeratname(L, cunn_CyclicRoll__, "nn");
    lua_pop(L,1);
}

