#include "utils.h"

__global__ void cyclic_slice(const int nthreads, const float *idata, float *odata,
                           const int width, const int height, const int inputSize)
{

    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int x0 = index % width;
        int y0 = (index / width) % height;

        int x2 = y0;
        int y2 = width - x0 -1;
        int width2 = height;
        int x3 = width - x0 -1;
        int y3 = height - y0 -1;
        int width3 = width;
        int x4 = height - y0 -1;
        int y4 = x0;
        int width4 = height;

        int offset = (index/width/height)*width*height;
        int ind_out1 = index;
        int ind_out2 = offset+y2*width2 + x2 + inputSize;
        int ind_out3 = offset+y3*width3 + x3 + inputSize*2;
        int ind_out4 = offset+y4*width4 + x4 + inputSize*3;

        //tile[threadIdx.x] = idata[index];
        //__syncthreads();
        float tmp = idata[index];

        odata[ind_out1] = tmp;
        odata[ind_out2] = tmp;
        odata[ind_out3] = tmp;
        odata[ind_out4] = tmp;
    }
}

__global__ void cyclic_slice_gradinput(const int nthreads, float *gradInputData, 
		const float *gradOutputData, const int width, const int height, 
		const int inputSize)
{
    //__shared__ float tile[CUDA_NUM_THREADS+32];

    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int x0 = index % width;
        int y0 = (index / width) % height;

        int x1 = x0;
        int y1 = y0;
        int width1 = height;
        int x2 = y0;
        int y2 = width - x0 -1;
        int width2 = height;
        int x3 = width - x0 -1;
        int y3 = height - y0 -1;
        int width3 = width;
        int x4 = height - y0 -1;
        int y4 = x0;
        int width4 = height;

        int offset = (index/width/height)*width*height;
        int ind_out1 = offset+y1*width1 + x1;
        int ind_out2 = offset+y2*width2 + x2 + inputSize;
        int ind_out3 = offset+y3*width3 + x3 + inputSize*2;
        int ind_out4 = offset+y4*width4 + x4 + inputSize*3;

        float tmp  = gradOutputData[ind_out1];
        tmp += gradOutputData[ind_out2];
        tmp += gradOutputData[ind_out3];
        tmp += gradOutputData[ind_out4];
		gradInputData[index] = tmp;
    }
}

static int cunn_CyclicSlice_updateOutput(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

    float *output_data;
    float *input_data;

    luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

    int width, height;
    int nDim = input->nDimension;
    height = input->size[nDim-2];
    width = input->size[nDim-1];

    THCudaTensor_zero(state, output);

    int count = THCudaTensor_nElement(state, input);
    input = THCudaTensor_newContiguous(state, input);

    output_data = THCudaTensor_data(state, output);
    input_data = THCudaTensor_data(state, input);

    cyclic_slice<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>
    (count, input_data, output_data, width, height, count);

    THCudaTensor_free(state, input);

    // check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in FracMaxPoolingForward.updateOutput: %s\n", cudaGetErrorString(err));
        THError("aborting");
    }
    return 1;
}

static int cunn_CyclicSlice_updateGradInput(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

    luaL_argcheck(L, gradOutput->nDimension == 3 || gradOutput->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

    float *gradInput_data;
    float *gradOutput_data;

    int width, height;
    int nDim = input->nDimension;
    height = input->size[nDim-2];
    width = input->size[nDim-1];

    gradOutput = THCudaTensor_newContiguous(state, gradOutput);
    int count =  THCudaTensor_nElement(state, input);
    THCudaTensor_resizeAs(state, gradInput, input);
    THCudaTensor_zero(state, gradInput);

    gradOutput_data = THCudaTensor_data(state, gradOutput);
    gradInput_data = THCudaTensor_data(state, gradInput);

    cyclic_slice_gradinput<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>
    (count, gradInput_data, gradOutput_data, width, height, count);

    // check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in CyclicSlice.updateGradInput: %s\n", cudaGetErrorString(err));
        THError("aborting");
    }
    // clean
    THCudaTensor_free(state, gradOutput);
    return 1;
}

static const struct luaL_Reg cunn_CyclicSlice__ [] = {
    {"CyclicSlice_updateOutput", cunn_CyclicSlice_updateOutput},
    {"CyclicSlice_updateGradInput", cunn_CyclicSlice_updateGradInput},
    {NULL, NULL}
};

static void cunnx_CyclicSlice_init(lua_State *L)
{
    luaT_pushmetatable(L, "torch.CudaTensor");
    luaT_registeratname(L, cunn_CyclicSlice__, "nn");
    lua_pop(L,1);
}
