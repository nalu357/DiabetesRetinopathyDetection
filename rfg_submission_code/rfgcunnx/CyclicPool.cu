#include "utils.h"

__global__ void cyclic_pool(const int nthreads, const float *idata, 
		float *odata, const int width, const int height, const int outputSize)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int x0 = index % width;
        int y0 = (index / width) % height;
        int x2, y2, x3, y3, x4, y4;
        int width2, width3, width4;

        x2 = y0;
        y2 = width - x0 -1;
        width2 = height;

        x3 = width - x0 -1;
        y3 = height - y0 -1;
        width3 = width;

        x4 = height - y0 -1;
        y4 = x0;
        width4 = height;

        int offset = (index/width/height)*width*height;
        int ind_out1 = index;
        int ind_out2 = offset+y2*width2 + x2 + outputSize;
        int ind_out3 = offset+y3*width3 + x3 + outputSize*2;
        int ind_out4 = offset+y4*width4 + x4 + outputSize*3;

        float tmp = idata[ind_out1] * idata[ind_out1];
        tmp += idata[ind_out2] * idata[ind_out2];
        tmp += idata[ind_out3] * idata[ind_out3];
        tmp += idata[ind_out4] * idata[ind_out4];
		odata[index] = sqrt(tmp/4);
    }
}

__global__ void cyclic_pool_gradInput(const int nthreads, const float * input, 
		const float * output, float *gradInput, const float *gradOutput, 
		const int width, const int height, const int outputSize)
{

    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int x0 = index % width;
        int y0 = (index / width) % height;
        int x2, y2, x3, y3, x4, y4;
        int width2, width3, width4;

        x2 = y0;
        y2 = width - x0 -1;
        width2 = height;

        x3 = width - x0 -1;
        y3 = height - y0 -1;
        width3 = width;

        x4 = height - y0 -1;
        y4 = x0;
        width4 = height;

        int offset = (index/width/height)*width*height;
        int ind_out1 = index;
        int ind_out2 = offset+y2*width2 + x2 + outputSize;
        int ind_out3 = offset+y3*width3 + x3 + outputSize*2;
        int ind_out4 = offset+y4*width4 + x4 + outputSize*3;

        float tmp  = gradOutput[index];
        tmp /= (4*output[index]);

        gradInput[ind_out1] = tmp*input[ind_out1];
        gradInput[ind_out2] = tmp*input[ind_out2];
        gradInput[ind_out3] = tmp*input[ind_out3];
        gradInput[ind_out4] = tmp*input[ind_out4];
    }
}

static int cunn_CyclicPool_updateOutput(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

    luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

    int nDim = input->nDimension;
    int height = input->size[nDim-2];
    int width = input->size[nDim-1];

    int count = THCudaTensor_nElement(state, output);
    input = THCudaTensor_newContiguous(state, input);

    float * input_data = THCudaTensor_data(state, input);
    float * output_data = THCudaTensor_data(state, output);

    cyclic_pool<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>
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

static int cunn_CyclicPool_updateGradInput(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

    THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

    luaL_argcheck(L, gradOutput->nDimension == 3 || gradOutput->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

    int nDim = gradOutput->nDimension;
    int height = gradOutput->size[nDim-2];
    int width = gradOutput->size[nDim-1];

    gradOutput = THCudaTensor_newContiguous(state, gradOutput);
    input = THCudaTensor_newContiguous(state, input);
    //output = THCudaTensor_newContiguous(state, output);

    int count =  THCudaTensor_nElement(state, gradOutput);
    int count_input =  THCudaTensor_nElement(state, input);
    THCudaTensor_resizeAs(state, gradInput, input);
    THCudaTensor_zero(state, gradInput);

    float * gradOutput_data = THCudaTensor_data(state, gradOutput);
    float * gradInput_data = THCudaTensor_data(state, gradInput);
    float * output_data = THCudaTensor_data(state, output);
    float * input_data = THCudaTensor_data(state, input);

    cyclic_pool_gradInput<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>
    (count, input_data, output_data, gradInput_data, gradOutput_data, width, height, count);

    // check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in CyclicPool.updateGradInput: %s\n", cudaGetErrorString(err));
        THError("aborting");
    }
    // clean
    THCudaTensor_free(state, gradOutput);
    THCudaTensor_free(state, input);
    //THCudaTensor_free(state, output);
    return 1;
}

static const struct luaL_Reg cunn_CyclicPool__ [] = {
    {"CyclicPool_updateOutput", cunn_CyclicPool_updateOutput},
    {"CyclicPool_updateGradInput", cunn_CyclicPool_updateGradInput},
    {NULL, NULL}
};

static void cunnx_CyclicPool_init(lua_State *L)
{
    luaT_pushmetatable(L, "torch.CudaTensor");
    luaT_registeratname(L, cunn_CyclicPool__, "nn");
    lua_pop(L,1);
}
