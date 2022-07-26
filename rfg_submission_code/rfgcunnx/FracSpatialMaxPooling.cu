#include "utils.h"
//implementation of fractional max pooling by Ben Graham, using a combination of code from caffe and torch
template <typename Dtype>
__global__ void FracMaxPoolForward(const int nthreads, const Dtype* bottom_data,
           const int channels, const int height,
           const int width, const int pooled_height, const int pooled_width,
           const Dtype *stride_h_start, const Dtype * stride_h_end, const Dtype *stride_w_start, const Dtype *stride_w_end, Dtype* top_data,
           Dtype* top_mask) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;

        int hstart = max((int)stride_h_start[ph], 0);
        int wstart = max((int)stride_w_start[pw], 0);
        int hend = min((int)stride_h_end[ph], height);
        int wend = min((int)stride_w_end[pw], width);

        int offset = (n * channels + c) * height * width;
		const Dtype * bottom = bottom_data + offset;

        //Dtype maxval = -FLT_MAX;
        //int maxidx = 0;
        int maxidx = hstart * width + wstart;
        Dtype maxval = bottom[maxidx];

        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                if (bottom[h * width + w] > maxval) {
                    maxidx = h * width + w;
                    maxval = bottom[maxidx];
                }
            }
        }
        top_data[index] = maxval;
        top_mask[index] = maxidx + offset;
    }
}

template <typename Dtype>
__global__ void FracMaxPoolBackward(const int nthreads,
                Dtype* top_diff, Dtype* top_mask, Dtype * bottom_diff) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int bottom_index = top_mask[index];
        bottom_diff[bottom_index] +=  top_diff[index];
    }
}

template <typename Dtype>
__global__ void FracMaxPoolBackwardAtomic(const int nthreads,
                Dtype* top_diff, Dtype* top_mask, Dtype * bottom_diff) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int bottom_index = top_mask[index];
        atomicAdd(bottom_diff+bottom_index,top_diff[index]);
    }
}

static int cunn_FracSpatialMaxPooling_updateOutput(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
    int nOutputRows = luaL_checkint(L, 3);
    int nOutputCols = luaL_checkint(L, 4);

    THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
    THCudaTensor *indices = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.CudaTensor");

    THCudaTensor *row_ind_start = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "row_ind_start", "torch.CudaTensor");
    THCudaTensor *row_ind_end = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "row_ind_end", "torch.CudaTensor");

    THCudaTensor *col_ind_start = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "col_ind_start", "torch.CudaTensor");
    THCudaTensor *col_ind_end = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "col_ind_end", "torch.CudaTensor");

    bool olap = luaT_getfieldcheckboolean(L, 1, "olap");

    float *indices_data;
    float *output_data;
    float *input_data;
    float *col_ind_start_data;
    float *col_ind_end_data;
    float *row_ind_start_data;
    float *row_ind_end_data;

    luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

    int nBatch = 1;
    int nInputPlane;
    int nInputRows;
    int nInputCols;

    if (input->nDimension == 3)
    {
        nInputPlane = input->size[0];
        nInputRows = input->size[1];
        nInputCols = input->size[2];
        //THCudaTensor_resize3d(state, output, nInputPlane, nOutputRows, nOutputCols);
    }
    else
    {
        nInputPlane = input->size[1];
        nInputRows = input->size[2];
        nInputCols = input->size[3];
        nBatch =  input->size[0];
        //THCudaTensor_resize4d(state, output, input->size[0], nInputPlane, nOutputRows, nOutputCols);
    }

    int count = nBatch * nInputPlane * nOutputRows * nOutputCols;
    //THCudaTensor_resize1d(state, indices, count);
    //output = THCudaTensor_newContiguous(state, output);
    input = THCudaTensor_newContiguous(state, input);

    indices_data = THCudaTensor_data(state, indices);
    output_data = THCudaTensor_data(state, output);
    input_data = THCudaTensor_data(state, input);
    col_ind_start_data = THCudaTensor_data(state, col_ind_start);
    row_ind_start_data = THCudaTensor_data(state, row_ind_start);
    col_ind_end_data = THCudaTensor_data(state, col_ind_end);
    row_ind_end_data = THCudaTensor_data(state, row_ind_end);

    FracMaxPoolForward<float><<<GET_BLOCKS(count),
         CUDA_NUM_THREADS>>>(count, input_data,
         nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
         row_ind_start_data, row_ind_end_data, col_ind_start_data, col_ind_end_data, output_data, indices_data);

    THCudaTensor_free(state, input);

    // check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in FracMaxPoolingForward.updateOutput: %s\n", cudaGetErrorString(err));
        THError("aborting");
    }
    return 1;
}

static int cunn_FracSpatialMaxPooling_updateGradInput(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
    THCudaTensor *indices = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.CudaTensor");

    luaL_argcheck(L, gradOutput->nDimension == 3 || gradOutput->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

    bool olap = luaT_getfieldcheckboolean(L, 1, "olap");

    float *indices_data;
    float *gradInput_data;
    float *gradOutput_data;

    gradOutput = THCudaTensor_newContiguous(state, gradOutput);
    int count =  THCudaTensor_nElement(state, gradOutput);
    //THCudaTensor_resizeAs(state, gradInput, input);
    //gradInput = THCudaTensor_newContiguous(state, gradInput);
    THCudaTensor_zero(state, gradInput);

    indices_data = THCudaTensor_data(state, indices);
    gradOutput_data = THCudaTensor_data(state, gradOutput);
    gradInput_data = THCudaTensor_data(state, gradInput);

	if (olap)
		FracMaxPoolBackwardAtomic<float><<<GET_BLOCKS(count),
                        CUDA_NUM_THREADS>>>(count, 
                        gradOutput_data, indices_data, gradInput_data);
	else
		FracMaxPoolBackward<float><<<GET_BLOCKS(count),
                        CUDA_NUM_THREADS>>>(count, 
                        gradOutput_data, indices_data, gradInput_data);

    // check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in FracSpatialMaxsampling.updateGradInput: %s\n", cudaGetErrorString(err));
        THError("aborting");
    }
    // clean
    THCudaTensor_free(state, gradOutput);
    return 1;
}

static const struct luaL_Reg cunn_FracSpatialMaxPooling__ [] = {
    {"FracSpatialMaxPooling_updateOutput", cunn_FracSpatialMaxPooling_updateOutput},
    {"FracSpatialMaxPooling_updateGradInput", cunn_FracSpatialMaxPooling_updateGradInput},
    {NULL, NULL}
};

static void cunnx_FracSpatialMaxPooling_init(lua_State *L)
{
    luaT_pushmetatable(L, "torch.CudaTensor");
    luaT_registeratname(L, cunn_FracSpatialMaxPooling__, "nn");
    lua_pop(L,1);
}
