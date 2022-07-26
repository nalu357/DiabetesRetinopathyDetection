#include "utils.h"

struct four_sigmoidupdateOutput_functor
{
  const double alpha;
  float th1;
  float th2;
  float th3;
  float th4;
  four_sigmoidupdateOutput_functor (double _alpha): alpha(_alpha)
  {
	  th1 = 1.5;
	  th2 = 2.5;
	  th3 = 3.5;
	  th4 = 4.5;
  }

  __host__ __device__ float operator()(const float& input) const
  {
	float out1 = 1./(1.+exp(-(input-th1)*alpha));
	float out2 = 1./(1.+exp(-(input-th2)*alpha));
	float out3 = 1./(1.+exp(-(input-th3)*alpha));
	float out4 = 1./(1.+exp(-(input-th4)*alpha));
    return out1+out2+out3+out4;
  }
};

static int cunn_FourSigmoid_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 2, input, output));

  double alpha = luaT_getfieldchecknumber(L, 1, "alpha");
  long size = THCudaTensor_nElement(state, input);

  input = THCudaTensor_newContiguous(state, input);
  THCudaTensor_resizeAs(state, output, input);

  thrust::device_ptr<float> output_data(THCudaTensor_data(state, output));
  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::transform(input_data, input_data+size, output_data,
                    four_sigmoidupdateOutput_functor(alpha));

  THCudaTensor_free(state, input);

  return 1;
}

struct four_sigmoidupdateGradInput_functor
{
  const double alpha;
  float th1;
  float th2;
  float th3;
  float th4;

  four_sigmoidupdateGradInput_functor (double _alpha): alpha(_alpha)
  {
	  th1 = 1.5;
	  th2 = 2.5;
	  th3 = 3.5;
	  th4 = 4.5;
  }

  __host__ __device__ float operator()(const float& input, const float& gradOutput) const
  {
	float out1 = 1./(1+exp(-(input-th1)*alpha));
	float out2 = 1./(1+exp(-(input-th2)*alpha));
	float out3 = 1./(1+exp(-(input-th3)*alpha));
	float out4 = 1./(1+exp(-(input-th4)*alpha));
    return gradOutput * alpha*((1.-out1) * out1 + (1-out2)*out2 + (1-out3)*out3 + (1-out4)*out4);
  }
};

static int cunn_FourSigmoid_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  long size = THCudaTensor_nElement(state, input);
  double alpha = luaT_getfieldchecknumber(L, 1, "alpha");

  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  input = THCudaTensor_newContiguous(state, input);
  THCudaTensor_resizeAs(state, gradInput, input);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> gradOutput_data(THCudaTensor_data(state, gradOutput));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(state, gradInput));
  thrust::transform(input_data, input_data+size, gradOutput_data, gradInput_data, four_sigmoidupdateGradInput_functor(alpha));

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, gradOutput);

  return 1;
}

static const struct luaL_Reg cunn_FourSigmoid__ [] = {
  {"FourSigmoid_updateOutput", cunn_FourSigmoid_updateOutput},
  {"FourSigmoid_updateGradInput", cunn_FourSigmoid_updateGradInput},
  {NULL, NULL}
};

static void cunnx_FourSigmoid_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_FourSigmoid__, "nn");
  lua_pop(L,1);
}
