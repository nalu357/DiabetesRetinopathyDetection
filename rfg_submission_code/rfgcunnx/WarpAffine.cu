#include "utils.h"
#include <TH.h>
#include <npp.h>
#include <cuda_runtime.h>


static int cunn_WarpAffine_updateOutput(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
    THFloatTensor *coeffs = (THFloatTensor *)luaT_checkudata(L, 3, "torch.FloatTensor");
    //THFloatTensor *roi = (THFloatTensor *)luaT_checkudata(L, 4, "torch.FloatTensor");

    THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

    int outWidth = luaT_getfieldcheckint(L, 1, "outWidth");
    int outHeight = luaT_getfieldcheckint(L, 1, "outHeight");


    luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");



    int nDim = input->nDimension;
    int height = input->size[nDim-2];
    int width = input->size[nDim-1];
    int nChannel = input->size[nDim-3];
    //int outImgSz = nChannel*outWidth*outHeight;
    //int srcImgSz = nChannel*width*height;
    int outImgSz = outWidth*outHeight;
    int srcImgSz = width*height;
    int nBatch = input->size[0];

    double nppCoeffs[2][3];

    float * coeff_data =  THFloatTensor_data(coeffs);

    if (nDim == 3)
    {
        THCudaTensor_resize3d(state, output, input->size[0], outHeight, outWidth );
        nBatch = 1;
    }
    else
    {
        THCudaTensor_resize4d(state, output, input->size[0], input->size[1], outHeight, outWidth );
    }

    luaL_argcheck(L, (coeffs->nDimension == 3 && coeffs->size[0] == nBatch) || (coeffs->nDimension == 2), 3, "coeffs must be [nBatch][2][3] or [2][3]");

    int count = THCudaTensor_nElement(state, input);
    input = THCudaTensor_newContiguous(state, input);
    THCudaTensor_zero(state, output);

    float * output_data = THCudaTensor_data(state, output);
    float * input_data = THCudaTensor_data(state, input);
    //float * roi_data = THFloatTensor_data(roi);

    NppiSize src_size = {width, height};
    NppiRect src_roi = {0, 0, width, height};
    NppiRect dst_roi = {0, 0, outWidth, outHeight};
    //dst_roi.x = (int) roi_data[0];
    //dst_roi.y = (int) roi_data[1];

    int k = 0;
    for (int i = 0 ; i < nBatch ; i++)
    {
        if (coeffs->nDimension >2 || k == 0)
        {
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    nppCoeffs[i][j] = coeff_data[k++];
        }

        /*
        if (roi->size[0]== nBatch)
        {
        	dst_roi.x = (int) roi_data[i*2];
        	dst_roi.y = (int) roi_data[i*2+1];
        }
        printf("roi x y %d %d\n", dst_roi.x, dst_roi.y);
        */

        int rval;
		int index = 0;
        for (int j= 0; j < nChannel/3; j++)
        {
            const Npp32f * pSrc[3];
            Npp32f * pDst[3];
            pSrc[0] = (Npp32f*) (input_data);
            pSrc[1] = (Npp32f*) (input_data + srcImgSz);
            pSrc[2] = (Npp32f*) (input_data + 2*srcImgSz);
            pDst[0] = (Npp32f*) output_data;
            pDst[1] = (Npp32f*) (output_data + outImgSz);
            pDst[2] = (Npp32f*) (output_data + 2*outImgSz);
            rval=nppiWarpAffine_32f_P3R(pSrc, src_size, width*sizeof(float), src_roi, pDst, outWidth*sizeof(float), dst_roi, nppCoeffs, NPPI_INTER_CUBIC);
			if (NPP_NO_ERROR != rval)
			{
				fprintf(stderr, "NPP error %d\n", rval);
				for (int i = 0; i < 2; i++)
					for (int j = 0; j < 3; j++)
						printf("%f ", nppCoeffs[i][j]);
				printf("\n");
				exit(1);
			}
			input_data += 3*srcImgSz;
			output_data += 3*outImgSz;
			index += 3;
		}

        for (int j = index; j < nChannel; j++)
        {
            rval=nppiWarpAffine_32f_C1R((Npp32f *) input_data, src_size, width*sizeof(float), src_roi, (Npp32f *) output_data, outWidth*sizeof(float), dst_roi, nppCoeffs, NPPI_INTER_CUBIC);
			if (NPP_NO_ERROR != rval)
			{
				fprintf(stderr, "NPP error %d\n", rval);
				exit(1);
			}
			input_data += srcImgSz;
			output_data += outImgSz;
		}
    }

    THCudaTensor_free(state, input);

    // check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in WarpAffineForward.updateOutput: %s\n", cudaGetErrorString(err));
        THError("aborting");
    }
    return 1;
}

static const struct luaL_Reg cunn_WarpAffine__ [] = {
    {"WarpAffine_updateOutput", cunn_WarpAffine_updateOutput},
    {NULL, NULL}
};

static void cunnx_WarpAffine_init(lua_State *L)
{
    luaT_pushmetatable(L, "torch.CudaTensor");
    luaT_registeratname(L, cunn_WarpAffine__, "nn");
    lua_pop(L,1);
}
