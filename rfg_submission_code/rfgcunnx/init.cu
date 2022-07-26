#include "luaT.h"
#include "THC.h"
#include "THLogAdd.h" /* DEBUG: WTF */

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include "cublas_v2.h"
#define CudaAssert( expression ) \
if ( !(expression)) { \
printf( "Assert failed %d:%d at %s:%d\n", blockIdx.x, threadIdx.x,  __FILE__, __LINE__ ); \
}

#include "utils.c"
#include "VeryLeakyReLU.cu"
#include "FracSpatialMaxPooling.cu"
#include "CyclicSlice.cu"
#include "CyclicRoll.cu"
#include "CyclicPool.cu"
#include "WarpAffine.cu"
#include "FourSigmoid.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_librfgcunnx(lua_State *L);

int luaopen_librfgcunnx(lua_State *L)
{
  lua_newtable(L);
  cunnx_VeryLeakyReLU_init(L);
  cunnx_FracSpatialMaxPooling_init(L);
  cunnx_CyclicSlice_init(L);
  cunnx_CyclicPool_init(L);
  cunnx_CyclicRoll_init(L);
  cunnx_WarpAffine_init(L);
  cunnx_FourSigmoid_init(L);
  return 1;
}
