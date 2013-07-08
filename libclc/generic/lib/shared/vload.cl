#include <clc/clc.h>

#define VLOAD_VECTORIZE(PRIM_TYPE, ADDR_SPACE) \
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##2 vload2(size_t offset, const ADDR_SPACE PRIM_TYPE *x) { \
    return (PRIM_TYPE##2)(x[offset] , x[offset+1]); \
  } \
\
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##3 vload3(size_t offset, const ADDR_SPACE PRIM_TYPE *x) { \
    return (PRIM_TYPE##3)(x[offset] , x[offset+1], x[offset+2]); \
  } \
\
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##4 vload4(size_t offset, const ADDR_SPACE PRIM_TYPE *x) { \
    return (PRIM_TYPE##4)(x[offset], x[offset+1], x[offset+2], x[offset+3]); \
  } \
\
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##8 vload8(size_t offset, const ADDR_SPACE PRIM_TYPE *x) { \
    return (PRIM_TYPE##8)(vload4(offset, x), vload4(offset+4, x)); \
  } \
\
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##16 vload16(size_t offset, const ADDR_SPACE PRIM_TYPE *x) { \
    return (PRIM_TYPE##16)(vload8(offset, x), vload8(offset+8, x)); \
  } \

#define VLOAD_ADDR_SPACES(__CLC_SCALAR_GENTYPE) \
    VLOAD_VECTORIZE(__CLC_SCALAR_GENTYPE, __private) \
    VLOAD_VECTORIZE(__CLC_SCALAR_GENTYPE, __local) \
    VLOAD_VECTORIZE(__CLC_SCALAR_GENTYPE, __constant) \
    VLOAD_VECTORIZE(__CLC_SCALAR_GENTYPE, __global) \

//int/uint are special... see below
#define VLOAD_TYPES() \
    VLOAD_ADDR_SPACES(char) \
    VLOAD_ADDR_SPACES(uchar) \
    VLOAD_ADDR_SPACES(short) \
    VLOAD_ADDR_SPACES(ushort) \
    VLOAD_ADDR_SPACES(long) \
    VLOAD_ADDR_SPACES(ulong) \
    VLOAD_ADDR_SPACES(float) \

VLOAD_TYPES()

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
    VLOAD_ADDR_SPACES(double)
#endif

VLOAD_VECTORIZE(int, __private)
VLOAD_VECTORIZE(int, __local)
VLOAD_VECTORIZE(int, __constant)
VLOAD_VECTORIZE(uint, __private)
VLOAD_VECTORIZE(uint, __local)
VLOAD_VECTORIZE(uint, __constant)

_CLC_OVERLOAD _CLC_DEF int2 vload2(size_t offset, const global int *x) {
  return (int2)(x[offset] , x[offset+1]);
}
_CLC_OVERLOAD _CLC_DEF int3 vload3(size_t offset, const global int *x) {
  return (int3)(vload2(offset, x), x[offset+2]);
}
_CLC_OVERLOAD _CLC_DEF uint2 vload2(size_t offset, const global uint *x) {
  return (uint2)(x[offset] , x[offset+1]);
}
_CLC_OVERLOAD _CLC_DEF uint3 vload3(size_t offset, const global uint *x) {
  return (uint3)(vload2(offset, x), x[offset+2]);
}
        
/*Note: It is known that R600 doesn't support load <2 x ?> and <3 x ?>... so
 * they aren't actually overridden here
 */
_CLC_DECL int4 __clc_vload4_int__global(size_t offset, const __global int *);
_CLC_DECL int8 __clc_vload8_int__global(size_t offset, const __global int *);
_CLC_DECL int16 __clc_vload16_int__global(size_t offset, const __global int *);

_CLC_OVERLOAD _CLC_DEF int4 vload4(size_t offset, const global int *x) {
  return __clc_vload4_int__global(offset, x);
}
_CLC_OVERLOAD _CLC_DEF int8 vload8(size_t offset, const global int *x) {
  return __clc_vload8_int__global(offset, x);
}
_CLC_OVERLOAD _CLC_DEF int16 vload16(size_t offset, const global int *x) {
  return __clc_vload16_int__global(offset, x);
}

_CLC_DECL uint4 __clc_vload4_uint__global(size_t offset, const __global uint *);
_CLC_DECL uint8 __clc_vload8_uint__global(size_t offset, const __global uint *);
_CLC_DECL uint16 __clc_vload16_uint__global(size_t offset, const __global uint *);

_CLC_OVERLOAD _CLC_DEF uint4 vload4(size_t offset, const global uint *x) {
  return __clc_vload4_uint__global(offset, x);
}
_CLC_OVERLOAD _CLC_DEF uint8 vload8(size_t offset, const global uint *x) {
  return __clc_vload8_uint__global(offset, x);
}
_CLC_OVERLOAD _CLC_DEF uint16 vload16(size_t offset, const global uint *x) {
  return __clc_vload16_uint__global(offset, x);
}
