#include <clc/clc.h>

#define VLOAD_VECTORIZE(PRIM_TYPE, ADDR_SPACE) \
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##2 vload2(size_t offset, const ADDR_SPACE PRIM_TYPE *x) { \
    return (PRIM_TYPE##2)(x[2*offset] , x[2*offset+1]); \
  } \
\
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##3 vload3(size_t offset, const ADDR_SPACE PRIM_TYPE *x) { \
    return (PRIM_TYPE##3)(x[3*offset] , x[3*offset+1], x[3*offset+2]); \
  } \
\
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##4 vload4(size_t offset, const ADDR_SPACE PRIM_TYPE *x) { \
    return (PRIM_TYPE##4)(x[4*offset], x[4*offset+1], x[4*offset+2], x[4*offset+3]); \
  } \
\
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##8 vload8(size_t offset, const ADDR_SPACE PRIM_TYPE *x) { \
    return (PRIM_TYPE##8)(vload4(0, &x[8*offset]), vload4(1, &x[8*offset])); \
  } \
\
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##16 vload16(size_t offset, const ADDR_SPACE PRIM_TYPE *x) { \
    return (PRIM_TYPE##16)(vload8(0, &x[16*offset]), vload8(1, &x[16*offset])); \
  } \

#define VLOAD_ADDR_SPACES(__CLC_SCALAR_GENTYPE) \
    VLOAD_VECTORIZE(__CLC_SCALAR_GENTYPE, __private) \
    VLOAD_VECTORIZE(__CLC_SCALAR_GENTYPE, __local) \
    VLOAD_VECTORIZE(__CLC_SCALAR_GENTYPE, __constant) \
    VLOAD_VECTORIZE(__CLC_SCALAR_GENTYPE, __global) \

#define VLOAD_TYPES() \
    VLOAD_ADDR_SPACES(char) \
    VLOAD_ADDR_SPACES(uchar) \
    VLOAD_ADDR_SPACES(short) \
    VLOAD_ADDR_SPACES(ushort) \
    VLOAD_ADDR_SPACES(int) \
    VLOAD_ADDR_SPACES(uint) \
    VLOAD_ADDR_SPACES(long) \
    VLOAD_ADDR_SPACES(ulong) \
    VLOAD_ADDR_SPACES(float) \

VLOAD_TYPES()

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
    VLOAD_ADDR_SPACES(double)
#endif
