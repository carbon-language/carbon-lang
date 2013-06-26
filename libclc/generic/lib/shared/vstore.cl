#include <clc/clc.h>

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define VSTORE_VECTORIZE(PRIM_TYPE, ADDR_SPACE) \
  _CLC_OVERLOAD _CLC_DEF void vstore2(PRIM_TYPE##2 vec, size_t offset, ADDR_SPACE PRIM_TYPE *mem) { \
    mem[offset] = vec.s0; \
    mem[offset+1] = vec.s1; \
  } \
\
  _CLC_OVERLOAD _CLC_DEF void vstore3(PRIM_TYPE##3 vec, size_t offset, ADDR_SPACE PRIM_TYPE *mem) { \
    mem[offset] = vec.s0; \
    mem[offset+1] = vec.s1; \
    mem[offset+2] = vec.s2; \
  } \
\
  _CLC_OVERLOAD _CLC_DEF void vstore4(PRIM_TYPE##4 vec, size_t offset, ADDR_SPACE PRIM_TYPE *mem) { \
    mem[offset] = vec.s0; \
    mem[offset+1] = vec.s1; \
    mem[offset+2] = vec.s2; \
    mem[offset+3] = vec.s3; \
  } \
\
  _CLC_OVERLOAD _CLC_DEF void vstore8(PRIM_TYPE##8 vec, size_t offset, ADDR_SPACE PRIM_TYPE *mem) { \
    vstore4(vec.lo, offset, mem); \
    vstore4(vec.hi, offset+4, mem); \
  } \
\
  _CLC_OVERLOAD _CLC_DEF void vstore16(PRIM_TYPE##16 vec, size_t offset, ADDR_SPACE PRIM_TYPE *mem) { \
    vstore8(vec.lo, offset, mem); \
    vstore8(vec.hi, offset+8, mem); \
  } \

#define VSTORE_ADDR_SPACES(SCALAR_GENTYPE) \
    VSTORE_VECTORIZE(SCALAR_GENTYPE, __private) \
    VSTORE_VECTORIZE(SCALAR_GENTYPE, __local) \
    VSTORE_VECTORIZE(SCALAR_GENTYPE, __global) \

#define VSTORE_TYPES() \
    VSTORE_ADDR_SPACES(char) \
    VSTORE_ADDR_SPACES(uchar) \
    VSTORE_ADDR_SPACES(short) \
    VSTORE_ADDR_SPACES(ushort) \
    VSTORE_ADDR_SPACES(int) \
    VSTORE_ADDR_SPACES(uint) \
    VSTORE_ADDR_SPACES(long) \
    VSTORE_ADDR_SPACES(ulong) \
    VSTORE_ADDR_SPACES(float) \

VSTORE_TYPES()

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
    VSTORE_ADDR_SPACES(double)
#endif

