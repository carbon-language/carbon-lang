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
    vstore2(vec.lo, offset, mem); \
    vstore2(vec.hi, offset+2, mem); \
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

#define VSTORE_ADDR_SPACES(__CLC_SCALAR___CLC_GENTYPE) \
    VSTORE_VECTORIZE(__CLC_SCALAR___CLC_GENTYPE, __private) \
    VSTORE_VECTORIZE(__CLC_SCALAR___CLC_GENTYPE, __local) \
    VSTORE_VECTORIZE(__CLC_SCALAR___CLC_GENTYPE, __global) \

//int/uint are special... see below
#define VSTORE_TYPES() \
    VSTORE_ADDR_SPACES(char) \
    VSTORE_ADDR_SPACES(uchar) \
    VSTORE_ADDR_SPACES(short) \
    VSTORE_ADDR_SPACES(ushort) \
    VSTORE_ADDR_SPACES(long) \
    VSTORE_ADDR_SPACES(ulong) \
    VSTORE_ADDR_SPACES(float) \

VSTORE_TYPES()

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
    VSTORE_ADDR_SPACES(double)
#endif

VSTORE_VECTORIZE(int, __private)
VSTORE_VECTORIZE(int, __local)
VSTORE_VECTORIZE(uint, __private)
VSTORE_VECTORIZE(uint, __local)

_CLC_OVERLOAD _CLC_DEF void vstore2(int2 vec, size_t offset, global int *mem) {
    mem[offset] = vec.s0;
    mem[offset+1] = vec.s1;
}
_CLC_OVERLOAD _CLC_DEF void vstore3(int3 vec, size_t offset, global int *mem) {
    mem[offset] = vec.s0;
    mem[offset+1] = vec.s1;
    mem[offset+2] = vec.s2;
}
_CLC_OVERLOAD _CLC_DEF void vstore2(uint2 vec, size_t offset, global uint *mem) {
    mem[offset] = vec.s0;
    mem[offset+1] = vec.s1;
}
_CLC_OVERLOAD _CLC_DEF void vstore3(uint3 vec, size_t offset, global uint *mem) {
    mem[offset] = vec.s0;
    mem[offset+1] = vec.s1;
    mem[offset+2] = vec.s2;
}

/*Note: R600 probably doesn't support store <2 x ?> and <3 x ?>... so
 * they aren't actually overridden here... lowest-common-denominator
 */
_CLC_DECL void __clc_vstore4_int__global(int4 vec, size_t offset, __global int *);
_CLC_DECL void __clc_vstore8_int__global(int8 vec, size_t offset, __global int *);
_CLC_DECL void __clc_vstore16_int__global(int16 vec, size_t offset, __global int *);

_CLC_OVERLOAD _CLC_DEF void vstore4(int4 vec, size_t offset, global int *x) {
    __clc_vstore4_int__global(vec, offset, x);
}
_CLC_OVERLOAD _CLC_DEF void vstore8(int8 vec, size_t offset, global int *x) {
    __clc_vstore8_int__global(vec, offset, x);
}
_CLC_OVERLOAD _CLC_DEF void vstore16(int16 vec, size_t offset, global int *x) {
    __clc_vstore16_int__global(vec, offset, x);
}

_CLC_DECL void __clc_vstore4_uint__global(uint4 vec, size_t offset, __global uint *);
_CLC_DECL void __clc_vstore8_uint__global(uint8 vec, size_t offset, __global uint *);
_CLC_DECL void __clc_vstore16_uint__global(uint16 vec, size_t offset, __global uint *);

_CLC_OVERLOAD _CLC_DEF void vstore4(uint4 vec, size_t offset, global uint *x) {
    __clc_vstore4_uint__global(vec, offset, x);
}
_CLC_OVERLOAD _CLC_DEF void vstore8(uint8 vec, size_t offset, global uint *x) {
    __clc_vstore8_uint__global(vec, offset, x);
}
_CLC_OVERLOAD _CLC_DEF void vstore16(uint16 vec, size_t offset, global uint *x) {
    __clc_vstore16_uint__global(vec, offset, x);
}
