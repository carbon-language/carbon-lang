#include <clc/clc.h>

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define VSTORE_VECTORIZE(PRIM_TYPE, ADDR_SPACE) \
  _CLC_OVERLOAD _CLC_DEF void vstore2(PRIM_TYPE##2 vec, size_t offset, ADDR_SPACE PRIM_TYPE *mem) { \
    mem[2*offset] = vec.s0; \
    mem[2*offset+1] = vec.s1; \
  } \
\
  _CLC_OVERLOAD _CLC_DEF void vstore3(PRIM_TYPE##3 vec, size_t offset, ADDR_SPACE PRIM_TYPE *mem) { \
    mem[3*offset] = vec.s0; \
    mem[3*offset+1] = vec.s1; \
    mem[3*offset+2] = vec.s2; \
  } \
\
  _CLC_OVERLOAD _CLC_DEF void vstore4(PRIM_TYPE##4 vec, size_t offset, ADDR_SPACE PRIM_TYPE *mem) { \
    vstore2(vec.lo, 0, &mem[offset*4]); \
    vstore2(vec.hi, 1, &mem[offset*4]); \
  } \
\
  _CLC_OVERLOAD _CLC_DEF void vstore8(PRIM_TYPE##8 vec, size_t offset, ADDR_SPACE PRIM_TYPE *mem) { \
    vstore4(vec.lo, 0, &mem[offset*8]); \
    vstore4(vec.hi, 1, &mem[offset*8]); \
  } \
\
  _CLC_OVERLOAD _CLC_DEF void vstore16(PRIM_TYPE##16 vec, size_t offset, ADDR_SPACE PRIM_TYPE *mem) { \
    vstore8(vec.lo, 0, &mem[offset*16]); \
    vstore8(vec.hi, 1, &mem[offset*16]); \
  } \

#define VSTORE_ADDR_SPACES(SCALAR_GENTYPE) \
    VSTORE_VECTORIZE(SCALAR_GENTYPE, __private) \
    VSTORE_VECTORIZE(SCALAR_GENTYPE, __local) \
    VSTORE_VECTORIZE(SCALAR_GENTYPE, __global) \

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

_CLC_OVERLOAD _CLC_DEF void vstore3(int3 vec, size_t offset, global int *mem) {
    mem[3*offset] = vec.s0;
    mem[3*offset+1] = vec.s1;
    mem[3*offset+2] = vec.s2;
}
_CLC_OVERLOAD _CLC_DEF void vstore3(uint3 vec, size_t offset, global uint *mem) {
    mem[3*offset] = vec.s0;
    mem[3*offset+1] = vec.s1;
    mem[3*offset+2] = vec.s2;
}

/*Note: R600 doesn't support store <3 x ?>... so
 * those functions aren't actually overridden here... lowest-common-denominator
 */

//We only define functions for signed_type vstoreN(), and then just cast the pointers/vectors for unsigned types
#define _CLC_VSTORE_ASM_DECL(PRIM_TYPE,LLVM_SCALAR_TYPE,ADDR_SPACE,ADDR_SPACE_ID) \
_CLC_DECL void __clc_vstore2_##LLVM_SCALAR_TYPE##__addr##ADDR_SPACE_ID (PRIM_TYPE##2, ADDR_SPACE PRIM_TYPE *); \
_CLC_DECL void __clc_vstore4_##LLVM_SCALAR_TYPE##__addr##ADDR_SPACE_ID (PRIM_TYPE##4, ADDR_SPACE PRIM_TYPE *); \
_CLC_DECL void __clc_vstore8_##LLVM_SCALAR_TYPE##__addr##ADDR_SPACE_ID (PRIM_TYPE##8, ADDR_SPACE PRIM_TYPE *); \
_CLC_DECL void __clc_vstore16_##LLVM_SCALAR_TYPE##__addr##ADDR_SPACE_ID (PRIM_TYPE##16, ADDR_SPACE PRIM_TYPE *); \

#define _CLC_VSTORE_ASM_DEFINE(PRIM_TYPE, S_PRIM_TYPE, LLVM_SCALAR_TYPE, VEC_WIDTH, ADDR_SPACE, ADDR_SPACE_ID) \
  _CLC_OVERLOAD _CLC_DEF void vstore##VEC_WIDTH(PRIM_TYPE##VEC_WIDTH vec, size_t offset, ADDR_SPACE PRIM_TYPE *x) { \
    __clc_vstore##VEC_WIDTH##_##LLVM_SCALAR_TYPE##__addr##ADDR_SPACE_ID (__builtin_astype(vec, S_PRIM_TYPE##VEC_WIDTH), (ADDR_SPACE S_PRIM_TYPE *)&x[ VEC_WIDTH * offset]); \
  } \

/*Note: R600 back-end doesn't support load <3 x ?>... so
 * those functions aren't actually overridden here... When the back-end supports
 * that, then clean add here, and remove the vstore3 definitions from above.
 */
#define _CLC_VSTORE_ASM_OVERLOAD_SIZES(PRIM_TYPE,S_PRIM_TYPE,LLVM_TYPE,ADDR_SPACE,ADDR_SPACE_ID) \
  _CLC_VSTORE_ASM_DEFINE(PRIM_TYPE, S_PRIM_TYPE, LLVM_TYPE, 2, ADDR_SPACE, ADDR_SPACE_ID) \
  _CLC_VSTORE_ASM_DEFINE(PRIM_TYPE, S_PRIM_TYPE, LLVM_TYPE, 4, ADDR_SPACE, ADDR_SPACE_ID) \
  _CLC_VSTORE_ASM_DEFINE(PRIM_TYPE, S_PRIM_TYPE, LLVM_TYPE, 8, ADDR_SPACE, ADDR_SPACE_ID) \
  _CLC_VSTORE_ASM_DEFINE(PRIM_TYPE, S_PRIM_TYPE, LLVM_TYPE, 16, ADDR_SPACE, ADDR_SPACE_ID) \

#define _CLC_VSTORE_ASM_OVERLOAD_ADDR_SPACES(PRIM_TYPE,S_PRIM_TYPE,LLVM_TYPE) \
  _CLC_VSTORE_ASM_OVERLOAD_SIZES(PRIM_TYPE, S_PRIM_TYPE, LLVM_TYPE, global, 1) \

#define _CLC_VSTORE_ASM_OVERLOADS() \
  _CLC_VSTORE_ASM_DECL(int,i32,__global,1) \
  _CLC_VSTORE_ASM_OVERLOAD_ADDR_SPACES(int,int,i32) \
  _CLC_VSTORE_ASM_OVERLOAD_ADDR_SPACES(uint,int,i32) \

_CLC_VSTORE_ASM_OVERLOADS()