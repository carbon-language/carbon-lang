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

#define VLOAD_ADDR_SPACES(SCALAR_GENTYPE) \
    VLOAD_VECTORIZE(SCALAR_GENTYPE, __private) \
    VLOAD_VECTORIZE(SCALAR_GENTYPE, __local) \
    VLOAD_VECTORIZE(SCALAR_GENTYPE, __constant) \
    VLOAD_VECTORIZE(SCALAR_GENTYPE, __global) \

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

//Assembly overrides start here

VLOAD_VECTORIZE(int, __private)
VLOAD_VECTORIZE(int, __local)
VLOAD_VECTORIZE(int, __constant)
VLOAD_VECTORIZE(uint, __private)
VLOAD_VECTORIZE(uint, __local)
VLOAD_VECTORIZE(uint, __constant)

_CLC_OVERLOAD _CLC_DEF int3 vload3(size_t offset, const global int *x) {
  return (int3)(vload2(0, &x[3*offset]), x[3*offset+2]);
}
_CLC_OVERLOAD _CLC_DEF uint3 vload3(size_t offset, const global uint *x) {
  return (uint3)(vload2(0, &x[3*offset]), x[3*offset+2]);
}

//We only define functions for typeN vloadN(), and then just bitcast the result for unsigned types
#define _CLC_VLOAD_ASM_DECL(PRIM_TYPE,LLVM_SCALAR_TYPE,ADDR_SPACE,ADDR_SPACE_ID) \
_CLC_DECL PRIM_TYPE##2 __clc_vload2_##LLVM_SCALAR_TYPE##__addr##ADDR_SPACE_ID (const ADDR_SPACE PRIM_TYPE *); \
_CLC_DECL PRIM_TYPE##4 __clc_vload4_##LLVM_SCALAR_TYPE##__addr##ADDR_SPACE_ID (const ADDR_SPACE PRIM_TYPE *); \
_CLC_DECL PRIM_TYPE##8 __clc_vload8_##LLVM_SCALAR_TYPE##__addr##ADDR_SPACE_ID (const ADDR_SPACE PRIM_TYPE *); \
_CLC_DECL PRIM_TYPE##16 __clc_vload16_##LLVM_SCALAR_TYPE##__addr##ADDR_SPACE_ID (const ADDR_SPACE PRIM_TYPE *); \

#define _CLC_VLOAD_ASM_DEFINE(PRIM_TYPE,S_PRIM_TYPE, LLVM_SCALAR_TYPE,VEC_WIDTH,ADDR_SPACE,ADDR_SPACE_ID) \
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##VEC_WIDTH vload##VEC_WIDTH (size_t offset, const ADDR_SPACE PRIM_TYPE *x) { \
    return __builtin_astype(__clc_vload##VEC_WIDTH##_##LLVM_SCALAR_TYPE##__addr##ADDR_SPACE_ID ((const ADDR_SPACE S_PRIM_TYPE *)&x[VEC_WIDTH * offset]), PRIM_TYPE##VEC_WIDTH); \
  } \

/*Note: R600 back-end doesn't support load <3 x ?>... so
 * those functions aren't actually overridden here
 */
#define _CLC_VLOAD_ASM_OVERLOAD_SIZES(PRIM_TYPE,S_PRIM_TYPE,LLVM_TYPE,ADDR_SPACE,ADDR_SPACE_ID) \
  _CLC_VLOAD_ASM_DEFINE(PRIM_TYPE, S_PRIM_TYPE, LLVM_TYPE, 2, ADDR_SPACE, ADDR_SPACE_ID) \
  _CLC_VLOAD_ASM_DEFINE(PRIM_TYPE, S_PRIM_TYPE, LLVM_TYPE, 4, ADDR_SPACE, ADDR_SPACE_ID) \
  _CLC_VLOAD_ASM_DEFINE(PRIM_TYPE, S_PRIM_TYPE, LLVM_TYPE, 8, ADDR_SPACE, ADDR_SPACE_ID) \
  _CLC_VLOAD_ASM_DEFINE(PRIM_TYPE, S_PRIM_TYPE, LLVM_TYPE, 16, ADDR_SPACE, ADDR_SPACE_ID) \

#define _CLC_VLOAD_ASM_OVERLOAD_ADDR_SPACES(PRIM_TYPE,S_PRIM_TYPE,LLVM_TYPE) \
  _CLC_VLOAD_ASM_OVERLOAD_SIZES(PRIM_TYPE, S_PRIM_TYPE, LLVM_TYPE, global, 1) \

#define _CLC_VLOAD_ASM_OVERLOADS() \
  _CLC_VLOAD_ASM_DECL(int,i32,__global,1) \
  _CLC_VLOAD_ASM_OVERLOAD_ADDR_SPACES(int,int,i32) \
  _CLC_VLOAD_ASM_OVERLOAD_ADDR_SPACES(uint,int,i32) \

_CLC_VLOAD_ASM_OVERLOADS()