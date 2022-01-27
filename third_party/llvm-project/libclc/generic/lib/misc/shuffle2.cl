//===-- generic/lib/misc/shuffle2.cl ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>

#define _CLC_ELEMENT_CASES2(VAR) \
    case 0: return VAR.s0; \
    case 1: return VAR.s1;

#define _CLC_ELEMENT_CASES4(VAR) \
    _CLC_ELEMENT_CASES2(VAR) \
    case 2: return VAR.s2; \
    case 3: return VAR.s3;

#define _CLC_ELEMENT_CASES8(VAR) \
    _CLC_ELEMENT_CASES4(VAR) \
    case 4: return VAR.s4; \
    case 5: return VAR.s5; \
    case 6: return VAR.s6; \
    case 7: return VAR.s7;

#define _CLC_ELEMENT_CASES16(VAR) \
    _CLC_ELEMENT_CASES8(VAR) \
    case 8: return VAR.s8; \
    case 9: return VAR.s9; \
    case 10: return VAR.sA; \
    case 11: return VAR.sB; \
    case 12: return VAR.sC; \
    case 13: return VAR.sD; \
    case 14: return VAR.sE; \
    case 15: return VAR.sF;

#define _CLC_GET_ELEMENT_DEFINE(ARGTYPE, ARGSIZE, IDXTYPE) \
     __attribute__((always_inline)) \
     ARGTYPE __clc_get_el_##ARGTYPE##ARGSIZE##_##IDXTYPE(ARGTYPE##ARGSIZE x, ARGTYPE##ARGSIZE y, IDXTYPE idx) {\
        if (idx < ARGSIZE) \
            switch (idx){ \
                _CLC_ELEMENT_CASES##ARGSIZE(x) \
                default: return 0; \
            } \
        else \
            switch (idx - ARGSIZE){ \
                _CLC_ELEMENT_CASES##ARGSIZE(y) \
                default: return 0; \
            } \
    } \

#define _CLC_SHUFFLE_SET_2_ELEMENTS(ARGTYPE, ARGSIZE, MASKTYPE) \
    ret_val.s0 = __clc_get_el_##ARGTYPE##ARGSIZE##_##MASKTYPE(x, y, mask.s0); \
    ret_val.s1 = __clc_get_el_##ARGTYPE##ARGSIZE##_##MASKTYPE(x, y, mask.s1);

#define _CLC_SHUFFLE_SET_4_ELEMENTS(ARGTYPE, ARGSIZE, MASKTYPE) \
    _CLC_SHUFFLE_SET_2_ELEMENTS(ARGTYPE, ARGSIZE, MASKTYPE) \
    ret_val.s2 = __clc_get_el_##ARGTYPE##ARGSIZE##_##MASKTYPE(x, y, mask.s2); \
    ret_val.s3 = __clc_get_el_##ARGTYPE##ARGSIZE##_##MASKTYPE(x, y, mask.s3);

#define _CLC_SHUFFLE_SET_8_ELEMENTS(ARGTYPE, ARGSIZE, MASKTYPE) \
    _CLC_SHUFFLE_SET_4_ELEMENTS(ARGTYPE, ARGSIZE, MASKTYPE) \
    ret_val.s4 = __clc_get_el_##ARGTYPE##ARGSIZE##_##MASKTYPE(x, y, mask.s4); \
    ret_val.s5 = __clc_get_el_##ARGTYPE##ARGSIZE##_##MASKTYPE(x, y, mask.s5); \
    ret_val.s6 = __clc_get_el_##ARGTYPE##ARGSIZE##_##MASKTYPE(x, y, mask.s6); \
    ret_val.s7 = __clc_get_el_##ARGTYPE##ARGSIZE##_##MASKTYPE(x, y, mask.s7);

#define _CLC_SHUFFLE_SET_16_ELEMENTS(ARGTYPE, ARGSIZE, MASKTYPE) \
    _CLC_SHUFFLE_SET_8_ELEMENTS(ARGTYPE, ARGSIZE, MASKTYPE) \
    ret_val.s8 = __clc_get_el_##ARGTYPE##ARGSIZE##_##MASKTYPE(x, y, mask.s8); \
    ret_val.s9 = __clc_get_el_##ARGTYPE##ARGSIZE##_##MASKTYPE(x, y, mask.s9); \
    ret_val.sA = __clc_get_el_##ARGTYPE##ARGSIZE##_##MASKTYPE(x, y, mask.sA); \
    ret_val.sB = __clc_get_el_##ARGTYPE##ARGSIZE##_##MASKTYPE(x, y, mask.sB); \
    ret_val.sC = __clc_get_el_##ARGTYPE##ARGSIZE##_##MASKTYPE(x, y, mask.sC); \
    ret_val.sD = __clc_get_el_##ARGTYPE##ARGSIZE##_##MASKTYPE(x, y, mask.sD); \
    ret_val.sE = __clc_get_el_##ARGTYPE##ARGSIZE##_##MASKTYPE(x, y, mask.sE); \
    ret_val.sF = __clc_get_el_##ARGTYPE##ARGSIZE##_##MASKTYPE(x, y, mask.sF); \

#define _CLC_SHUFFLE_DEFINE2(ARGTYPE, ARGSIZE, MASKTYPE) \
_CLC_DEF _CLC_OVERLOAD ARGTYPE##2 shuffle2(ARGTYPE##ARGSIZE x, ARGTYPE##ARGSIZE y, MASKTYPE##2 mask){ \
    ARGTYPE##2 ret_val; \
    mask &= (MASKTYPE##2)(ARGSIZE * 2 - 1); \
    _CLC_SHUFFLE_SET_2_ELEMENTS(ARGTYPE, ARGSIZE, MASKTYPE) \
    return ret_val; \
}

#define _CLC_SHUFFLE_DEFINE4(ARGTYPE, ARGSIZE, MASKTYPE) \
_CLC_DEF _CLC_OVERLOAD ARGTYPE##4 shuffle2(ARGTYPE##ARGSIZE x, ARGTYPE##ARGSIZE y, MASKTYPE##4 mask){ \
    ARGTYPE##4 ret_val; \
    mask &= (MASKTYPE##4)(ARGSIZE * 2 - 1); \
    _CLC_SHUFFLE_SET_4_ELEMENTS(ARGTYPE, ARGSIZE, MASKTYPE) \
    return ret_val; \
}

#define _CLC_SHUFFLE_DEFINE8(ARGTYPE, ARGSIZE, MASKTYPE) \
_CLC_DEF _CLC_OVERLOAD ARGTYPE##8 shuffle2(ARGTYPE##ARGSIZE x, ARGTYPE##ARGSIZE y, MASKTYPE##8 mask){ \
    ARGTYPE##8 ret_val; \
    mask &= (MASKTYPE##8)(ARGSIZE * 2 - 1); \
    _CLC_SHUFFLE_SET_8_ELEMENTS(ARGTYPE, ARGSIZE, MASKTYPE) \
    return ret_val; \
}

#define _CLC_SHUFFLE_DEFINE16(ARGTYPE, ARGSIZE, MASKTYPE) \
_CLC_DEF _CLC_OVERLOAD ARGTYPE##16 shuffle2(ARGTYPE##ARGSIZE x, ARGTYPE##ARGSIZE y, MASKTYPE##16 mask){ \
    ARGTYPE##16 ret_val; \
    mask &= (MASKTYPE##16)(ARGSIZE * 2 - 1); \
    _CLC_SHUFFLE_SET_16_ELEMENTS(ARGTYPE, ARGSIZE, MASKTYPE) \
    return ret_val; \
}

#define _CLC_VECTOR_SHUFFLE_MASKSIZE(INTYPE, ARGSIZE, MASKTYPE) \
  _CLC_GET_ELEMENT_DEFINE(INTYPE, ARGSIZE, MASKTYPE) \
  _CLC_SHUFFLE_DEFINE2(INTYPE, ARGSIZE, MASKTYPE) \
  _CLC_SHUFFLE_DEFINE4(INTYPE, ARGSIZE, MASKTYPE) \
  _CLC_SHUFFLE_DEFINE8(INTYPE, ARGSIZE, MASKTYPE) \
  _CLC_SHUFFLE_DEFINE16(INTYPE, ARGSIZE, MASKTYPE) \

#define _CLC_VECTOR_SHUFFLE_INSIZE(TYPE, MASKTYPE) \
  _CLC_VECTOR_SHUFFLE_MASKSIZE(TYPE, 2, MASKTYPE) \
  _CLC_VECTOR_SHUFFLE_MASKSIZE(TYPE, 4, MASKTYPE) \
  _CLC_VECTOR_SHUFFLE_MASKSIZE(TYPE, 8, MASKTYPE) \
  _CLC_VECTOR_SHUFFLE_MASKSIZE(TYPE, 16, MASKTYPE) \



_CLC_VECTOR_SHUFFLE_INSIZE(char, uchar)
_CLC_VECTOR_SHUFFLE_INSIZE(short, ushort)
_CLC_VECTOR_SHUFFLE_INSIZE(int, uint)
_CLC_VECTOR_SHUFFLE_INSIZE(long, ulong)
_CLC_VECTOR_SHUFFLE_INSIZE(uchar, uchar)
_CLC_VECTOR_SHUFFLE_INSIZE(ushort, ushort)
_CLC_VECTOR_SHUFFLE_INSIZE(uint, uint)
_CLC_VECTOR_SHUFFLE_INSIZE(ulong, ulong)
_CLC_VECTOR_SHUFFLE_INSIZE(float, uint)
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
_CLC_VECTOR_SHUFFLE_INSIZE(double, ulong)
#endif
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
_CLC_VECTOR_SHUFFLE_INSIZE(half, ushort)
#endif

#undef _CLC_ELEMENT_CASES2
#undef _CLC_ELEMENT_CASES4
#undef _CLC_ELEMENT_CASES8
#undef _CLC_ELEMENT_CASES16
#undef _CLC_GET_ELEMENT_DEFINE
#undef _CLC_SHUFFLE_SET_2_ELEMENTS
#undef _CLC_SHUFFLE_SET_4_ELEMENTS
#undef _CLC_SHUFFLE_SET_8_ELEMENTS
#undef _CLC_SHUFFLE_SET_16_ELEMENTS
#undef _CLC_SHUFFLE_DEFINE2
#undef _CLC_SHUFFLE_DEFINE4
#undef _CLC_SHUFFLE_DEFINE8
#undef _CLC_SHUFFLE_DEFINE16
#undef _CLC_VECTOR_SHUFFLE_MASKSIZE
#undef _CLC_VECTOR_SHUFFLE_INSIZE
