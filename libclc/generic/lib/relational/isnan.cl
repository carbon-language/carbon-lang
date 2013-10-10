#include <clc/clc.h>

_CLC_DEFINE_UNARY_BUILTIN(int, isnan, __builtin_isnan, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of isnan(double) returns an int, but the vector versions
// return long.
_CLC_DEF _CLC_OVERLOAD int isnan(double x) {
  return __builtin_isnan(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, isnan, double)

#endif
