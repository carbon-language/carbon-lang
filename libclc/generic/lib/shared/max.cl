#include <clc/clc.h>

#define BODY <max.inc>
#include <clc/integer/gentype.inc>

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define BODY <max.inc>
#include <clc/math/gentype.inc>
