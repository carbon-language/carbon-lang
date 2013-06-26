#ifndef cl_clang_storage_class_specifiers
#error Implementation requires cl_clang_storage_class_specifiers extension!
#endif

#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

/* Function Attributes */
#include <clc/clcfunc.h>

/* Pattern Macro Definitions */
#include <clc/clcmacro.h>

/* 6.1 Supported Data Types */
#include <clc/clctypes.h>

/* 6.2.3 Explicit Conversions */
#include <clc/convert.h>

/* 6.2.4.2 Reinterpreting Types Using as_type() and as_typen() */
#include <clc/as_type.h>

/* 6.11.1 Work-Item Functions */
#include <clc/workitem/get_global_size.h>
#include <clc/workitem/get_global_id.h>
#include <clc/workitem/get_local_size.h>
#include <clc/workitem/get_local_id.h>
#include <clc/workitem/get_num_groups.h>
#include <clc/workitem/get_group_id.h>

/* 6.11.2 Math Functions */
#include <clc/math/cos.h>
#include <clc/math/ceil.h>
#include <clc/math/exp.h>
#include <clc/math/exp2.h>
#include <clc/math/fabs.h>
#include <clc/math/floor.h>
#include <clc/math/fma.h>
#include <clc/math/fmax.h>
#include <clc/math/fmin.h>
#include <clc/math/hypot.h>
#include <clc/math/log.h>
#include <clc/math/log2.h>
#include <clc/math/mad.h>
#include <clc/math/pow.h>
#include <clc/math/sin.h>
#include <clc/math/sqrt.h>
#include <clc/math/native_cos.h>
#include <clc/math/native_divide.h>
#include <clc/math/native_exp.h>
#include <clc/math/native_exp2.h>
#include <clc/math/native_log.h>
#include <clc/math/native_log2.h>
#include <clc/math/native_powr.h>
#include <clc/math/native_sin.h>
#include <clc/math/native_sqrt.h>
#include <clc/math/rsqrt.h>

/* 6.11.3 Integer Functions */
#include <clc/integer/abs.h>
#include <clc/integer/abs_diff.h>
#include <clc/integer/add_sat.h>
#include <clc/integer/clz.h>
#include <clc/integer/rotate.h>
#include <clc/integer/sub_sat.h>

/* 6.11.2 and 6.11.3 Shared Integer/Math Functions */
#include <clc/shared/clamp.h>
#include <clc/shared/max.h>
#include <clc/shared/min.h>

/* 6.11.5 Geometric Functions */
#include <clc/geometric/cross.h>
#include <clc/geometric/dot.h>
#include <clc/geometric/length.h>
#include <clc/geometric/normalize.h>

/* 6.11.6 Relational Functions */
#include <clc/relational/any.h>
#include <clc/relational/select.h>

/* 6.11.8 Synchronization Functions */
#include <clc/synchronization/cl_mem_fence_flags.h>
#include <clc/synchronization/barrier.h>

#pragma OPENCL EXTENSION all : disable
