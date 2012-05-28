#include <clc/clc.h>

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define CONVERT_ID(FROM_TYPE, TO_TYPE, SUFFIX) \
  _CLC_OVERLOAD _CLC_DEF TO_TYPE convert_##TO_TYPE##SUFFIX(FROM_TYPE x) { \
    return x; \
  }

#define CONVERT_VECTORIZE(FROM_TYPE, TO_TYPE, SUFFIX) \
  _CLC_OVERLOAD _CLC_DEF TO_TYPE##2 convert_##TO_TYPE##2##SUFFIX(FROM_TYPE##2 x) { \
    return (TO_TYPE##2)(convert_##TO_TYPE##SUFFIX(x.x), convert_##TO_TYPE##SUFFIX(x.y)); \
  } \
\
  _CLC_OVERLOAD _CLC_DEF TO_TYPE##3 convert_##TO_TYPE##3##SUFFIX(FROM_TYPE##3 x) { \
    return (TO_TYPE##3)(convert_##TO_TYPE##SUFFIX(x.x), convert_##TO_TYPE##SUFFIX(x.y), convert_##TO_TYPE##SUFFIX(x.z)); \
  } \
\
  _CLC_OVERLOAD _CLC_DEF TO_TYPE##4 convert_##TO_TYPE##4##SUFFIX(FROM_TYPE##4 x) { \
    return (TO_TYPE##4)(convert_##TO_TYPE##2##SUFFIX(x.lo), convert_##TO_TYPE##2##SUFFIX(x.hi)); \
  } \
\
  _CLC_OVERLOAD _CLC_DEF TO_TYPE##8 convert_##TO_TYPE##8##SUFFIX(FROM_TYPE##8 x) { \
    return (TO_TYPE##8)(convert_##TO_TYPE##4##SUFFIX(x.lo), convert_##TO_TYPE##4##SUFFIX(x.hi)); \
  } \
\
  _CLC_OVERLOAD _CLC_DEF TO_TYPE##16 convert_##TO_TYPE##16##SUFFIX(FROM_TYPE##16 x) { \
    return (TO_TYPE##16)(convert_##TO_TYPE##8##SUFFIX(x.lo), convert_##TO_TYPE##8##SUFFIX(x.hi)); \
  }

CONVERT_ID(long, char, )
CONVERT_ID(ulong, uchar, )
CONVERT_ID(long, short, )
CONVERT_ID(ulong, ushort, )
CONVERT_ID(long, int, )
CONVERT_ID(ulong, uint, )
CONVERT_ID(long, long, )
CONVERT_ID(ulong, ulong, )
#ifdef cl_khr_fp64
CONVERT_ID(double, float, )
CONVERT_ID(double, double, )
#else
CONVERT_ID(float, float, )
#endif

_CLC_OVERLOAD _CLC_DEF char convert_char_sat(long l) {
  return l > 127 ? 127 : l < -128 ? -128 : l;
}

_CLC_OVERLOAD _CLC_DEF uchar convert_uchar_sat(ulong l) {
  return l > 255U ? 255U : l;
}

_CLC_OVERLOAD _CLC_DEF short convert_short_sat(long l) {
  return l > 32767 ? 32767 : l < -32768 ? -32768 : l;
}

_CLC_OVERLOAD _CLC_DEF ushort convert_ushort_sat(ulong l) {
  return l > 65535U ? 65535U : l;
}

_CLC_OVERLOAD _CLC_DEF int convert_int_sat(long l) {
  return l > ((1L<<31)-1) ? ((1L<<31L)-1) : l < -(1L<<31) ? -(1L<<31) : l;
}

_CLC_OVERLOAD _CLC_DEF uint convert_uint_sat(ulong l) {
  return l > ((1UL<<32)-1) ? ((1UL<<32)-1) : l;
}

CONVERT_ID(long, long, _sat)
CONVERT_ID(ulong, ulong, _sat)
#ifdef cl_khr_fp64
CONVERT_ID(double, float, _sat)
CONVERT_ID(double, double, _sat)
#else
CONVERT_ID(float, float, _sat)
#endif

#define CONVERT_VECTORIZE_FROM1(FROM_TYPE, SUFFIX) \
  CONVERT_VECTORIZE(FROM_TYPE, char, SUFFIX) \
  CONVERT_VECTORIZE(FROM_TYPE, uchar, SUFFIX) \
  CONVERT_VECTORIZE(FROM_TYPE, int, SUFFIX) \
  CONVERT_VECTORIZE(FROM_TYPE, uint, SUFFIX) \
  CONVERT_VECTORIZE(FROM_TYPE, short, SUFFIX) \
  CONVERT_VECTORIZE(FROM_TYPE, ushort, SUFFIX) \
  CONVERT_VECTORIZE(FROM_TYPE, long, SUFFIX) \
  CONVERT_VECTORIZE(FROM_TYPE, ulong, SUFFIX) \
  CONVERT_VECTORIZE(FROM_TYPE, float, SUFFIX)

#ifdef cl_khr_fp64
#define CONVERT_VECTORIZE_FROM(FROM_TYPE, SUFFIX) \
  CONVERT_VECTORIZE_FROM1(FROM_TYPE, SUFFIX) \
  CONVERT_VECTORIZE(FROM_TYPE, double, SUFFIX)
#else
#define CONVERT_VECTORIZE_FROM(FROM_TYPE, SUFFIX) \
  CONVERT_VECTORIZE_FROM1(FROM_TYPE, SUFFIX)
#endif

#define CONVERT_VECTORIZE_TO1(SUFFIX) \
  CONVERT_VECTORIZE_FROM(char, SUFFIX) \
  CONVERT_VECTORIZE_FROM(uchar, SUFFIX) \
  CONVERT_VECTORIZE_FROM(int, SUFFIX) \
  CONVERT_VECTORIZE_FROM(uint, SUFFIX) \
  CONVERT_VECTORIZE_FROM(short, SUFFIX) \
  CONVERT_VECTORIZE_FROM(ushort, SUFFIX) \
  CONVERT_VECTORIZE_FROM(long, SUFFIX) \
  CONVERT_VECTORIZE_FROM(ulong, SUFFIX) \
  CONVERT_VECTORIZE_FROM(float, SUFFIX)

#ifdef cl_khr_fp64
#define CONVERT_VECTORIZE_TO(SUFFIX) \
  CONVERT_VECTORIZE_TO1(SUFFIX) \
  CONVERT_VECTORIZE_FROM(double, SUFFIX)
#else
#define CONVERT_VECTORIZE_TO(SUFFIX) \
  CONVERT_VECTORIZE_TO1(SUFFIX)
#endif

CONVERT_VECTORIZE_TO()
CONVERT_VECTORIZE_TO(_sat)
