// -*- C++ -*-
//===--------------------- support/win32/limits_win32.h -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_SUPPORT_WIN32_LIMITS_WIN32_H
#define _LIBCPP_SUPPORT_WIN32_LIMITS_WIN32_H

#if !defined(_MSC_VER)
#error "This header is MSVC specific, Clang and GCC should not include it"
#else

#include <float.h> // limit constants

#define __FLT_MANT_DIG__   FLT_MANT_DIG
#define __FLT_DIG__        FLT_DIG
#define __FLT_RADIX__      FLT_RADIX
#define __FLT_MIN_EXP__    FLT_MIN_EXP
#define __FLT_MIN_10_EXP__ FLT_MIN_10_EXP
#define __FLT_MAX_EXP__    FLT_MAX_EXP
#define __FLT_MAX_10_EXP__ FLT_MAX_10_EXP
#define __FLT_MIN__        FLT_MIN
#define __FLT_MAX__        FLT_MAX
#define __FLT_EPSILON__    FLT_EPSILON
// predefined by MinGW GCC
#define __FLT_DENORM_MIN__ 1.40129846432481707092e-45F

#define __DBL_MANT_DIG__   DBL_MANT_DIG
#define __DBL_DIG__        DBL_DIG
#define __DBL_RADIX__      DBL_RADIX
#define __DBL_MIN_EXP__    DBL_MIN_EXP
#define __DBL_MIN_10_EXP__ DBL_MIN_10_EXP
#define __DBL_MAX_EXP__    DBL_MAX_EXP
#define __DBL_MAX_10_EXP__ DBL_MAX_10_EXP
#define __DBL_MIN__        DBL_MIN
#define __DBL_MAX__        DBL_MAX
#define __DBL_EPSILON__    DBL_EPSILON
// predefined by MinGW GCC
#define __DBL_DENORM_MIN__ double(4.94065645841246544177e-324L)

#define __LDBL_MANT_DIG__   LDBL_MANT_DIG
#define __LDBL_DIG__        LDBL_DIG
#define __LDBL_RADIX__      LDBL_RADIX
#define __LDBL_MIN_EXP__    LDBL_MIN_EXP
#define __LDBL_MIN_10_EXP__ LDBL_MIN_10_EXP
#define __LDBL_MAX_EXP__    LDBL_MAX_EXP
#define __LDBL_MAX_10_EXP__ LDBL_MAX_10_EXP
#define __LDBL_MIN__        LDBL_MIN
#define __LDBL_MAX__        LDBL_MAX
#define __LDBL_EPSILON__    LDBL_EPSILON
// predefined by MinGW GCC
#define __LDBL_DENORM_MIN__ 3.64519953188247460253e-4951L

// __builtin replacements/workarounds
#include <ymath.h> // internal MSVC header providing the needed functionality
#define __builtin_huge_val()  HUGE_VAL
#define __builtin_huge_valf() _FInf
#define __builtin_huge_vall() _LInf
#define __builtin_nan()       _Nan
#define __builtin_nanf()      _FNan
#define __builtin_nanl()      _LNan
#define __builtin_nans()      _Snan
#define __builtin_nansf()     _FSnan
#define __builtin_nansl()     _LSnan

#endif // _MSC_VER

#endif // _LIBCPP_SUPPORT_WIN32_LIMITS_WIN32_H
