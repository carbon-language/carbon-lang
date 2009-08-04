//===-- int_lib.h - configuration header for libgcc replacement -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a configuration header for libgcc replacement.
// This file is not part of the interface of this library.
//
//===----------------------------------------------------------------------===//

#ifndef INT_LIB_H
#define INT_LIB_H

// Assumption:  signed integral is 2's complement
// Assumption:  right shift of signed negative is arithmetic shift

#include <limits.h>
#include <math.h>

#if !defined(INFINITY) && defined(HUGE_VAL)
#define INFINITY HUGE_VAL
#endif

// TODO: Improve this to minimal pre-processor hackish'ness.
#if defined (__SVR4) && defined (__sun)
// config.h build via CMake.
//#include <config.h>

// Solaris header for endian and byte swap
//#if defined HAVE_SYS_BYTEORDER_H
#include <sys/byteorder.h>

// Solaris defines endian by setting _LITTLE_ENDIAN or _BIG_ENDIAN
#ifdef _BIG_ENDIAN
# define IS_BIG_ENDIAN
#endif
#ifdef _LITTLE_ENDIAN
# define IS_LITTLE_ENDIAN
#endif

#ifdef IS_BIG_ENDIAN
#define __BIG_ENDIAN__ 1
#define __LITTLE_ENDIAN__ 0
#endif
#ifdef IS_LITTLE_ENDIAN
#define __BIG_ENDIAN__ 0
#define __LITTLE_ENDIAN__ 1
#endif

#endif //End of Solaris ifdef.

#ifdef __LITTLE_ENDIAN__
#if __LITTLE_ENDIAN__
#define _YUGA_LITTLE_ENDIAN 1
#define _YUGA_BIG_ENDIAN    0
#endif
#endif

#ifdef __BIG_ENDIAN__
#if __BIG_ENDIAN__
#define _YUGA_LITTLE_ENDIAN 0
#define _YUGA_BIG_ENDIAN    1
#endif
#endif

#if !defined(_YUGA_LITTLE_ENDIAN) || !defined(_YUGA_BIG_ENDIAN)
#error unable to determine endian
#endif

typedef      int si_int;
typedef unsigned su_int;

typedef          long long di_int;
typedef unsigned long long du_int;

typedef union
{
    di_int all;
    struct
    {
#if _YUGA_LITTLE_ENDIAN
        su_int low;
        si_int high;
#else
        si_int high;
        su_int low;
#endif
    };
} dwords;

typedef union
{
    du_int all;
    struct
    {
#if _YUGA_LITTLE_ENDIAN
        su_int low;
        su_int high;
#else
        su_int high;
        su_int low;
#endif
    };
} udwords;

#if __x86_64

typedef int      ti_int __attribute__ ((mode (TI)));
typedef unsigned tu_int __attribute__ ((mode (TI)));

typedef union
{
    ti_int all;
    struct
    {
#if _YUGA_LITTLE_ENDIAN
        du_int low;
        di_int high;
#else
        di_int high;
        du_int low;
#endif
    };
} twords;

typedef union
{
    tu_int all;
    struct
    {
#if _YUGA_LITTLE_ENDIAN
        du_int low;
        du_int high;
#else
        du_int high;
        du_int low;
#endif
    };
} utwords;

#endif

typedef union
{
    su_int u;
    float f;
} float_bits;

typedef union
{
    udwords u;
    double  f;
} double_bits;

typedef struct
{
#if _YUGA_LITTLE_ENDIAN
    udwords low;
    udwords high;
#else
    udwords high;
    udwords low;
#endif
} uqwords;

typedef union
{
    uqwords     u;
    long double f;
} long_double_bits;

#endif
