/*===---- stdint.h - Standard header for sized integer types --------------===*\
 *
 * Copyright (c) 2009 Chris Lattner
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
\*===----------------------------------------------------------------------===*/

#ifndef __STDINT_H
#define __STDINT_H

/* We currently only support targets with power of two, 2s complement integers.
 */

/* C99 7.18.1.1 Exact-width integer types.
 * C99 7.18.1.2 Minimum-width integer types.
 * C99 7.18.1.3 Fastest minimum-width integer types.
 * Since we only support pow-2 targets, these map directly to exact width types.
 */

typedef signed __INT8_TYPE__ int8_t;
typedef unsigned __INT8_TYPE__ uint8_t;
typedef int8_t     int_least8_t;
typedef uint8_t   uint_least8_t;
typedef int8_t     int_fast8_t;
typedef uint8_t   uint_fast8_t;

typedef __INT16_TYPE__ int16_t;
typedef unsigned __INT16_TYPE__ uint16_t;
typedef int16_t   int_least16_t;
typedef uint16_t uint_least16_t;
typedef int16_t   int_fast16_t;
typedef uint16_t uint_fast16_t;

typedef __INT32_TYPE__ int32_t;
typedef unsigned __INT32_TYPE__ uint32_t;
typedef int32_t   int_least32_t;
typedef uint32_t uint_least32_t;
typedef int32_t   int_fast32_t;
typedef uint32_t uint_fast32_t;

/* Some 16-bit targets do not have a 64-bit datatype.  Only define the 64-bit
 * typedefs if there is something to typedef them to.
 */
#ifdef __INT64_TYPE__
typedef __INT64_TYPE__ int64_t;
typedef unsigned __INT64_TYPE__ uint64_t;
typedef int64_t   int_least64_t;
typedef uint64_t uint_least64_t;
typedef int64_t   int_fast64_t;
typedef uint64_t uint_fast64_t;
#endif


/* C99 7.18.1.4 Integer types capable of holding object pointers.
 */
#if __POINTER_WIDTH__ == 64
typedef int64_t  intptr_t;
typedef uint64_t uintptr_t;
#elif __POINTER_WIDTH__ == 32
typedef int32_t  intptr_t;
typedef uint32_t uintptr_t;
#elif __POINTER_WIDTH__ == 16
typedef int16_t  intptr_t;
typedef uint16_t uintptr_t;
#else
#error "unknown or unset pointer width!"
#endif

/* C99 7.18.1.5 Greatest-width integer types.
 */
typedef __INTMAX_TYPE__   intmax_t;
typedef __UINTMAX_TYPE__ uintmax_t;

/* C99 7.18.2.1 Limits of exact-width integer types. 
 * Fixed sized values have fixed size max/min.
 * C99 7.18.2.2 Limits of minimum-width integer types.
 * Since we map these directly onto fixed-sized types, these values the same.
 * C99 7.18.2.3 Limits of fastest minimum-width integer types.
 */

#if !defined(__cplusplus) || defined(__STDC_LIMIT_MACROS)

#define INT8_MAX    ((int8_t)127)
#define INT8_MIN   ((int8_t)-128)
#define UINT8_MAX  ((uint8_t)255)
#define INT_LEAST8_MIN   INT8_MIN
#define INT_LEAST8_MAX   INT8_MAX
#define UINT_LEAST8_MAX UINT8_MAX
#define INT_FAST8_MIN    INT8_MIN
#define INT_FAST8_MAX    INT8_MAX
#define UINT_FAST8_MAX  UINT8_MAX

#define INT16_MAX    ((int16_t)32767)
#define INT16_MIN   ((int16_t)-32768)
#define UINT16_MAX ((uint16_t)(65535))
#define INT_LEAST16_MIN   INT16_MIN
#define INT_LEAST16_MAX   INT16_MAX
#define UINT_LEAST16_MAX UINT16_MAX
#define INT_FAST16_MIN    INT16_MIN
#define INT_FAST16_MAX    INT16_MAX
#define UINT_FAST16_MAX  UINT16_MAX

#define INT32_MAX         ((int32_t)2147483647LL)
#define INT32_MIN    ((int32_t)(-2147483647LL-1))
#define UINT32_MAX      ((uint32_t)4294967295ULL)
#define INT_LEAST32_MIN  INT32_MIN
#define INT_LEAST32_MAX  INT32_MAX
#define UINT_LEAST32_MAX UINT32_MAX
#define INT_FAST32_MIN   INT32_MIN
#define INT_FAST32_MAX   INT32_MAX
#define UINT_FAST32_MAX  UINT32_MAX

/* If we do not have 64-bit support, don't define the 64-bit size macros. */
#ifndef __INT64_TYPE__
#define INT64_MAX      ((int64_t)9223372036854775807LL)
#define INT64_MIN ((int64_t)(-9223372036854775807LL-1))
#define UINT64_MAX  ((uint64_t)18446744073709551615ULL)
#define INT_LEAST64_MIN  INT64_MIN
#define INT_LEAST64_MAX  INT64_MAX
#define UINT_LEAST64_MAX UINT64_MAX
#define INT_FAST64_MIN    INT64_MIN
#define INT_FAST64_MAX    INT64_MAX
#define UINT_FAST64_MAX  UINT64_MAX
#endif

/* C99 7.18.2.4 Limits of integer types capable of holding object pointers. */
/* C99 7.18.3 Limits of other integer types. */

#if __POINTER_WIDTH__ == 64

#define  INTPTR_MIN  INT64_MIN
#define  INTPTR_MAX  INT64_MAX
#define UINTPTR_MAX UINT64_MAX
#define PTRDIFF_MIN  INT64_MIN
#define PTRDIFF_MAX  INT64_MAX
#define SIZE_MAX    UINT64_MAX

#elif __POINTER_WIDTH__ == 32

#define  INTPTR_MIN  INT32_MIN
#define  INTPTR_MAX  INT32_MAX
#define UINTPTR_MAX UINT32_MAX
#define PTRDIFF_MIN  INT32_MIN
#define PTRDIFF_MAX  INT32_MAX
#define SIZE_MAX    UINT32_MAX

#elif __POINTER_WIDTH__ == 16

#define  INTPTR_MIN  INT16_MIN
#define  INTPTR_MAX  INT16_MAX
#define UINTPTR_MAX UINT16_MAX
#define PTRDIFF_MIN  INT16_MIN
#define PTRDIFF_MAX  INT16_MAX
#define SIZE_MAX    UINT16_MAX

#else
#error "unknown or unset pointer width!"
#endif

/* C99 7.18.2.5 Limits of greatest-width integer types. */
#define INTMAX_MIN  (-__INTMAX_MAX__-1)
#define INTMAX_MAX   __INTMAX_MAX__
#define UINTMAX_MAX (__INTMAX_MAX__*2+1)

/* C99 7.18.3 Limits of other integer types. */
#define SIG_ATOMIC_MIN INT32_MIN
#define SIG_ATOMIC_MAX INT32_MAX
#define WINT_MIN       INT32_MIN
#define WINT_MAX       INT32_MAX

/* FIXME: if we ever support a target with unsigned wchar_t, this should be
 * 0 .. Max.
 */
#ifndef WCHAR_MAX
#define WCHAR_MAX __WCHAR_MAX__
#endif
#ifndef WCHAR_MIN
#define WCHAR_MIN (-__WCHAR_MAX__-1)
#endif

#endif /* C++ needs __STDC_LIMIT_MACROS. */

/* C99 7.18.4 Macros for minimum-width integer constants. */

/* C99 Footnote 220: C++ requires __STDC_CONSTANT_MACROS. */
#if !defined(__cplusplus) || defined(__STDC_CONSTANT_MACROS)

#define INT8_C(v)   (v)
#define UINT8_C(v)  (v##U)
#define INT16_C(v)  (v)
#define UINT16_C(v) (v##U)
#define INT32_C(v)  (v)
#define UINT32_C(v) (v##U)

/* If we do not have 64-bit support, don't define the 64-bit size macros. */
#ifndef __INT64_TYPE__
#define INT64_C(v)  (v##LL)
#define UINT64_C(v) (v##ULL)
#endif

/* 7.18.4.2 Macros for greatest-width integer constants. */
#define INTMAX_C(v)  (v ## LL)
#define UINTMAX_C(v) (v ## ULL)

#endif  /* C++ requires __STDC_CONSTANT_MACROS */

#endif /* __STDINT_H */
