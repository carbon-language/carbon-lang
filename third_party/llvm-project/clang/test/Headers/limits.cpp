// RUN: %clang_cc1 -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -fno-signed-char -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++11 -ffreestanding -fsyntax-only -verify %s
// expected-no-diagnostics

#include <limits.h>

_Static_assert(SCHAR_MAX == -(SCHAR_MIN+1), "");
_Static_assert(SHRT_MAX == -(SHRT_MIN+1), "");
_Static_assert(INT_MAX == -(INT_MIN+1), "");
_Static_assert(LONG_MAX == -(LONG_MIN+1L), "");

_Static_assert(SCHAR_MAX == UCHAR_MAX/2, "");
_Static_assert(SHRT_MAX == USHRT_MAX/2, "");
_Static_assert(INT_MAX == UINT_MAX/2, "");
_Static_assert(LONG_MAX == ULONG_MAX/2, "");

_Static_assert(SCHAR_MIN == -SCHAR_MAX-1, "");
_Static_assert(SHRT_MIN == -SHRT_MAX-1, "");
_Static_assert(INT_MIN == -INT_MAX-1, "");
_Static_assert(LONG_MIN == -LONG_MAX-1L, "");

_Static_assert(UCHAR_MAX == (unsigned char)~0ULL, "");
_Static_assert(USHRT_MAX == (unsigned short)~0ULL, "");
_Static_assert(UINT_MAX == (unsigned int)~0ULL, "");
_Static_assert(ULONG_MAX == (unsigned long)~0ULL, "");

_Static_assert(MB_LEN_MAX >= 1, "");

_Static_assert(CHAR_BIT >= 8, "");

const bool char_is_signed = (char)-1 < (char)0;
_Static_assert(CHAR_MIN == (char_is_signed ? -CHAR_MAX-1 : 0), "");
_Static_assert(CHAR_MAX == (char_is_signed ? -(CHAR_MIN+1) : (char)~0ULL), "");

#if __STDC_VERSION__ >= 199901 || __cplusplus >= 201103L
_Static_assert(LLONG_MAX == -(LLONG_MIN+1LL), "");
_Static_assert(LLONG_MIN == -LLONG_MAX-1LL, "");
_Static_assert(ULLONG_MAX == (unsigned long long)~0ULL, "");
#else
int LLONG_MIN, LLONG_MAX, ULLONG_MAX; // Not defined.
#endif
