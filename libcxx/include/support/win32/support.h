// -*- C++ -*-
//===----------------------- support/win32/support.h ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_SUPPORT_WIN32_SUPPORT_H
#define _LIBCPP_SUPPORT_WIN32_SUPPORT_H

/*
   Functions and constants used in libc++ that are missing from the Windows C library.
  */

#include <__config>
#include <wchar.h>  // mbstate_t
#include <stdio.h> // _snwprintf
#define swprintf _snwprintf
#define vswprintf _vsnwprintf
#define vfscnaf fscanf

int vasprintf( char **sptr, const char *__restrict fmt , va_list ap );
int asprintf( char **sptr, const char *__restrict fmt, ...);
//int vfscanf( FILE *__restrict stream, const char *__restrict format,
//             va_list arg);

size_t mbsnrtowcs( wchar_t *__restrict dst, const char **__restrict src,
                   size_t nmc, size_t len, mbstate_t *__restrict ps );
size_t wcsnrtombs( char *__restrict dst, const wchar_t **__restrict src,
                   size_t nwc, size_t len, mbstate_t *__restrict ps );
				   
#if defined(_MSC_VER)
#define snprintf _snprintf
inline int isblank( int c, locale_t /*loc*/ )
{ return ( c == ' ' || c == '\t' ); }
inline int iswblank( wint_t c, locale_t /*loc*/ )
{ return ( c == L' ' || c == L'\t' ); }
#include <xlocinfo.h>
#define atoll _atoi64
#define strtoll _strtoi64
#define strtoull _strtoui64
#define wcstoll _wcstoi64
#define wcstoull _wcstoui64
_LIBCPP_ALWAYS_INLINE float strtof( const char *nptr, char **endptr )
{ return _Stof(nptr, endptr, 0); }
_LIBCPP_ALWAYS_INLINE double strtod( const char *nptr, char **endptr )
{ return _Stod(nptr, endptr, 0); }
_LIBCPP_ALWAYS_INLINE long double strtold( const char *nptr, char **endptr )
{ return _Stold(nptr, endptr, 0); }
_LIBCPP_ALWAYS_INLINE float wcstof( const wchar_t *nptr, char** endptr )

#define _Exit _exit

#include <intrin.h>
#define __builtin_popcount __popcnt
#define __builtin_popcountl __popcnt
#define __builtin_popcountll(__i) static_cast<int>(__popcnt64(__i))

_LIBCPP_ALWAYS_INLINE int __builtin_ctz( unsigned int x )
{
   DWORD r = 0;
   _BitScanReverse(&r, x);
   return static_cast<int>(r);
}
// sizeof(long) == sizeof(int) on Windows
_LIBCPP_ALWAYS_INLINE int __builtin_ctzl( unsigned long x )
{ return __builtin_ctz( static_cast<int>(x) ); }
_LIBCPP_ALWAYS_INLINE int __builtin_ctzll( unsigned long long x )
{
    DWORD r = 0;
	_BitScanReverse64(&r, x);
	return static_cast<int>(r);
}
_LIBCPP_ALWAYS_INLINE int __builtin_clz( unsigned int x )
{
   DWORD r = 0;
   _BitScanForward(&r, x);
   return static_cast<int>(r);
}
// sizeof(long) == sizeof(int) on Windows
_LIBCPP_ALWAYS_INLINE int __builtin_clzl( unsigned long x )
{ return __builtin_clz( static_cast<int>(x) ); }
_LIBCPP_ALWAYS_INLINE int __builtin_clzll( unsigned long long x )
{
    DWORD r = 0;
	_BitScanForward64(&r, x);
	return static_cast<int>(r);
}

#endif

#endif // _LIBCPP_SUPPORT_WIN32_SUPPORT_H