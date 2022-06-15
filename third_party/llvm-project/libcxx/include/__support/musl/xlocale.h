// -*- C++ -*-
//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This adds support for the extended locale functions that are currently
// missing from the Musl C library.
//
// This only works when the specified locale is "C" or "POSIX", but that's
// about as good as we can do without implementing full xlocale support
// in Musl.
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_SUPPORT_MUSL_XLOCALE_H
#define _LIBCPP_SUPPORT_MUSL_XLOCALE_H

#include <cstdlib>
#include <cwchar>

#ifdef __cplusplus
extern "C" {
#endif

inline _LIBCPP_HIDE_FROM_ABI long long
strtoll_l(const char *nptr, char **endptr, int base, locale_t) {
  return ::strtoll(nptr, endptr, base);
}

inline _LIBCPP_HIDE_FROM_ABI unsigned long long
strtoull_l(const char *nptr, char **endptr, int base, locale_t) {
  return ::strtoull(nptr, endptr, base);
}

inline _LIBCPP_HIDE_FROM_ABI long long
wcstoll_l(const wchar_t *nptr, wchar_t **endptr, int base, locale_t) {
  return ::wcstoll(nptr, endptr, base);
}

inline _LIBCPP_HIDE_FROM_ABI long long
wcstoull_l(const wchar_t *nptr, wchar_t **endptr, int base, locale_t) {
  return ::wcstoull(nptr, endptr, base);
}

inline _LIBCPP_HIDE_FROM_ABI long double
wcstold_l(const wchar_t *nptr, wchar_t **endptr, locale_t) {
  return ::wcstold(nptr, endptr);
}

#ifdef __cplusplus
}
#endif

#endif // _LIBCPP_SUPPORT_MUSL_XLOCALE_H
