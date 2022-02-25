// -*- C++ -*-
//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_SUPPORT_IBM_XLOCALE_H
#define _LIBCPP_SUPPORT_IBM_XLOCALE_H

#include <__support/ibm/locale_mgmt_aix.h>
#include <__support/ibm/locale_mgmt_zos.h>
#include <stdarg.h>

#include "cstdlib"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_AIX)
#if !defined(_AIX71)
// AIX 7.1 and higher has these definitions.  Definitions and stubs
// are provied here as a temporary workaround on AIX 6.1.
static inline
int isalnum_l(int c, locale_t locale)
{
  return __xisalnum(locale, c);
}
static inline
int isalpha_l(int c, locale_t locale)
{
  return __xisalpha(locale, c);
}
static inline
int isblank_l(int c, locale_t locale)
{
  return __xisblank(locale, c);
}
static inline
int iscntrl_l(int c, locale_t locale)
{
  return __xiscntrl(locale, c);
}
static inline
int isdigit_l(int c, locale_t locale)
{
  return __xisdigit(locale, c);
}
static inline
int isgraph_l(int c, locale_t locale)
{
  return __xisgraph(locale, c);
}
static inline
int islower_l(int c, locale_t locale)
{
  return __xislower(locale, c);
}
static inline
int isprint_l(int c, locale_t locale)
{
  return __xisprint(locale, c);
}

static inline
int ispunct_l(int c, locale_t locale)
{
  return __xispunct(locale, c);
}
static inline
int isspace_l(int c, locale_t locale)
{
  return __xisspace(locale, c);
}
static inline
int isupper_l(int c, locale_t locale)
{
  return __xisupper(locale, c);
}

static inline
int isxdigit_l(int c, locale_t locale)
{
  return __xisxdigit(locale, c);
}

static inline
int iswalnum_l(wchar_t wc, locale_t locale)
{
  return __xiswalnum(locale, wc);
}

static inline
int iswalpha_l(wchar_t wc, locale_t locale)
{
  return __xiswalpha(locale, wc);
}

static inline
int iswblank_l(wchar_t wc, locale_t locale)
{
  return __xiswblank(locale, wc);
}

static inline
int iswcntrl_l(wchar_t wc, locale_t locale)
{
  return __xiswcntrl(locale, wc);
}

static inline
int iswdigit_l(wchar_t wc, locale_t locale)
{
  return __xiswdigit(locale, wc);
}

static inline
int iswgraph_l(wchar_t wc, locale_t locale)
{
  return __xiswgraph(locale, wc);
}

static inline
int iswlower_l(wchar_t wc, locale_t locale)
{
  return __xiswlower(locale, wc);
}

static inline
int iswprint_l(wchar_t wc, locale_t locale)
{
  return __xiswprint(locale, wc);
}

static inline
int iswpunct_l(wchar_t wc, locale_t locale)
{
  return __xiswpunct(locale, wc);
}

static inline
int iswspace_l(wchar_t wc, locale_t locale)
{
  return __xiswspace(locale, wc);
}

static inline
int iswupper_l(wchar_t wc, locale_t locale)
{
  return __xiswupper(locale, wc);
}

static inline
int iswxdigit_l(wchar_t wc, locale_t locale)
{
  return __xiswxdigit(locale, wc);
}

static inline
int iswctype_l(wint_t wc, wctype_t desc, locale_t locale)
{
  return __xiswctype(locale, wc, desc);
}

static inline
int toupper_l(int c, locale_t locale)
{
  return __xtoupper(locale, c);
}
static inline
int tolower_l(int c, locale_t locale)
{
  return __xtolower(locale, c);
}
static inline
wint_t towupper_l(wint_t wc, locale_t locale)
{
  return __xtowupper(locale, wc);
}
static inline
wint_t towlower_l(wint_t wc, locale_t locale)
{
  return __xtowlower(locale, wc);
}

static inline
int strcoll_l(const char *__s1, const char *__s2, locale_t locale)
{
  return __xstrcoll(locale, __s1, __s2);
}
static inline
int wcscoll_l(const wchar_t *__s1, const wchar_t *__s2, locale_t locale)
{
  return __xwcscoll(locale, __s1, __s2);
}
static inline
size_t strxfrm_l(char *__s1, const char *__s2, size_t __n, locale_t locale)
{
  return __xstrxfrm(locale, __s1, __s2, __n);
}

static inline
size_t wcsxfrm_l(wchar_t *__ws1, const wchar_t *__ws2, size_t __n,
    locale_t locale)
{
  return __xwcsxfrm(locale, __ws1, __ws2, __n);
}
#endif // !defined(_AIX71)

// strftime_l() is defined by POSIX. However, AIX 7.1 and z/OS do not have it
// implemented yet. z/OS retrieves it from the POSIX fallbacks.
#if !defined(_AIX72)
static inline
size_t strftime_l(char *__s, size_t __size, const char *__fmt,
                  const struct tm *__tm, locale_t locale) {
  return __xstrftime(locale, __s, __size, __fmt, __tm);
}
#endif

#elif defined(__MVS__)
#include <wctype.h>
// POSIX routines
#include <__support/xlocale/__posix_l_fallback.h>
#endif // defined(__MVS__)

namespace {

struct __setAndRestore {
  explicit __setAndRestore(locale_t locale) {
    if (locale == (locale_t)0) {
      __cloc = newlocale(LC_ALL_MASK, "C", /* base */ (locale_t)0);
      __stored = uselocale(__cloc);
    } else {
      __stored = uselocale(locale);
    }
  }

  ~__setAndRestore() {
    uselocale(__stored);
    if (__cloc)
      freelocale(__cloc);
  }

private:
  locale_t __stored = (locale_t)0;
  locale_t __cloc = (locale_t)0;
};

} // namespace

// The following are not POSIX routines.  These are quick-and-dirty hacks
// to make things pretend to work
static inline
long long strtoll_l(const char *__nptr, char **__endptr,
    int __base, locale_t locale) {
  __setAndRestore __newloc(locale);
  return strtoll(__nptr, __endptr, __base);
}

static inline
long strtol_l(const char *__nptr, char **__endptr,
    int __base, locale_t locale) {
  __setAndRestore __newloc(locale);
  return strtol(__nptr, __endptr, __base);
}

static inline
double strtod_l(const char *__nptr, char **__endptr,
    locale_t locale) {
  __setAndRestore __newloc(locale);
  return strtod(__nptr, __endptr);
}

static inline
float strtof_l(const char *__nptr, char **__endptr,
    locale_t locale) {
  __setAndRestore __newloc(locale);
  return strtof(__nptr, __endptr);
}

static inline
long double strtold_l(const char *__nptr, char **__endptr,
    locale_t locale) {
  __setAndRestore __newloc(locale);
  return strtold(__nptr, __endptr);
}

static inline
unsigned long long strtoull_l(const char *__nptr, char **__endptr,
    int __base, locale_t locale) {
  __setAndRestore __newloc(locale);
  return strtoull(__nptr, __endptr, __base);
}

static inline
unsigned long strtoul_l(const char *__nptr, char **__endptr,
    int __base, locale_t locale) {
  __setAndRestore __newloc(locale);
  return strtoul(__nptr, __endptr, __base);
}

static inline
int vasprintf(char **strp, const char *fmt, va_list ap) {
  const size_t buff_size = 256;
  if ((*strp = (char *)malloc(buff_size)) == NULL) {
    return -1;
  }

  va_list ap_copy;
  // va_copy may not be provided by the C library in C++ 03 mode.
#if defined(_LIBCPP_CXX03_LANG) && __has_builtin(__builtin_va_copy)
  __builtin_va_copy(ap_copy, ap);
#else
  va_copy(ap_copy, ap);
#endif
  int str_size = vsnprintf(*strp, buff_size, fmt,  ap_copy);
  va_end(ap_copy);

  if ((size_t) str_size >= buff_size) {
    if ((*strp = (char *)realloc(*strp, str_size + 1)) == NULL) {
      return -1;
    }
    str_size = vsnprintf(*strp, str_size + 1, fmt,  ap);
  }
  return str_size;
}

#ifdef __cplusplus
}
#endif
#endif // _LIBCPP_SUPPORT_IBM_XLOCALE_H
