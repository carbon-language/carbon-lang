// -*- C++ -*-
//===--------------------- support/ibm/xlocale.h -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_SUPPORT_IBM_XLOCALE_H
#define _LIBCPP_SUPPORT_IBM_XLOCALE_H

#if defined(_AIX)
#include "cstdlib"

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(_AIX71)
// AIX 7.1 and higher has these definitions.  Definitions and stubs
// are provied here as a temporary workaround on AIX 6.1.

#define LC_COLLATE_MASK         1
#define LC_CTYPE_MASK           2
#define LC_MESSAGES_MASK        4
#define LC_MONETARY_MASK        8
#define LC_NUMERIC_MASK         16
#define LC_TIME_MASK            32
#define LC_ALL_MASK             (LC_COLLATE_MASK | LC_CTYPE_MASK | \
                                 LC_MESSAGES_MASK | LC_MONETARY_MASK |\
                                 LC_NUMERIC_MASK | LC_TIME_MASK)

typedef void* locale_t;

// The following are stubs.  They are not supported on AIX 6.1.
static inline
locale_t newlocale(int category_mask, const char *locale, locale_t base)
{
  _LC_locale_t *newloc, *loc;
  if ((loc = (_LC_locale_t *)__xopen_locale(locale)) == NULL)
  {
    errno = EINVAL;
    return (locale_t)0;
  }
  if ((newloc = (_LC_locale_t *)calloc(1, sizeof(_LC_locale_t))) == NULL)
  {
    errno = ENOMEM;
    return (locale_t)0;
  }
  if (!base)
    base = (_LC_locale_t *)__xopen_locale("C");
  memcpy(newloc, base, sizeof (_LC_locale_t));
  if (category_mask & LC_COLLATE_MASK) 
    newloc->lc_collate = loc->lc_collate;
  if (category_mask & LC_CTYPE_MASK)
    newloc->lc_ctype = loc->lc_ctype;
  //if (category_mask & LC_MESSAGES_MASK)
  //  newloc->lc_messages = loc->lc_messages;
  if (category_mask & LC_MONETARY_MASK)
    newloc->lc_monetary = loc->lc_monetary;
  if (category_mask & LC_TIME_MASK)
    newloc->lc_time = loc->lc_time;
  if (category_mask & LC_NUMERIC_MASK)
    newloc->lc_numeric = loc->lc_numeric;
  return (locale_t)newloc; 
}
static inline
void freelocale(locale_t locobj)
{
  free(locobj);
}
static inline
locale_t uselocale(locale_t newloc)
{
  return (locale_t)0;
}

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

// strftime_l() is defined by POSIX. However, AIX 7.1 does not have it
// implemented yet.
static inline
size_t strftime_l(char *__s, size_t __size, const char *__fmt,
                  const struct tm *__tm, locale_t locale) {
  return __xstrftime(locale, __s, __size, __fmt, __tm);
}

// The following are not POSIX routines.  These are quick-and-dirty hacks
// to make things pretend to work
static inline
long long strtoll_l(const char *__nptr, char **__endptr,
    int __base, locale_t locale) {
  return strtoll(__nptr, __endptr, __base);
}
static inline
long strtol_l(const char *__nptr, char **__endptr,
    int __base, locale_t locale) {
  return strtol(__nptr, __endptr, __base);
}
static inline
long double strtold_l(const char *__nptr, char **__endptr,
    locale_t locale) {
  return strtold(__nptr, __endptr);
}
static inline
unsigned long long strtoull_l(const char *__nptr, char **__endptr,
    int __base, locale_t locale) {
  return strtoull(__nptr, __endptr, __base);
}
static inline
unsigned long strtoul_l(const char *__nptr, char **__endptr,
    int __base, locale_t locale) {
  return strtoul(__nptr, __endptr, __base);
}

static inline
int vasprintf(char **strp, const char *fmt, va_list ap)
{
  const size_t buff_size = 256;
  int str_size;
  if ((*strp = (char *)malloc(buff_size)) == NULL)
  {
    return -1;
  }
  if ((str_size = vsnprintf(*strp, buff_size, fmt,  ap)) >= buff_size)
  {
    if ((*strp = (char *)realloc(*strp, str_size + 1)) == NULL)
    {
      return -1;
    }
    str_size = vsnprintf(*strp, str_size + 1, fmt,  ap);
  }
  return str_size;
}  

#ifdef __cplusplus
}
#endif
#endif // defined(_AIX)
#endif // _LIBCPP_SUPPORT_IBM_XLOCALE_H
