// -*- C++ -*-
//===--------------------------- support/win32/locale.h --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Locale stuff
// FIXME: the *_l functions are fairly new, only available on Vista?/7+
#include <xlocinfo.h> // 
#define locale_t _locale_t
#define strtoll_l _strtoi64_l
#define strtoull_l _strtoui64_l
// FIXME: current msvcrt does not know about long double
#define strtold_l _strtod_l
#define isdigit_l _isdigit_l
#define isxdigit_l _isxdigit_l
#define strcoll_l _strcoll_l
#define strxfrm_l _strxfrm_l
#define wcscoll_l _wcscoll_l
#define wcsxfrm_l _wcsxfrm_l
#define toupper_l _toupper_l
#define tolower_l _tolower_l
#define iswspace_l _iswspace_l
#define iswprint_l _iswprint_l
#define iswcntrl_l _iswcntrl_l
#define iswupper_l _iswupper_l
#define iswlower_l _iswlower_l
#define iswalpha_l _iswalpha_l
#define iswdigit_l _iswdigit_l
#define iswpunct_l _iswpunct_l
#define iswxdigit_l _iswxdigit_l
#define towupper_l _towupper_l
#define towlower_l _towlower_l
#define strftime_l _strftime_l
inline int isblank_l( int c, locale_t /*loc*/ )
{
    return ( c == ' ' || c == '\t' );
}
inline int iswblank_l( wint_t c, locale_t /*loc*/ )
{
    return ( c == L' ' || c == L'\t' );
}
#define freelocale _free_locale
// ignore base; it is always 0 in libc++ code
inline locale_t newlocale( int mask, const char * locale, locale_t /*base*/ )
{
    return _create_locale( mask, locale );
}

// FIXME: first call _configthreadlocale(_ENABLE_PER_THREAD_LOCALE) somewhere
// FIXME: return types are different, need to make locale_t from char*
inline locale_t uselocale(locale_t newloc)
{
    return _create_locale( LC_ALL, setlocale(LC_ALL, newloc->locinfo->lc_category[LC_ALL].locale) );
}

#define LC_COLLATE_MASK _M_COLLATE
#define LC_CTYPE_MASK _M_CTYPE
#define LC_MONETARY_MASK _M_MONETARY
#define LC_NUMERIC_MASK _M_NUMERIC
#define LC_TIME_MASK _M_TIME
#define LC_MESSAGES_MASK _M_MESSAGES
#define LC_ALL_MASK (  LC_COLLATE_MASK \
                     | LC_CTYPE_MASK \
                     | LC_MESSAGES_MASK \
                     | LC_MONETARY_MASK \
                     | LC_NUMERIC_MASK \
                     | LC_TIME_MASK )
