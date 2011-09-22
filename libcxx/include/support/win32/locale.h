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
#define newlocale _create_locale
#define freelocale _free_locale
// FIXME: first call _configthreadlocale(_ENABLE_PER_THREAD_LOCALE) somewhere
// FIXME: return types are different, need to make locale_t from char*
inline locale_t uselocale(locale_t newloc)
{
    return newlocale( LC_ALL, setlocale(LC_ALL, newloc->locinfo->lc_category[LC_ALL].locale) );
}

#define LC_COLLATE_MASK _M_COLLATE
#define LC_CTYPE_MASK _M_CTYPE
#define LC_MONETARY_MASK _M_MONETARY
#define LC_NUMERIC_MASK _M_NUMERIC
#define LC_TIME_MASK _M_TIME
#define LC_MESSAGES_MASK _M_MESSAGES

enum { NL_SETD=0, NL_CAT_LOCALE=1 };
