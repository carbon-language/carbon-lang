/*
 * kmp_ftn_extra.c -- Fortran 'extra' linkage support for OpenMP.
 * $Revision: 42951 $
 * $Date: 2014-01-21 14:41:41 -0600 (Tue, 21 Jan 2014) $
 */


//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#include "kmp.h"

#if KMP_OS_WINDOWS
#   define KMP_FTN_ENTRIES KMP_FTN_PLAIN
#elif KMP_OS_UNIX
#   define KMP_FTN_ENTRIES KMP_FTN_APPEND
#endif

// Note: This string is not printed when KMP_VERSION=1.
char const __kmp_version_ftnextra[] = KMP_VERSION_PREFIX "Fortran \"extra\" OMP support: "
#ifdef KMP_FTN_ENTRIES
    "yes";
#   define FTN_STDCALL /* nothing to do */
#   include "kmp_ftn_os.h"
#   include "kmp_ftn_entry.h"
#else
    "no";
#endif /* KMP_FTN_ENTRIES */
