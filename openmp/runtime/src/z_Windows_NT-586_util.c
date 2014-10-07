/*
 * z_Windows_NT-586_util.c -- platform specific routines.
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

#if (KMP_ARCH_X86 || KMP_ARCH_X86_64)
/* Only 32-bit "add-exchange" instruction on IA-32 architecture causes us to
 * use compare_and_store for these routines
 */

kmp_int32
__kmp_test_then_or32( volatile kmp_int32 *p, kmp_int32 d )
{
    kmp_int32 old_value, new_value;

    old_value = TCR_4( *p );
    new_value = old_value | d;

    while ( ! __kmp_compare_and_store32 ( p, old_value, new_value ) )
    {
        KMP_CPU_PAUSE();
        old_value = TCR_4( *p );
        new_value = old_value | d;
    }

    return old_value;
}

kmp_int32
__kmp_test_then_and32( volatile kmp_int32 *p, kmp_int32 d )
{
    kmp_int32 old_value, new_value;

    old_value = TCR_4( *p );
    new_value = old_value & d;

    while ( ! __kmp_compare_and_store32 ( p, old_value, new_value ) )
    {
        KMP_CPU_PAUSE();
        old_value = TCR_4( *p );
        new_value = old_value & d;
    }
    return old_value;
}

#if KMP_ARCH_X86
kmp_int64
__kmp_test_then_add64( volatile kmp_int64 *p, kmp_int64 d )
{
    kmp_int64 old_value, new_value;

    old_value = TCR_8( *p );
    new_value = old_value + d;
    while ( ! __kmp_compare_and_store64 ( p, old_value, new_value ) )
    {
        KMP_CPU_PAUSE();
        old_value = TCR_8( *p );
        new_value = old_value + d;
    }

    return old_value;
}
#endif /* KMP_ARCH_X86 */

kmp_int64
__kmp_test_then_or64( volatile kmp_int64 *p, kmp_int64 d )
{
    kmp_int64 old_value, new_value;

    old_value = TCR_8( *p );
    new_value = old_value | d;
    while ( ! __kmp_compare_and_store64 ( p, old_value, new_value ) )
    {
        KMP_CPU_PAUSE();
        old_value = TCR_8( *p );
        new_value = old_value | d;
    }

    return old_value;
}

kmp_int64
__kmp_test_then_and64( volatile kmp_int64 *p, kmp_int64 d )
{
    kmp_int64 old_value, new_value;

    old_value = TCR_8( *p );
    new_value = old_value & d;
    while ( ! __kmp_compare_and_store64 ( p, old_value, new_value ) )
    {
        KMP_CPU_PAUSE();
        old_value = TCR_8( *p );
        new_value = old_value & d;
    }

    return old_value;
}

#endif /* KMP_ARCH_X86 || KMP_ARCH_X86_64 */

/* ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------ */

