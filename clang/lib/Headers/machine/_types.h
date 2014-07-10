/*===---- _types.h - Machine-dependent type definitions on FreeBSD --------===*\
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

#ifndef __MACHINE_XTYPES_H
#define __MACHINE_XTYPES_H

/* Fix some definitions on x86-64 FreeBSD 9.2 in 32-bit mode. */
#if defined(__FreeBSD__) && defined(__i386__)
# include <osreldate.h>
# if __FreeBSD_version <= 902001  // v9.2
#  define __FIX_FREEBSD_9_2_DEFINITIONS
# endif  /* __FreeBSD_version <= 902001 */
#endif  /* defined(__FreeBSD__) && defined(__i386__) */

#if defined(__FIX_FREEBSD_9_2_DEFINITIONS)
#define __int64_t __broken_int64_t
#define __uint64_t __broken_uint64_t
#define __critical_t __broken_critical_t
#define __intfptr_t __broken_intfptr_t
#define __intptr_t __broken_intptr_t
#define __intmax_t __broken_intmax_t
#define __int_fast64_t __broken_int_fast64_t
#define __int_least64_t __broken_int_least64_t
#define __ptrdiff_t __broken_ptrdiff_t
#define __register_t __broken_register_t
#define __segsz_t __broken_segsz_t
#define __size_t __broken_size_t
#define __ssize_t __broken_ssize_t
#define __time_t __broken_time_t
#define __uintfptr_t __broken_uintfptr_t
#define __uintptr_t __broken_uintptr_t
#define __uintmax_t __broken_uintmax_t
#define __uint_fast64_t __broken_uint_fast64_t
#define __uint_least64_t __broken_uint_least64_t
#define __u_register_t __broken_u_register_t
#define __vm_offset_t __broken_vm_offset_t
#define __vm_paddr_t __broken_vm_paddr_t
#define __vm_size_t __broken_vm_size_t
#define __vm_ooffset_t __broken_vm_ooffset_t
#define __vm_pindex_t __broken_vm_pindex_t
#endif  // defined(__FIX_FREEBSD_9_2_DEFINITIONS)

#include_next <machine/_types.h>

#if defined(__FIX_FREEBSD_9_2_DEFINITIONS)
#undef __int64_t
typedef long long __int64_t;

#undef __uint64_t
typedef unsigned long long __uint64_t;

#undef __critical_t
typedef __int32_t __critical_t;

#undef __intfptr_t
typedef __int32_t __intfptr_t;

#undef __intptr_t
typedef __int32_t __intptr_t;

#undef __intmax_t
typedef __int64_t __intmax_t;

#undef __int_fast64_t
typedef __int64_t __int_fast64_t;

#undef __int_least64_t
typedef __int64_t __int_least64_t;

#undef __ptrdiff_t
typedef __int32_t __ptrdiff_t;

#undef __register_t
typedef __int64_t __register_t;

#undef __segsz_t
typedef __int32_t __segsz_t;

#undef __size_t
typedef __uint32_t __size_t;

#undef __ssize_t
typedef __int32_t __ssize_t;

#undef __time_t
typedef __int32_t __time_t;

#undef __uintfptr_t
typedef __uint32_t __uintfptr_t;

#undef __uintptr_t
typedef __uint32_t __uintptr_t;

#undef __uintmax_t
typedef __uint64_t __uintmax_t;

#undef __uint_fast64_t
typedef __uint64_t __uint_fast64_t;

#undef __uint_least64_t
typedef __uint64_t __uint_least64_t;

#undef __u_register_t
typedef __uint32_t __u_register_t;

#undef __vm_offset_t
typedef __uint32_t __vm_offset_t;

#undef __vm_paddr_t
#ifdef PAE
typedef __uint64_t __vm_paddr_t;
#else
typedef __uint32_t __vm_paddr_t;
#endif

#undef __vm_size_t
typedef __uint32_t __vm_size_t;

#undef __vm_ooffset_t
typedef __int64_t __vm_ooffset_t;

#undef __vm_pindex_t
typedef __uint64_t __vm_pindex_t;
#endif  // defined(__FIX_FREEBSD_9_2_DEFINITIONS)

#endif  /* !__MACHINE_XTYPES_H */
