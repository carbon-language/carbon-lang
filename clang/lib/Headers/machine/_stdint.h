/*===---- _stdint.h - C99 type-related definitions on FreeBSD -------------===*\
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

#ifndef __MACHINE_XSTDINT_H
#define __MACHINE_XSTDINT_H

#include_next <machine/_stdint.h>

/* Fix some definitions on x86-64 FreeBSD 9.2 in 32-bit mode. */
#if defined(__FreeBSD__) && defined(__i386__)
# include <osreldate.h>
# if __FreeBSD_version <= 902001  // v9.2
#  if !defined(__cplusplus) || defined(__STDC_CONSTANT_MACROS)
#   undef INT64_C
#   define INT64_C(c) (c ## LL)

#   undef UINT64_C
#   define UINT64_C(c) (c ## ULL)
#  endif  /* !defined(__cplusplus) || defined(__STDC_CONSTANT_MACROS) */

#  if !defined(__cplusplus) || defined(__STDC_LIMIT_MACROS)
#   undef INT64_MIN
#   define INT64_MIN (-0x7fffffffffffffffLL-1)

#   undef INT64_MAX
#   define INT64_MAX 0x7fffffffffffffffLL

#   undef UINT64_MAX
#   define UINT64_MAX 0xffffffffffffffffULL

#   undef INTPTR_MIN
#   define INTPTR_MIN INT32_MIN

#   undef INTPTR_MAX
#   define INTPTR_MAX INT32_MAX

#   undef UINTPTR_MAX
#   define UINTPTR_MAX UINT32_MAX

#   undef PTRDIFF_MIN
#   define PTRDIFF_MIN INT32_MIN

#   undef PTRDIFF_MAX
#   define PTRDIFF_MAX INT32_MAX

#   undef SIZE_MAX
#   define SIZE_MAX UINT32_MAX
#  endif  /* !defined(__cplusplus) || defined(__STDC_LIMIT_MACROS) */
# endif  /* __FreeBSD_version <= 902001 */
#endif  /* defined(__FreeBSD__) && defined(__i386__) */

#endif /* !__MACHINE_XSTDINT_H */
