/*===---- intrin.h - Microsoft VS compatible X86 intrinsics -----------------===
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
 *===-----------------------------------------------------------------------===
 */

/* Unless we're compiling targeting MSVC platform, this header shouldn't even
 * *exist*. If there is a system header with the same name, defer to that,
 * etherwise produce an error for the user.
 */
#ifndef _MSC_VER
# if defined(__has_include_next) && __has_include_next(<intrin.h>)
#  include_next <intrin.h>
# else
#  error The <intrin.h> builtin header is for use when targeting Windows and \
         provides MSVC compatible intrinsic declarations. It shouldn't be used \
         on non-Windows targets. Instead, see <x86intrin.h> which is supported \
         by Clang, GCC, and ICC on all platforms.
# endif
#else /* _MSC_VER */

#ifndef __INTRIN_H
#define __INTRIN_H

/* These headers need to be provided by intrin.h in case users depend on any of
 * their contents. However, some of them are unavailable in freestanding
 * builds, so guard them appropriately.
 */
#if __STDC_HOSTED__
# include <crtdefs.h>
# include <setjmp.h>
#endif
#include <stddef.h>

/* Microsoft includes all of the intrinsics, and then restricts their
 * availability based on the particular target CPU; with Clang we rely on the
 * guarded includes used in our generic x86intrin header to pull in the
 * intrinsic declarations / definitions which should be available for the
 * target CPU variant.
 */
#include <x86intrin.h>

/* FIXME: We need to provide declarations for Microsoft-specific intrinsics in
 * addition to the chip-vendor intrinsics provided by x86intrin.h.
 */

#endif /* __INTRIN_H */

#endif /* _MSC_VER */
