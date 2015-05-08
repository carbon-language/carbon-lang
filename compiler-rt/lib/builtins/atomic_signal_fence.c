/*===-- atomic_signal_fence.c -----------------------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 *===------------------------------------------------------------------------===
 *
 * This file implements atomic_signal_fence from C11's stdatomic.h.
 *
 *===------------------------------------------------------------------------===
 */

#include <stdatomic.h>
#undef atomic_signal_fence
void atomic_signal_fence(memory_order order) {
  __c11_atomic_signal_fence(order);
}
