/*===-- atomic_flag_test_and_set.c ------------------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 *===------------------------------------------------------------------------===
 *
 * This file implements atomic_flag_test_and_set from C11's stdatomic.h.
 *
 *===------------------------------------------------------------------------===
 */

#include <stdatomic.h>
#undef atomic_flag_test_and_set
_Bool atomic_flag_test_and_set(volatile atomic_flag *object) {
  return __c11_atomic_exchange(&(object)->_Value, 1, __ATOMIC_SEQ_CST);
}
