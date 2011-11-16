/* ===-- int_lib.h - configuration header for compiler-rt  -----------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file is a configuration header for compiler-rt.
 * This file is not part of the interface of this library.
 *
 * ===----------------------------------------------------------------------===
 */

#ifndef INT_LIB_H
#define INT_LIB_H

/* Assumption: Signed integral is 2's complement. */
/* Assumption: Right shift of signed negative is arithmetic shift. */
/* Assumption: Endianness is little or big (not mixed). */

/* ABI macro definitions */

#if __ARM_EABI__
# define ARM_EABI_FNALIAS(aeabi_name, name)         \
  void __aeabi_##aeabi_name() __attribute__((alias("__" #name)));
# define COMPILER_RT_ABI __attribute__((pcs("aapcs")))
#else
# define ARM_EABI_FNALIAS(aeabi_name, name)
# define COMPILER_RT_ABI
#endif

/* Include the standard compiler builtin headers we use functionality from. */
#include <limits.h>
#include <stdint.h>
#include <stdbool.h>
#include <float.h>

/* Include the commonly used internal type definitions. */
#include "int_types.h"

/* Include internal utility function declarations. */
#include "int_util.h"

#endif /* INT_LIB_H */
