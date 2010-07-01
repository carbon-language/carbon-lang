/*
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 */

#define SINGLE_PRECISION
#include "fp_lib.h"

fp_t __negsf2(fp_t a) {
    return fromRep(toRep(a) ^ signBit);
}
