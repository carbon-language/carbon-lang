/*===- InstrProfiling.c - Support library for PGO instrumentation ---------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"
#include <string.h>

uint64_t __llvm_profile_get_magic(void) {
  /* Magic number to detect file format and endianness. */
  return
    (uint64_t)'l' << 56 |
    (uint64_t)'p' << 48 |
    (uint64_t)'r' << 40 |
    (uint64_t)'o' << 32 |
    (uint64_t)'f' << 24 |
    (uint64_t)'r' << 16 |
    (uint64_t)'a' <<  8 |
    (uint64_t)'w';
}

uint64_t __llvm_profile_get_version(void) {
  /* This should be bumped any time the output format changes. */
  return 1;
}

void __llvm_profile_reset_counters(void) {
  uint64_t *I = __llvm_profile_counters_begin();
  uint64_t *E = __llvm_profile_counters_end();

  memset(I, 0, sizeof(uint64_t)*(E - I));
}
