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

__attribute__((visibility("hidden")))
uint64_t __llvm_profile_get_magic(void) {
  /* Magic number to detect file format and endianness.
   *
   * Use 255 at one end, since no UTF-8 file can use that character.  Avoid 0,
   * so that utilities, like strings, don't grab it as a string.  129 is also
   * invalid UTF-8, and high enough to be interesting.
   *
   * Use "lprofr" in the centre to stand for "LLVM Profile Raw", or "lprofR"
   * for 32-bit platforms.
   */
  unsigned char R = sizeof(void *) == sizeof(uint64_t) ? 'r' : 'R';
  return
    (uint64_t)255 << 56 |
    (uint64_t)'l' << 48 |
    (uint64_t)'p' << 40 |
    (uint64_t)'r' << 32 |
    (uint64_t)'o' << 24 |
    (uint64_t)'f' << 16 |
    (uint64_t) R  <<  8 |
    (uint64_t)129;
}

__attribute__((visibility("hidden")))
uint64_t __llvm_profile_get_version(void) {
  /* This should be bumped any time the output format changes. */
  return 1;
}

__attribute__((visibility("hidden")))
void __llvm_profile_reset_counters(void) {
  uint64_t *I = __llvm_profile_begin_counters();
  uint64_t *E = __llvm_profile_end_counters();

  memset(I, 0, sizeof(uint64_t)*(E - I));
}
