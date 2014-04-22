/*===- InstrProfilingPlatformDarwin.c - Profile data on Darwin ------------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"

#if defined(__APPLE__)
/* Use linker magic to find the bounds of the Data section. */
extern __llvm_profile_data DataStart __asm("section$start$__DATA$__llvm_prf_data");
extern __llvm_profile_data DataEnd   __asm("section$end$__DATA$__llvm_prf_data");
extern char NamesStart __asm("section$start$__DATA$__llvm_prf_names");
extern char NamesEnd   __asm("section$end$__DATA$__llvm_prf_names");
extern uint64_t CountersStart __asm("section$start$__DATA$__llvm_prf_cnts");
extern uint64_t CountersEnd   __asm("section$end$__DATA$__llvm_prf_cnts");

const __llvm_profile_data *__llvm_profile_data_begin(void) {
  return &DataStart;
}
const __llvm_profile_data *__llvm_profile_data_end(void) {
  return &DataEnd;
}
const char *__llvm_profile_names_begin(void) { return &NamesStart; }
const char *__llvm_profile_names_end(void) { return &NamesEnd; }
uint64_t *__llvm_profile_counters_begin(void) { return &CountersStart; }
uint64_t *__llvm_profile_counters_end(void) { return &CountersEnd; }
#endif
