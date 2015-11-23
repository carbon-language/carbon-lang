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
LLVM_LIBRARY_VISIBILITY
extern __llvm_profile_data
    DataStart __asm("section$start$__DATA$" INSTR_PROF_DATA_SECT_NAME_STR);
LLVM_LIBRARY_VISIBILITY
extern __llvm_profile_data
    DataEnd __asm("section$end$__DATA$" INSTR_PROF_DATA_SECT_NAME_STR);
LLVM_LIBRARY_VISIBILITY
extern char
    NamesStart __asm("section$start$__DATA$" INSTR_PROF_NAME_SECT_NAME_STR);
LLVM_LIBRARY_VISIBILITY
extern char NamesEnd __asm("section$end$__DATA$" INSTR_PROF_NAME_SECT_NAME_STR);
LLVM_LIBRARY_VISIBILITY
extern uint64_t
    CountersStart __asm("section$start$__DATA$" INSTR_PROF_CNTS_SECT_NAME_STR);
LLVM_LIBRARY_VISIBILITY
extern uint64_t
    CountersEnd __asm("section$end$__DATA$" INSTR_PROF_CNTS_SECT_NAME_STR);

LLVM_LIBRARY_VISIBILITY
const __llvm_profile_data *__llvm_profile_begin_data(void) {
  return &DataStart;
}
LLVM_LIBRARY_VISIBILITY
const __llvm_profile_data *__llvm_profile_end_data(void) { return &DataEnd; }
LLVM_LIBRARY_VISIBILITY
const char *__llvm_profile_begin_names(void) { return &NamesStart; }
LLVM_LIBRARY_VISIBILITY
const char *__llvm_profile_end_names(void) { return &NamesEnd; }
LLVM_LIBRARY_VISIBILITY
uint64_t *__llvm_profile_begin_counters(void) { return &CountersStart; }
LLVM_LIBRARY_VISIBILITY
uint64_t *__llvm_profile_end_counters(void) { return &CountersEnd; }
#endif
