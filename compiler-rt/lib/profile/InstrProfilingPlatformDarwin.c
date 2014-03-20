/*===- InstrProfilingDarwin.c - Profile data on Darwin --------------------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"

/* Use linker magic to find the bounds of the Data section. */
extern __llvm_pgo_data DataStart __asm("section$start$__DATA$__llvm_pgo_data");
extern __llvm_pgo_data DataEnd   __asm("section$end$__DATA$__llvm_pgo_data");
extern char NamesStart __asm("section$start$__DATA$__llvm_pgo_names");
extern char NamesEnd   __asm("section$end$__DATA$__llvm_pgo_names");
extern uint64_t CountersStart __asm("section$start$__DATA$__llvm_pgo_cnts");
extern uint64_t CountersEnd   __asm("section$end$__DATA$__llvm_pgo_cnts");

const __llvm_pgo_data *__llvm_pgo_data_begin() { return &DataStart; }
const __llvm_pgo_data *__llvm_pgo_data_end()   { return &DataEnd; }
const char *__llvm_pgo_names_begin() { return &NamesStart; }
const char *__llvm_pgo_names_end()   { return &NamesEnd; }
uint64_t *__llvm_pgo_counters_begin() { return &CountersStart; }
uint64_t *__llvm_pgo_counters_end()   { return &CountersEnd; }
