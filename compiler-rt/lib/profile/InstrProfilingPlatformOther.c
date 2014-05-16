/*===- InstrProfilingPlatformOther.c - Profile data default platform ------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"

#if !defined(__APPLE__)
#include <stdlib.h>

static const __llvm_profile_data *DataFirst = NULL;
static const __llvm_profile_data *DataLast = NULL;
static const char *NamesFirst = NULL;
static const char *NamesLast = NULL;
static uint64_t *CountersFirst = NULL;
static uint64_t *CountersLast = NULL;

/*!
 * \brief Register an instrumented function.
 *
 * Calls to this are emitted by clang with -fprofile-instr-generate.  Such
 * calls are only required (and only emitted) on targets where we haven't
 * implemented linker magic to find the bounds of the sections.
 */
__attribute__((visibility("hidden")))
void __llvm_profile_register_function(void *Data_) {
  /* TODO: Only emit this function if we can't use linker magic. */
  const __llvm_profile_data *Data = (__llvm_profile_data*)Data_;
  if (!DataFirst) {
    DataFirst = Data;
    DataLast = Data + 1;
    NamesFirst = Data->Name;
    NamesLast = Data->Name + Data->NameSize;
    CountersFirst = Data->Counters;
    CountersLast = Data->Counters + Data->NumCounters;
    return;
  }

#define UPDATE_FIRST(First, New) \
  First = New < First ? New : First
  UPDATE_FIRST(DataFirst, Data);
  UPDATE_FIRST(NamesFirst, Data->Name);
  UPDATE_FIRST(CountersFirst, Data->Counters);
#undef UPDATE_FIRST

#define UPDATE_LAST(Last, New) \
  Last = New > Last ? New : Last
  UPDATE_LAST(DataLast, Data + 1);
  UPDATE_LAST(NamesLast, Data->Name + Data->NameSize);
  UPDATE_LAST(CountersLast, Data->Counters + Data->NumCounters);
#undef UPDATE_LAST
}

__attribute__((visibility("hidden")))
const __llvm_profile_data *__llvm_profile_data_begin(void) {
  return DataFirst;
}
__attribute__((visibility("hidden")))
const __llvm_profile_data *__llvm_profile_data_end(void) {
  return DataLast;
}
__attribute__((visibility("hidden")))
const char *__llvm_profile_names_begin(void) { return NamesFirst; }
__attribute__((visibility("hidden")))
const char *__llvm_profile_names_end(void) { return NamesLast; }
__attribute__((visibility("hidden")))
uint64_t *__llvm_profile_counters_begin(void) { return CountersFirst; }
__attribute__((visibility("hidden")))
uint64_t *__llvm_profile_counters_end(void) { return CountersLast; }
#endif
