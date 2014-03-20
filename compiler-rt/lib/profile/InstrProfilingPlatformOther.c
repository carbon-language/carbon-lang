/*===- InstrProfilingDefault.c - Profile data default platfrom ------------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"

static const __llvm_pgo_data *DataFirst = NULL;
static const __llvm_pgo_data *DataLast = NULL;
static const char *NamesFirst = NULL;
static const char *NamesLast = NULL;
static const uint64_t *CountersFirst = NULL;
static const uint64_t *CountersLast = NULL;

/*!
 * \brief Register an instrumented function.
 *
 * Calls to this are emitted by clang with -fprofile-instr-generate.  Such
 * calls are only required (and only emitted) on targets where we haven't
 * implemented linker magic to find the bounds of the sections.
 */
void __llvm_pgo_register_function(void *Data_) {
  /* TODO: Only emit this function if we can't use linker magic. */
  const __llvm_pgo_data *Data = (__llvm_pgo_data*)Data_;
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

const __llvm_pgo_data *__llvm_pgo_data_begin() { return DataFirst; }
const __llvm_pgo_data *__llvm_pgo_data_end() { return DataLast; }
const char *__llvm_pgo_names_begin() { return NamesFirst; }
const char *__llvm_pgo_names_end() { return NamesLast; }
const uint64_t *__llvm_pgo_counters_begin() { return CountersFirst; }
const uint64_t *__llvm_pgo_counters_end() { return CountersLast; }
