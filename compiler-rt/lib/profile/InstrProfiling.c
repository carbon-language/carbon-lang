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

/* TODO: void __llvm_profile_get_size_for_buffer(void);  */

static void writeFunction(FILE *OutputFile, const __llvm_profile_data *Data) {
  /* TODO: Requires libc: break requirement by writing directly to a buffer
   * instead of a FILE stream.
   */
  uint32_t I;
  for (I = 0; I < Data->NameSize; ++I)
    fputc(Data->Name[I], OutputFile);
  fprintf(OutputFile, "\n%" PRIu64 "\n%u\n", Data->FuncHash, Data->NumCounters);
  for (I = 0; I < Data->NumCounters; ++I)
    fprintf(OutputFile, "%" PRIu64 "\n", Data->Counters[I]);
  fprintf(OutputFile, "\n");
}

void __llvm_profile_write_buffer(FILE *OutputFile) {
  /* TODO: Requires libc: break requirement by taking a char* buffer instead of
   * a FILE stream.
   */
  const __llvm_profile_data *I, *E;

  for (I = __llvm_profile_data_begin(), E = __llvm_profile_data_end();
       I != E; ++I)
    writeFunction(OutputFile, I);
}

void __llvm_profile_reset_counters(void) {
  uint64_t *I = __llvm_profile_counters_begin();
  uint64_t *E = __llvm_profile_counters_end();

  memset(I, 0, sizeof(uint64_t)*(E - I));
}
