/*===- InstrProfiling.c - Support library for PGO instrumentation ---------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"

/* TODO: void __llvm_pgo_get_size_for_buffer(void);  */

static void writeFunction(FILE *OutputFile, const __llvm_pgo_data *Data) {
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

void __llvm_pgo_write_buffer(FILE *OutputFile) {
  /* TODO: Requires libc: break requirement by taking a char* buffer instead of
   * a FILE stream.
   */
  const __llvm_pgo_data *I, *E;

  for (I = __llvm_pgo_data_begin(), E = __llvm_pgo_data_end();
       I != E; ++I)
    writeFunction(OutputFile, I);
}
