/*===- InstrProfiling.c - Support library for PGO instrumentation ---------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"

/* TODO: Calculate these with linker magic. */
static __llvm_pgo_data *First = NULL;
static __llvm_pgo_data *Final = NULL;

/*!
 * \brief Register an instrumented function.
 *
 * Calls to this are emitted by clang with -fprofile-instr-generate.  Such
 * calls are only required (and only emitted) on targets where we haven't
 * implemented linker magic to find the bounds of the section.
 *
 * For now, that's all targets.
 */
void __llvm_pgo_register_function(void *Data_) {
  /* TODO: Only emit this function if we can't use linker magic. */
  __llvm_pgo_data *Data = (__llvm_pgo_data*)Data_;
  if (!First || Data < First)
    First = Data;
  if (!Final || Data > Final)
    Final = Data;
}

/*! \brief Get the first instrumentation record. */
static __llvm_pgo_data *getFirst() {
  /* TODO: Use extern + linker magic instead of a static variable. */
  return First;
}

/*! \brief Get the last instrumentation record. */
static __llvm_pgo_data *getLast() {
  /* TODO: Use extern + linker magic instead of a static variable. */
  return Final + 1;
}

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
  __llvm_pgo_data *I, *E;

  for (I = getFirst(), E = getLast(); I != E; ++I)
    writeFunction(OutputFile, I);
}

