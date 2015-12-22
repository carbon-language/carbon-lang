/*===- InstrProfilingWriter.c - Write instrumentation to a file or buffer -===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"
#include "InstrProfilingInternal.h"
#include <string.h>

/* The buffer writer is reponsponsible in keeping writer state
 * across the call.
 */
COMPILER_RT_VISIBILITY uint32_t llvmBufferWriter(ProfDataIOVec *IOVecs,
                                                 uint32_t NumIOVecs,
                                                 void **WriterCtx) {
  uint32_t I;
  char **Buffer = (char **)WriterCtx;
  for (I = 0; I < NumIOVecs; I++) {
    size_t Length = IOVecs[I].ElmSize * IOVecs[I].NumElm;
    memcpy(*Buffer, IOVecs[I].Data, Length);
    *Buffer += Length;
  }
  return 0;
}

COMPILER_RT_VISIBILITY int llvmWriteProfData(WriterCallback Writer,
                                             void *WriterCtx,
                                             const uint8_t *ValueDataBegin,
                                             const uint64_t ValueDataSize) {
  /* Match logic in __llvm_profile_write_buffer(). */
  const __llvm_profile_data *DataBegin = __llvm_profile_begin_data();
  const __llvm_profile_data *DataEnd = __llvm_profile_end_data();
  const uint64_t *CountersBegin = __llvm_profile_begin_counters();
  const uint64_t *CountersEnd = __llvm_profile_end_counters();
  const char *NamesBegin = __llvm_profile_begin_names();
  const char *NamesEnd = __llvm_profile_end_names();
  return llvmWriteProfDataImpl(Writer, WriterCtx, DataBegin, DataEnd,
                               CountersBegin, CountersEnd, ValueDataBegin,
                               ValueDataSize, NamesBegin, NamesEnd);
}

COMPILER_RT_VISIBILITY int llvmWriteProfDataImpl(
    WriterCallback Writer, void *WriterCtx,
    const __llvm_profile_data *DataBegin, const __llvm_profile_data *DataEnd,
    const uint64_t *CountersBegin, const uint64_t *CountersEnd,
    const uint8_t *ValueDataBegin, const uint64_t ValueDataSize,
    const char *NamesBegin, const char *NamesEnd) {

  /* Calculate size of sections. */
  const uint64_t DataSize = DataEnd - DataBegin;
  const uint64_t CountersSize = CountersEnd - CountersBegin;
  const uint64_t NamesSize = NamesEnd - NamesBegin;
  const uint64_t Padding = __llvm_profile_get_num_padding_bytes(NamesSize);

  /* Enough zeroes for padding. */
  const char Zeroes[sizeof(uint64_t)] = {0};

  /* Create the header. */
  __llvm_profile_header Header;

  if (!DataSize)
    return 0;

  /* Initialize header struture.  */
#define INSTR_PROF_RAW_HEADER(Type, Name, Init) Header.Name = Init;
#include "InstrProfData.inc"

  /* Write the data. */
  ProfDataIOVec IOVec[] = {
      {&Header, sizeof(__llvm_profile_header), 1},
      {DataBegin, sizeof(__llvm_profile_data), DataSize},
      {CountersBegin, sizeof(uint64_t), CountersSize},
      {NamesBegin, sizeof(char), NamesSize},
      {Zeroes, sizeof(char), Padding}};
  if (Writer(IOVec, sizeof(IOVec) / sizeof(*IOVec), &WriterCtx))
    return -1;
  if (ValueDataBegin) {
    ProfDataIOVec IOVec2[] = {{ValueDataBegin, sizeof(char), ValueDataSize}};
    if (Writer(IOVec2, sizeof(IOVec2) / sizeof(*IOVec2), &WriterCtx))
      return -1;
  }
  return 0;
}
