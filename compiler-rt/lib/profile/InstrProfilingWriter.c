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

LLVM_LIBRARY_VISIBILITY int llvmWriteProfData(WriterCallback Writer,
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

LLVM_LIBRARY_VISIBILITY int llvmWriteProfDataImpl(
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

  Header.Magic = __llvm_profile_get_magic();
  Header.Version = __llvm_profile_get_version();
  Header.DataSize = DataSize;
  Header.CountersSize = CountersSize;
  Header.NamesSize = NamesSize;
  Header.CountersDelta = (uintptr_t)CountersBegin;
  Header.NamesDelta = (uintptr_t)NamesBegin;
  Header.ValueKindLast = IPVK_Last;
  Header.ValueDataSize = ValueDataSize;
  Header.ValueDataDelta = (uintptr_t)ValueDataBegin;

  /* Write the data. */
  ProfDataIOVec IOVec[] = {
      {&Header, sizeof(__llvm_profile_header), 1},
      {DataBegin, sizeof(__llvm_profile_data), DataSize},
      {CountersBegin, sizeof(uint64_t), CountersSize},
      {NamesBegin, sizeof(char), NamesSize},
      {Zeroes, sizeof(char), Padding}};
  if (Writer(IOVec, sizeof(IOVec) / sizeof(ProfDataIOVec), &WriterCtx))
    return -1;
  if (ValueDataBegin) {
    ProfDataIOVec IOVec[1] = {{ValueDataBegin, sizeof(char), ValueDataSize}};
    if (Writer(IOVec, 1, &WriterCtx))
      return -1;
  }
  return 0;
}
