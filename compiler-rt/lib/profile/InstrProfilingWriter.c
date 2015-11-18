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

__attribute__((visibility("hidden"))) int
llvmWriteProfData(void *BufferOrFile, const uint8_t *ValueDataBegin,
                  const uint64_t ValueDataSize, WriterCallback Writer) {
  /* Match logic in __llvm_profile_write_buffer(). */
  const __llvm_profile_data *DataBegin = __llvm_profile_begin_data();
  const __llvm_profile_data *DataEnd = __llvm_profile_end_data();
  const uint64_t *CountersBegin = __llvm_profile_begin_counters();
  const uint64_t *CountersEnd = __llvm_profile_end_counters();
  const char *NamesBegin = __llvm_profile_begin_names();
  const char *NamesEnd = __llvm_profile_end_names();
  return llvmWriteProfDataImpl(BufferOrFile, Writer, DataBegin, DataEnd,
                               CountersBegin, CountersEnd, ValueDataBegin,
                               ValueDataSize, NamesBegin, NamesEnd);
}

__attribute__((visibility("hidden"))) int llvmWriteProfDataImpl(
    void *BufferOrFile, WriterCallback Writer,
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
  Header.ValueKindLast = VK_LAST;
  Header.ValueDataSize = ValueDataSize;
  Header.ValueDataDelta = (uintptr_t)ValueDataBegin;

/* Write the data. */
#define CHECK_write(Data, Size, Length, BuffOrFile)                            \
  do {                                                                         \
    if (Writer(Data, Size, Length, &BuffOrFile) != Length)                     \
      return -1;                                                               \
  } while (0)
  CHECK_write(&Header, sizeof(__llvm_profile_header), 1, BufferOrFile);
  CHECK_write(DataBegin, sizeof(__llvm_profile_data), DataSize, BufferOrFile);
  CHECK_write(CountersBegin, sizeof(uint64_t), CountersSize, BufferOrFile);
  CHECK_write(NamesBegin, sizeof(char), NamesSize, BufferOrFile);
  CHECK_write(Zeroes, sizeof(char), Padding, BufferOrFile);
  if (ValueDataBegin)
    CHECK_write(ValueDataBegin, sizeof(char), ValueDataSize, BufferOrFile);
#undef CHECK_write
  return 0;
}
