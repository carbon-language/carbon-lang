/*===- InstrProfilingBuffer.c - Write instrumentation to a memory buffer --===*\
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

__attribute__((visibility("hidden")))
uint64_t __llvm_profile_get_size_for_buffer(void) {
  const __llvm_profile_data *DataBegin = __llvm_profile_begin_data();
  const __llvm_profile_data *DataEnd = __llvm_profile_end_data();
  const uint64_t *CountersBegin = __llvm_profile_begin_counters();
  const uint64_t *CountersEnd = __llvm_profile_end_counters();
  const char *NamesBegin = __llvm_profile_begin_names();
  const char *NamesEnd = __llvm_profile_end_names();

  return __llvm_profile_get_size_for_buffer_internal(
      DataBegin, DataEnd, CountersBegin, CountersEnd, NamesBegin, NamesEnd);
}

#define PROFILE_RANGE_SIZE(Range) (Range##End - Range##Begin)

__attribute__((visibility("hidden")))
uint64_t __llvm_profile_get_size_for_buffer_internal(
        const __llvm_profile_data *DataBegin,
        const __llvm_profile_data *DataEnd, const uint64_t *CountersBegin,
        const uint64_t *CountersEnd, const char *NamesBegin,
        const char *NamesEnd) {
  /* Match logic in __llvm_profile_write_buffer(). */
  const uint64_t NamesSize = PROFILE_RANGE_SIZE(Names) * sizeof(char);
  const uint64_t Padding = sizeof(uint64_t) - NamesSize % sizeof(uint64_t);
  return sizeof(uint64_t) * PROFILE_HEADER_SIZE +
      PROFILE_RANGE_SIZE(Data) * sizeof(__llvm_profile_data) +
      PROFILE_RANGE_SIZE(Counters) * sizeof(uint64_t) +
      NamesSize + Padding;
}

__attribute__((visibility("hidden")))
int __llvm_profile_write_buffer(char *Buffer) {
  /* Match logic in __llvm_profile_get_size_for_buffer().
   * Match logic in __llvm_profile_write_file().
   */
  const __llvm_profile_data *DataBegin = __llvm_profile_begin_data();
  const __llvm_profile_data *DataEnd = __llvm_profile_end_data();
  const uint64_t *CountersBegin = __llvm_profile_begin_counters();
  const uint64_t *CountersEnd   = __llvm_profile_end_counters();
  const char *NamesBegin = __llvm_profile_begin_names();
  const char *NamesEnd   = __llvm_profile_end_names();

  return __llvm_profile_write_buffer_internal(Buffer, DataBegin, DataEnd,
                                              CountersBegin, CountersEnd,
                                              NamesBegin, NamesEnd);
}

__attribute__((visibility("hidden")))
int __llvm_profile_write_buffer_internal(
    char *Buffer, const __llvm_profile_data *DataBegin,
    const __llvm_profile_data *DataEnd, const uint64_t *CountersBegin,
    const uint64_t *CountersEnd, const char *NamesBegin, const char *NamesEnd) {
  /* Match logic in __llvm_profile_get_size_for_buffer().
   * Match logic in __llvm_profile_write_file().
   */

  /* Calculate size of sections. */
  const uint64_t DataSize = DataEnd - DataBegin;
  const uint64_t CountersSize = CountersEnd - CountersBegin;
  const uint64_t NamesSize = NamesEnd - NamesBegin;
  const uint64_t Padding = sizeof(uint64_t) - NamesSize % sizeof(uint64_t);

  /* Enough zeroes for padding. */
  const char Zeroes[sizeof(uint64_t)] = {0};

  /* Create the header. */
  uint64_t Header[PROFILE_HEADER_SIZE];
  Header[0] = __llvm_profile_get_magic();
  Header[1] = __llvm_profile_get_version();
  Header[2] = DataSize;
  Header[3] = CountersSize;
  Header[4] = NamesSize;
  Header[5] = (uintptr_t)CountersBegin;
  Header[6] = (uintptr_t)NamesBegin;

  /* Write the data. */
#define UPDATE_memcpy(Data, Size) \
  do {                            \
    memcpy(Buffer, Data, Size);   \
    Buffer += Size;               \
  } while (0)
  UPDATE_memcpy(Header,  PROFILE_HEADER_SIZE * sizeof(uint64_t));
  UPDATE_memcpy(DataBegin,     DataSize      * sizeof(__llvm_profile_data));
  UPDATE_memcpy(CountersBegin, CountersSize  * sizeof(uint64_t));
  UPDATE_memcpy(NamesBegin,    NamesSize     * sizeof(char));
  UPDATE_memcpy(Zeroes,        Padding       * sizeof(char));
#undef UPDATE_memcpy

  return 0;
}
