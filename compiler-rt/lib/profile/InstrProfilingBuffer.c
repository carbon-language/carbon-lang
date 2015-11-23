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

LLVM_LIBRARY_VISIBILITY
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

LLVM_LIBRARY_VISIBILITY
uint64_t __llvm_profile_get_size_for_buffer_internal(
    const __llvm_profile_data *DataBegin, const __llvm_profile_data *DataEnd,
    const uint64_t *CountersBegin, const uint64_t *CountersEnd,
    const char *NamesBegin, const char *NamesEnd) {
  /* Match logic in __llvm_profile_write_buffer(). */
  const uint64_t NamesSize = PROFILE_RANGE_SIZE(Names) * sizeof(char);
  const uint8_t Padding = __llvm_profile_get_num_padding_bytes(NamesSize);
  return sizeof(__llvm_profile_header) +
         PROFILE_RANGE_SIZE(Data) * sizeof(__llvm_profile_data) +
         PROFILE_RANGE_SIZE(Counters) * sizeof(uint64_t) + NamesSize + Padding;
}

/* The buffer writer is reponsponsible in keeping writer state
 * across the call.
 */
static uint32_t bufferWriter(ProfDataIOVec *IOVecs, uint32_t NumIOVecs,
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

LLVM_LIBRARY_VISIBILITY
__llvm_profile_write_buffer(char *Buffer) {
  return llvmWriteProfData(bufferWriter, Buffer, 0, 0);
}

LLVM_LIBRARY_VISIBILITY int __llvm_profile_write_buffer_internal(
    char *Buffer, const __llvm_profile_data *DataBegin,
    const __llvm_profile_data *DataEnd, const uint64_t *CountersBegin,
    const uint64_t *CountersEnd, const char *NamesBegin, const char *NamesEnd) {
  return llvmWriteProfDataImpl(bufferWriter, Buffer, DataBegin, DataEnd,
                               CountersBegin, CountersEnd, 0, 0, NamesBegin,
                               NamesEnd);
}
