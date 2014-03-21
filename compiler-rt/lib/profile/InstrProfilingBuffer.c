/*===- InstrProfilingBuffer.c - Write instrumentation to a memory buffer --===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"
#include <string.h>

/* TODO: uint64_t __llvm_profile_get_size_for_buffer(void) */

int __llvm_profile_write_buffer(FILE *OutputFile) {
  const __llvm_profile_data *DataBegin = __llvm_profile_data_begin();
  const __llvm_profile_data *DataEnd = __llvm_profile_data_end();
  const uint64_t *CountersBegin = __llvm_profile_counters_begin();
  const uint64_t *CountersEnd   = __llvm_profile_counters_end();
  const char *NamesBegin = __llvm_profile_names_begin();
  const char *NamesEnd   = __llvm_profile_names_end();

  /* Calculate size of sections. */
  const uint64_t DataSize = DataEnd - DataBegin;
  const uint64_t CountersSize = CountersEnd - CountersBegin;
  const uint64_t NamesSize = NamesEnd - NamesBegin;

  /* Get rest of header data. */
  const uint64_t Magic = __llvm_profile_get_magic();
  const uint64_t Version = __llvm_profile_get_version();
  const uint64_t CountersDelta = (uint64_t)CountersBegin;
  const uint64_t NamesDelta = (uint64_t)NamesBegin;

#define CHECK_fwrite(Data, Size, Length, File) \
  do { if (fwrite(Data, Size, Length, File) != Length) return -1; } while (0)

  /* Write the header. */
  CHECK_fwrite(&Magic,         sizeof(uint64_t), 1, OutputFile);
  CHECK_fwrite(&Version,       sizeof(uint64_t), 1, OutputFile);
  CHECK_fwrite(&DataSize,      sizeof(uint64_t), 1, OutputFile);
  CHECK_fwrite(&CountersSize,  sizeof(uint64_t), 1, OutputFile);
  CHECK_fwrite(&NamesSize,     sizeof(uint64_t), 1, OutputFile);
  CHECK_fwrite(&CountersDelta, sizeof(uint64_t), 1, OutputFile);
  CHECK_fwrite(&NamesDelta,    sizeof(uint64_t), 1, OutputFile);

  /* Write the data. */
  CHECK_fwrite(DataBegin, sizeof(__llvm_profile_data), DataSize, OutputFile);
  CHECK_fwrite(CountersBegin, sizeof(uint64_t), CountersSize, OutputFile);
  CHECK_fwrite(NamesBegin, sizeof(char), NamesSize, OutputFile);

#undef CHECK_fwrite

   return 0;
}
