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

static uint64_t getMagic(void) {
  return
    (uint64_t)'l' << 56 |
    (uint64_t)'p' << 48 |
    (uint64_t)'r' << 40 |
    (uint64_t)'o' << 32 |
    (uint64_t)'f' << 24 |
    (uint64_t)'r' << 16 |
    (uint64_t)'a' <<  8 |
    (uint64_t)'w';
}

static uint64_t getVersion(void) {
  return 1;
}

int __llvm_profile_write_buffer(FILE *OutputFile) {
  /* TODO: Requires libc: break requirement by taking a char* buffer instead of
   * a FILE stream.
   */
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
  const uint64_t Magic = getMagic();
  const uint64_t Version = getVersion();
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

void __llvm_profile_reset_counters(void) {
  uint64_t *I = __llvm_profile_counters_begin();
  uint64_t *E = __llvm_profile_counters_end();

  memset(I, 0, sizeof(uint64_t)*(E - I));
}
