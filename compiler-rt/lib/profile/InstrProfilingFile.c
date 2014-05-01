/*===- InstrProfilingFile.c - Write instrumentation to a file -------------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int writeFile(FILE *File) {
  /* Match logic in __llvm_profile_write_buffer(). */
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
#define CHECK_fwrite(Data, Size, Length, File) \
  do { if (fwrite(Data, Size, Length, File) != Length) return -1; } while (0)
  CHECK_fwrite(Header,        sizeof(uint64_t), PROFILE_HEADER_SIZE, File);
  CHECK_fwrite(DataBegin,     sizeof(__llvm_profile_data), DataSize, File);
  CHECK_fwrite(CountersBegin, sizeof(uint64_t), CountersSize, File);
  CHECK_fwrite(NamesBegin,    sizeof(char), NamesSize, File);
#undef CHECK_fwrite

   return 0;
}

static int writeFileWithName(const char *OutputName) {
  int RetVal;
  FILE *OutputFile;
  if (!OutputName || !OutputName[0])
    return -1;
  OutputFile = fopen(OutputName, "w");
  if (!OutputFile)
    return -1;

  RetVal = writeFile(OutputFile);

  fclose(OutputFile);
  return RetVal;
}

static const char *CurrentFilename = NULL;
void __llvm_profile_set_filename(const char *Filename) {
  CurrentFilename = Filename;
}

int getpid(void);
int __llvm_profile_write_file(void) {
  char *AllocatedFilename = NULL;
  int I, J;
  int RetVal;

#define MAX_PID_SIZE 16
  char PidChars[MAX_PID_SIZE] = { 0 };
  int PidLength = 0;
  int NumPids = 0;

  /* Get the filename. */
  const char *Filename = CurrentFilename;
#define UPDATE_FILENAME(NextFilename) \
  if (!Filename || !Filename[0]) Filename = NextFilename
  UPDATE_FILENAME(getenv("LLVM_PROFILE_FILE"));
  UPDATE_FILENAME("default.profraw");
#undef UPDATE_FILENAME

  /* Check the filename for "%p", which indicates a pid-substitution. */
  for (I = 0; Filename[I]; ++I)
    if (Filename[I] == '%' && Filename[++I] == 'p')
      if (!NumPids++) {
        PidLength = snprintf(PidChars, MAX_PID_SIZE, "%d", getpid());
        if (PidLength <= 0)
          return -1;
      }
  if (NumPids) {
    /* Allocate enough space for the substituted filename. */
    AllocatedFilename = (char*)malloc(I + NumPids*(PidLength - 2) + 1);
    if (!AllocatedFilename)
      return -1;

    /* Construct the new filename. */
    for (I = 0, J = 0; Filename[I]; ++I)
      if (Filename[I] == '%') {
        if (Filename[++I] == 'p') {
          memcpy(AllocatedFilename + J, PidChars, PidLength);
          J += PidLength;
        }
        /* Drop any unknown substitutions. */
      } else
        AllocatedFilename[J++] = Filename[I];
    AllocatedFilename[J] = 0;

    /* Actually use the computed name. */
    Filename = AllocatedFilename;
  }

  /* Write the file. */
  RetVal = writeFileWithName(Filename);

  /* Free the filename. */
  if (AllocatedFilename)
    free(AllocatedFilename);

  return RetVal;
}

static void writeFileWithoutReturn(void) {
  __llvm_profile_write_file();
}

int __llvm_profile_register_write_file_atexit(void) {
  static int HasBeenRegistered = 0;

  if (HasBeenRegistered)
    return 0;

  HasBeenRegistered = 1;
  return atexit(writeFileWithoutReturn);
}
