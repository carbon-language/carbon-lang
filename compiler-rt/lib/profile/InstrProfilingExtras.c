/*===- InstrProfilingExtras.c - Support library for PGO instrumentation ---===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"
#include <string.h>

static void __llvm_profile_write_file_with_name(const char *OutputName) {
  FILE *OutputFile;
  if (!OutputName || !OutputName[0])
    return;
  OutputFile = fopen(OutputName, "w");
  if (!OutputFile) return;

  /* TODO: mmap file to buffer of size __llvm_profile_get_size_for_buffer() and
   * pass the buffer in, instead of the file.
   */
  __llvm_profile_write_buffer(OutputFile);

  fclose(OutputFile);
}

static const char *CurrentFilename = NULL;
void __llvm_profile_set_filename(const char *Filename) {
  CurrentFilename = Filename;
}

int getpid(void);
void __llvm_profile_write_file(void) {
  char *AllocatedFilename = NULL;
  int I, J;

#define MAX_PID_SIZE 16
  char PidChars[MAX_PID_SIZE] = { 0 };
  int PidLength = 0;
  int NumPids = 0;

  // Get the filename.
  const char *Filename = CurrentFilename;
#define UPDATE_FILENAME(NextFilename) \
  if (!Filename || !Filename[0]) Filename = NextFilename
  UPDATE_FILENAME(getenv("LLVM_PROFILE_FILE"));
  UPDATE_FILENAME("default.profdata");
#undef UPDATE_FILENAME

  // Check the filename for "%p", which indicates a pid-substitution.
  for (I = 0; Filename[I]; ++I)
    if (Filename[I] == '%' && Filename[++I] == 'p')
      if (!NumPids++) {
        PidLength = snprintf(PidChars, MAX_PID_SIZE, "%d", getpid());
        if (PidLength <= 0)
          return;
      }
  if (NumPids) {
    // Allocate enough space for the substituted filename.
    AllocatedFilename = (char*)malloc(I + NumPids*(PidLength - 2) + 1);
    if (!AllocatedFilename)
      return;

    // Construct the new filename.
    for (I = 0, J = 0; Filename[I]; ++I)
      if (Filename[I] == '%') {
        if (Filename[++I] == 'p') {
          memcpy(AllocatedFilename + J, PidChars, PidLength);
          J += PidLength;
        }
        // Drop any unknown substitutions.
      } else
        AllocatedFilename[J++] = Filename[I];
    AllocatedFilename[J] = 0;

    // Actually use the computed name.
    Filename = AllocatedFilename;
  }

  // Write the file.
  __llvm_profile_write_file_with_name(Filename);

  // Free the filename.
  if (AllocatedFilename)
    free(AllocatedFilename);
}

void __llvm_profile_register_write_file_atexit(void) {
  static int HasBeenRegistered = 0;

  if (!HasBeenRegistered) {
    HasBeenRegistered = 1;
    atexit(__llvm_profile_write_file);
  }
}
