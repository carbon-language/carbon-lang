/*===- InstrProfilingExtras.c - Support library for PGO instrumentation ---===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"

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

void __llvm_profile_write_file() {
  const char *Filename = CurrentFilename;

#define UPDATE_FILENAME(NextFilename) \
  if (!Filename || !Filename[0]) Filename = NextFilename
  UPDATE_FILENAME(getenv("LLVM_PROFILE_FILE"));
  UPDATE_FILENAME("default.profdata");
#undef UPDATE_FILENAME

  __llvm_profile_write_file_with_name(Filename);
}

void __llvm_profile_register_write_file_atexit() {
  static int HasBeenRegistered = 0;

  if (!HasBeenRegistered) {
    HasBeenRegistered = 1;
    atexit(__llvm_profile_write_file);
  }
}
