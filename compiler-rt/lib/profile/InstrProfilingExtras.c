/*===- InstrProfilingExtras.c - Support library for PGO instrumentation ---===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"

/*! \brief Write instrumentation data to the given file. */
void __llvm_pgo_write_file(const char *OutputName) {
  /* TODO: Requires libc: move to separate translation unit. */
  FILE *OutputFile;
  if (!OutputName || !OutputName[0])
    return;
  OutputFile = fopen(OutputName, "w");
  if (!OutputFile) return;

  /* TODO: mmap file to buffer of size __llvm_pgo_get_size_for_buffer() and
   * pass the buffer in, instead of the file.
   */
  __llvm_pgo_write_buffer(OutputFile);

  fclose(OutputFile);
}

/*! \brief Write instrumentation data to the default file. */
void __llvm_pgo_write_default_file() {
  /* TODO: Requires libc: move to separate translation unit. */
  const char *OutputName = getenv("LLVM_PROFILE_FILE");
  if (OutputName == NULL || OutputName[0] == '\0')
    OutputName = "default.profdata";
  __llvm_pgo_write_file(OutputName);
}

/*!
 * \brief Register to write instrumentation data to the default file at exit.
 */
void __llvm_pgo_register_write_atexit() {
  /* TODO: Requires libc: move to separate translation unit. */
  static int HasBeenRegistered = 0;

  if (!HasBeenRegistered) {
    HasBeenRegistered = 1;
    atexit(__llvm_pgo_write_default_file);
  }
}
