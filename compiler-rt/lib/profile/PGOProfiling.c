/*===- PGOProfiling.c - Support library for PGO instrumentation -----------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include <stdio.h>
#include <stdlib.h>

#define I386_FREEBSD (defined(__FreeBSD__) && defined(__i386__))

#if !I386_FREEBSD
#include <inttypes.h>
#endif

#if !defined(_MSC_VER) && !I386_FREEBSD
#include <stdint.h>
#endif

#if defined(_MSC_VER)
typedef unsigned int uint32_t;
typedef unsigned int uint64_t;
#elif I386_FREEBSD
/* System headers define 'size_t' incorrectly on x64 FreeBSD (prior to
 * FreeBSD 10, r232261) when compiled in 32-bit mode.
 */
#define PRIu64 "llu"
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#endif

typedef struct __ProfileData {
  const uint32_t NameSize;
  const uint32_t NumCounters;
  const char *const Name;
  const uint64_t *const Counters;
} __ProfileData;

/* TODO: Calculate these with linker magic. */
static __ProfileData *First = NULL;
static __ProfileData *Final = NULL;
/*!
 * \brief Register an instrumented function.
 *
 * Calls to this are emitted by clang with -fprofile-instr-generate.  Such
 * calls are only required (and only emitted) on targets where we haven't
 * implemented linker magic to find the bounds of the section.
 *
 * For now, that's all targets.
 */
void __llvm_pgo_register_function(void *Data_) {
  /* TODO: Only emit this function if we can't use linker magic. */
  __ProfileData *Data = (__ProfileData*)Data_;
  if (!First || Data < First)
    First = Data;
  if (!Final || Data > Final)
    Final = Data;
}

/*! \brief Get the first instrumentation record. */
static __ProfileData *getFirst() {
  /* TODO: Use extern + linker magic instead of a static variable. */
  return First;
}

/*! \brief Get the last instrumentation record. */
static __ProfileData *getLast() {
  /* TODO: Use extern + linker magic instead of a static variable. */
  return Final + 1;
}

/* TODO: void __llvm_pgo_get_size_for_buffer(void);  */
/* TODO: void __llvm_pgo_write_buffer(char *Buffer); */

static void writeFunction(FILE *OutputFile, const __ProfileData *Data) {
  /* TODO: Requires libc: break requirement by writing directly to a buffer
   * instead of a FILE stream.
   */
  uint32_t I;
  for (I = 0; I < Data->NameSize; ++I)
    fputc(Data->Name[I], OutputFile);
  fprintf(OutputFile, " %u\n", Data->NumCounters);
  for (I = 0; I < Data->NumCounters; ++I)
    fprintf(OutputFile, "%" PRIu64 "\n", Data->Counters[I]);
  fprintf(OutputFile, "\n");
}

/*! \brief Write instrumentation data to the given file. */
void __llvm_pgo_write_file(const char *OutputName) {
  /* TODO: Requires libc: move to separate translation unit. */
  __ProfileData *I, *E;
  FILE *OutputFile;
  if (!OutputName || !OutputName[0])
    return;
  OutputFile = fopen(OutputName, "w");
  if (!OutputFile) return;

  /* TODO: mmap file to buffer of size __llvm_pgo_get_size_for_buffer() and
   * call __llvm_pgo_write_buffer().
   */
  for (I = getFirst(), E = getLast(); I != E; ++I)
    writeFunction(OutputFile, I);

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

