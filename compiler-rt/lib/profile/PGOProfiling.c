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

static FILE *OutputFile = NULL;

/*
 * A list of functions to write out the data.
 */
typedef void (*writeout_fn)();

struct writeout_fn_node {
  writeout_fn fn;
  struct writeout_fn_node *next;
};

static struct writeout_fn_node *writeout_fn_head = NULL;
static struct writeout_fn_node *writeout_fn_tail = NULL;

void llvm_pgo_emit(const char *MangledName, uint32_t NumCounters,
                   uint64_t *Counters) {
  uint32_t i;
  fprintf(OutputFile, "%s %u\n", MangledName, NumCounters);
  for (i = 0; i < NumCounters; ++i)
    fprintf(OutputFile, "%" PRIu64 "\n", Counters[i]);
  fprintf(OutputFile, "\n");
}

void llvm_pgo_register_writeout_function(writeout_fn fn) {
  struct writeout_fn_node *new_node = malloc(sizeof(struct writeout_fn_node));
  new_node->fn = fn;
  new_node->next = NULL;

  if (!writeout_fn_head) {
    writeout_fn_head = writeout_fn_tail = new_node;
  } else {
    writeout_fn_tail->next = new_node;
    writeout_fn_tail = new_node;
  }
}

void llvm_pgo_writeout_files() {
  const char *OutputName = getenv("LLVM_PROFILE_FILE");
  if (OutputName == NULL || OutputName[0] == '\0')
    OutputName = "default.profdata";
  OutputFile = fopen(OutputName, "w");
  if (!OutputFile) return;

  while (writeout_fn_head) {
    struct writeout_fn_node *node = writeout_fn_head;
    writeout_fn_head = writeout_fn_head->next;
    node->fn();
    free(node);
  }

  fclose(OutputFile);
}

void llvm_pgo_init(writeout_fn wfn) {
  static int atexit_ran = 0;

  if (wfn)
    llvm_pgo_register_writeout_function(wfn);

  if (atexit_ran == 0) {
    atexit_ran = 1;
    atexit(llvm_pgo_writeout_files);
  }
}
