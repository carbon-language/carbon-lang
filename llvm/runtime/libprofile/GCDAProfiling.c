/*===- GCDAProfiling.c - Support library for GCDA file emission -----------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|* 
|*===----------------------------------------------------------------------===*|
|* 
|* This file implements the call back routines for the gcov profiling
|* instrumentation pass. Link against this library when running code through
|* the -insert-gcov-profiling LLVM pass.
|*
|* We emit files in a corrupt version of GCOV's "gcda" file format. These files
|* are only close enough that LCOV will happily parse them. Anything that lcov
|* ignores is missing.
|*
\*===----------------------------------------------------------------------===*/

#include "llvm/Support/DataTypes.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* #define DEBUG_GCDAPROFILING */

/*
 * --- GCOV file format I/O primitives ---
 */

static FILE *output_file = NULL;

static void write_int32(uint32_t i) {
  fwrite(&i, 4, 1, output_file);
}

static void write_int64(uint64_t i) {
  uint32_t lo, hi;
  lo = i & 0x00000000ffffffff;
  hi = i & 0xffffffff00000000;

  write_int32(lo);
  write_int32(hi);
}

/*
 * --- LLVM line counter API ---
 */

/* A file in this case is a translation unit. Each .o file built with line
 * profiling enabled will emit to a different file. Only one file may be
 * started at a time.
 */
void llvm_gcda_start_file(const char *filename) {
  output_file = fopen(filename, "w+");

  /* gcda file, version 404*, stamp LLVM. */
  fwrite("adcg*404MVLL", 12, 1, output_file);

#ifdef DEBUG_GCDAPROFILING
  printf("[%s]\n", filename);
#endif
}

void llvm_gcda_emit_function(uint32_t ident) {
#ifdef DEBUG_GCDAPROFILING
  printf("function id=%x\n", ident);
#endif

  /* function tag */  
  fwrite("\0\0\0\1", 4, 1, output_file);
  write_int32(2);
  write_int32(ident);
  write_int32(0);
}

void llvm_gcda_emit_arcs(uint32_t num_counters, uint64_t *counters) {
  uint32_t i;
  /* counter #1 (arcs) tag */
  fwrite("\0\0\xa1\1", 4, 1, output_file);
  write_int32(num_counters * 2);
  for (i = 0; i < num_counters; ++i) {
    write_int64(counters[i]);
  }

#ifdef DEBUG_GCDAPROFILING
  printf("  %u arcs\n", num_counters);
  for (i = 0; i < num_counters; ++i) {
    printf("  %llu\n", (unsigned long long)counters[i]);
  }
#endif
}

void llvm_gcda_end_file() {
  /* Write out EOF record. */
  fwrite("\0\0\0\0\0\0\0\0", 8, 1, output_file);
  fclose(output_file);
  output_file = NULL;

#ifdef DEBUG_GCDAPROFILING
  printf("-----\n");
#endif
}
