/*===- LineProfiling.c - Support library for line profiling ---------------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|* 
|*===----------------------------------------------------------------------===*|
|* 
|* This file implements the call back routines for the line profiling
|* instrumentation pass. Link against this library when running code through
|* the -insert-line-profiling LLVM pass.
|*
\*===----------------------------------------------------------------------===*/

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

/* A file in this case is a translation unit. Each .o file built with line
 * profiling enabled will emit to a different file. Only one file may be
 * started at a time.
 */
void llvm_prof_linectr_start_file(const char *orig_filename) {
  printf("[%s]\n", orig_filename);
}

/* Emit data about a counter to the data file. */
void llvm_prof_linectr_emit_counter(const char *dir, const char *file,
                                    uint32_t line, uint32_t column,
                                    uint64_t *counter) {
  printf("%s/%s:%u:%u %lu\n", dir, file, line, column, *counter);
}

void llvm_prof_linectr_end_file() {
  printf("-----\n");
}
