/*===-- CommonProfiling.c - Profiling support library support -------------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file was developed by the LLVM research group and is distributed under
|* the University of Illinois Open Source License. See LICENSE.TXT for details.
|* 
|*===----------------------------------------------------------------------===*|
|* 
|* This file implements functions used by the various different types of
|* profiling implementations.
|*
\*===----------------------------------------------------------------------===*/

#include "Profiling.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int SavedArgc = 0;
static const char **SavedArgv = 0;

/* save_arguments - Save argc and argv as passed into the program for the file
 * we output.
 */
void save_arguments(int argc, const char **argv) {
  if (SavedArgv) return;  /* This can be called multiple times */

  /* FIXME: this should copy the arguments out of argv into a string of our own,
   * because the program might modify the arguments!
   */
  SavedArgc = argc;
  SavedArgv = argv;
}




/* write_profiling_data - Write a raw block of profiling counters out to the
 * llvmprof.out file.  Note that we allow programs to be instrumented with
 * multiple different kinds of instrumentation.  For this reason, this function
 * may be called more than once.
 */
void write_profiling_data(enum ProfilingType PT, unsigned *Start,
                          unsigned NumElements) {
  static int OutFile = -1;
  int PTy;
  
  /* If this is the first time this function is called, open the output file for
   * appending, creating it if it does not already exist.
   */
  if (OutFile == -1) {
    off_t Offset;
    OutFile = open("llvmprof.out", O_CREAT | O_WRONLY | O_APPEND, 0666);
    if (OutFile == -1) {
      perror("LLVM profiling: while opening 'llvmprof.out'");
      return;
    }

    /* Output the command line arguments to the file. */
    {
      const char *Args = "";
      int PTy = Arguments;
      int ArgLength = strlen(Args);
      int Zeros = 0;
      write(OutFile, &PTy, sizeof(int));
      write(OutFile, &ArgLength, sizeof(int));
      write(OutFile, Args, ArgLength);
      /* Pad out to a multiple of four bytes */
      if (ArgLength & 3)
        write(OutFile, &Zeros, 4-(ArgLength&3));
    }
  }
 
  /* Write out this record! */
  PTy = PT;
  write(OutFile, &PTy, sizeof(int));
  write(OutFile, Start, NumElements*sizeof(unsigned));
}
