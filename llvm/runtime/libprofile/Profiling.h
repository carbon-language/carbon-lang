/*===-- Profiling.h - Profiling support library support routines --*- C -*-===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file was developed by the LLVM research group and is distributed under
|* the University of Illinois Open Source License. See LICENSE.TXT for details.
|* 
|*===----------------------------------------------------------------------===*|
|* 
|* This file defines functions shared by the various different profiling
|* implementations.
|*
\*===----------------------------------------------------------------------===*/

#ifndef PROFILING_H
#define PROFILING_H

/* save_arguments - Save argc and argv as passed into the program for the file
 * we output.
 */
int save_arguments(int argc, const char **argv);

enum ProfilingType {
  Arguments = 1,   /* The command line argument block */
  Function  = 2,   /* Function profiling information  */
  Block     = 3,   /* Block profiling information     */
  Edge      = 4,   /* Edge profiling information      */
  Path      = 5    /* Path profiling information      */
};

void write_profiling_data(enum ProfilingType PT, unsigned *Start,
                          unsigned NumElements);

#endif
