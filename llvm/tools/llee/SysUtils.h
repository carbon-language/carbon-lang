/*===- sysutils.h - Utilities to do low-level system stuff -------*- C -*--===*\
 *                                                                            *
 * This file contains functions used to do a variety of low-level, often      *
 * system-specific, tasks.                                                    *
 *                                                                            *
\*===----------------------------------------------------------------------===*/

#ifndef SYSUTILS_H
#define SYSUTILS_H

typedef unsigned bool;
enum { false = 0, true = 1 };

/*
 * isExecutableFile - This function returns true if the filename specified
 * exists and is executable.
 */
bool isExecutableFile(const char *ExeFileName);

/*
 * FindExecutable - Find a named executable, giving the argv[0] of program
 * being executed. This allows us to find another LLVM tool if it is built into
 * the same directory, but that directory is neither the current directory, nor
 * in the PATH.  If the executable cannot be found, return an empty string.
 */ 
char *FindExecutable(const char *ExeName);

#endif
