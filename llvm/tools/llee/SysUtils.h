/*===- SysUtils.h - Utilities to do low-level system stuff -------*- C -*--===*\
 *                                                                            *
 * This file contains functions used to do a variety of low-level, often      *
 * system-specific, tasks.                                                    *
 *                                                                            *
\*===----------------------------------------------------------------------===*/

#ifndef SYSUTILS_H
#define SYSUTILS_H

struct stat;

/*
 * isExecutable - This function returns true if given struct stat describes the
 * file as being executable.
 */ 
unsigned isExecutable(const struct stat *buf);
  
/*
 * isExecutableFile - This function returns true if the filename specified
 * exists and is executable.
 */
unsigned isExecutableFile(const char *ExeFileName);

/*
 * FindExecutable - Find a named executable in the path.
 */ 
char *FindExecutable(const char *ExeName);

/*
 * This method finds the real `execve' call in the C library and executes the
 * given program.
 */
int
executeProgram(const char *filename, char *const argv[], char *const envp[]);

#endif
