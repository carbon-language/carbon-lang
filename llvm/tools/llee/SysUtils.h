/*===- SysUtils.h - Utilities to do low-level system stuff -------*- C -*--===*\
 *                                                                            *
 * This file contains functions used to do a variety of low-level, often      *
 * system-specific, tasks.                                                    *
 *                                                                            *
\*===----------------------------------------------------------------------===*/

#ifndef SYSUTILS_H
#define SYSUTILS_H

/*
 * isExecutableFile - This function returns true if the filename specified
 * exists and is executable.
 */
unsigned isExecutableFile(const char *ExeFileName);

/*
 * FindExecutable - Find a named executable in the path.
 */ 
char *FindExecutable(const char *ExeName);

#endif
