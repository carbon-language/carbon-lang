/*===- OSInterface.h - Interface to query OS for functionality ---*- C -*--===*\
 *                                                                            *
 * This file defines the prototype interface that we will expect operating    *
 * systems to implement if they wish to support offline cachine.              *
 *                                                                            *
\*===----------------------------------------------------------------------===*/

#ifndef OS_INTERFACE_H
#define OS_INTERFACE_H

#include "Config/sys/types.h"

struct stat;

/*
 * llvmStat - equivalent to stat(3), except the key may not necessarily
 * correspond to a file by that name, implementation is up to the OS.
 * Values returned in buf are similar as they are in Unix.
 */
void llvmStat(const char *key, struct stat *buf);

/*
 * llvmWriteFile - implements a 'save' of a file in the OS. 'key' may not
 * necessarily map to a file of the same name.
 * Returns:
 * 0 - success
 * non-zero - error
 */ 
int llvmWriteFile(const char *key, const void *data, size_t len);

/* 
 * llvmLoadFile - tells the OS to load data corresponding to a particular key
 * somewhere into memory.
 * Returns:
 * 0 - failure
 * non-zero - address of loaded file
 *
 * Value of size is the length of data loaded into memory.
 */ 
void* llvmReadFile(const char *key, size_t *size);

/*
 * llvmExecve - execute a file from cache. This is a temporary proof-of-concept
 * because we do not relocate what we can read from disk.
 */ 
int llvmExecve(const char *filename, char *const argv[], char *const envp[]);

#endif
