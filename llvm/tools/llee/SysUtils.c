//===- SystemUtils.h - Utilities to do low-level system stuff --*- C++ -*--===//
//
// This file contains functions used to do a variety of low-level, often
// system-specific, tasks.
//
//===----------------------------------------------------------------------===//

#include "SysUtils.h"
#include "Config/dlfcn.h"
#include "Config/errno.h"
#include "Config/fcntl.h"
#include "Config/unistd.h"
#include "Config/sys/stat.h"
#include "Config/sys/types.h"
#include "Config/sys/wait.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * isExecutable - This function returns true if given struct stat describes the
 * file as being executable.
 */ 
unsigned isExecutable(const struct stat *buf) {
  if (!(buf->st_mode & S_IFREG))
    return 0;                         // Not a regular file?

  if (buf->st_uid == getuid())        // Owner of file?
    return buf->st_mode & S_IXUSR;
  else if (buf->st_gid == getgid())   // In group of file?
    return buf->st_mode & S_IXGRP;
  else                                // Unrelated to file?
    return buf->st_mode & S_IXOTH;
}

/*
 * isExecutableFile - This function returns true if the filename specified
 * exists and is executable.
 */
unsigned isExecutableFile(const char *ExeFileName) {
  struct stat buf;
  if (stat(ExeFileName, &buf))
    return 0;                        // Must not be executable!

  return isExecutable(&buf);
}


/*
 * FindExecutable - Find a named executable in the directories listed in $PATH.
 * If the executable cannot be found, returns NULL.
 */ 
char *FindExecutable(const char *ExeName) {
  /* Try to find the executable in the path */
  const char *PathStr = getenv("PATH");
  if (PathStr == 0) return 0;

  /* Now we have a colon separated list of directories to search, try them. */
  unsigned PathLen = strlen(PathStr);
  while (PathLen) {
    /* Find the next colon */
    const char *Colon = strchr(PathStr, ':');
    
    /* Check to see if this first directory contains the executable... */
    unsigned DirLen = Colon ? (unsigned)(Colon-PathStr) : strlen(PathStr);
    char *FilePath = alloca(sizeof(char) * (DirLen+1+strlen(ExeName)+1));
    unsigned i, e;
    for (i = 0; i != DirLen; ++i)
      FilePath[i] = PathStr[i];
    FilePath[i] = '/';
    for (i = 0, e = strlen(ExeName); i != e; ++i)
      FilePath[DirLen + 1 + i] = ExeName[i];
    FilePath[DirLen + 1 + i] = '\0';
    if (isExecutableFile(FilePath))
      return strdup(FilePath); /* Found the executable! */

    /* If Colon is NULL, there are no more colon separators and no more dirs */
    if (!Colon) break;

    /* Nope, it wasn't in this directory, check the next range! */
    PathLen -= DirLen;
    PathStr = Colon;
    while (*PathStr == ':') {   /* Advance past colons */
      PathStr++;
      PathLen--;
    }

    /* Advance past the colon */
    ++Colon;
  }

  /* If we fell out, we ran out of directories to search, return failure. */
  return NULL;
}

/*
 * The type of the execve() function is long and boring, but required.
 */
typedef int(*execveTy)(const char*, char *const[], char *const[]);

/*
 * This method finds the real `execve' call in the C library and executes the
 * given program.
 */
int executeProgram(const char *filename, char *const argv[], char *const envp[])
{
  /*
   * Find a pointer to the *real* execve() function starting the search in the
   * next library and forward, to avoid finding the one defined in this file.
   */
  char *error;
  execveTy execvePtr = (execveTy) dlsym(RTLD_NEXT, "execve");
  if ((error = dlerror()) != NULL) {
    fprintf(stderr, "%s\n", error);
    return -1;
  }

  /* Really execute the program */
  return execvePtr(filename, argv, envp);
}
