//===- SystemUtils.h - Utilities to do low-level system stuff --*- C++ -*--===//
//
// This file contains functions used to do a variety of low-level, often
// system-specific, tasks.
//
//===----------------------------------------------------------------------===//

#include "SysUtils.h"
#include "Config/sys/types.h"
#include "Config/sys/stat.h"
#include "Config/fcntl.h"
#include "Config/sys/wait.h"
#include "Config/unistd.h"
#include "Config/errno.h"
#include <stdlib.h>
#include <string.h>

/*
 * isExecutableFile - This function returns true if the filename specified
 * exists and is executable.
 */
unsigned isExecutableFile(const char *ExeFileName) {
  struct stat Buf;
  if (stat(ExeFileName, &Buf))
    return 0;                        // Must not be executable!

  if (!(Buf.st_mode & S_IFREG))
    return 0;                        // Not a regular file?

  if (Buf.st_uid == getuid())        // Owner of file?
    return Buf.st_mode & S_IXUSR;
  else if (Buf.st_gid == getgid())   // In group of file?
    return Buf.st_mode & S_IXGRP;
  else                               // Unrelated to file?
    return Buf.st_mode & S_IXOTH;
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
