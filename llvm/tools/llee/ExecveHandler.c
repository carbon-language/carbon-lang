//===-- ExecveHandler.c - Replaces execve() to run LLVM files -------------===//
//
// This file implements a replacement execve() to spawn off LLVM programs to run
// transparently, without needing to be (JIT-)compiled manually by the user.
//
//===----------------------------------------------------------------------===//

#include "SysUtils.h"
#include "Config/unistd.h"
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>

/*
 * This is the expected header for all valid LLVM bytecode files.
 * The first four characters must be exactly this.
 */
static const char llvmHeader[] = "llvm";

/*
 * This replacement execve() function first checks the file to be executed
 * to see if it is a valid LLVM bytecode file, and then either invokes our
 * execution environment or passes it on to the system execve() call.
 */
int execve(const char *filename, char *const argv[], char *const envp[])
{
  /* Open the file, test to see if first four characters are "llvm" */
  size_t headerSize = strlen(llvmHeader);
  char header[headerSize];
  char* realFilename = 0;
  /* 
   * If the program is specified with a relative or absolute path, 
   * then just use the path and filename as is, otherwise search for it.
   */
  if (filename[0] != '.' && filename[0] != '/')
    realFilename = FindExecutable(filename);
  else
    realFilename = (char*) filename;
  if (!realFilename) {
    fprintf(stderr, "Cannot find path to `%s', exiting.\n", filename);
    return -1;
  }
  errno = 0;
  int file = open(realFilename, O_RDONLY);
  /* Check validity of `file' */
  if (errno) return EIO;
  /* Read the header from the file */
  ssize_t bytesRead = read(file, header, headerSize);
  close(file);
  if (bytesRead != (ssize_t)headerSize) return EIO;
  if (!memcmp(llvmHeader, header, headerSize)) {
    /* 
     * This is a bytecode file, so execute the JIT with the program and
     * parameters.
     */
    unsigned argvSize, idx;
    for (argvSize = 0, idx = 0; argv[idx] && argv[idx][0]; ++idx)
      ++argvSize;
    char **LLIargs = (char**) malloc(sizeof(char*) * (argvSize+2));
    char *LLIpath = FindExecutable("lli");
    if (!LLIpath) {
      fprintf(stderr, "Cannot find path to `lli', exiting.\n");
      return -1;
    }
    LLIargs[0] = LLIpath;
    LLIargs[1] = realFilename;
    for (idx = 1; idx != argvSize; ++idx)
      LLIargs[idx+1] = argv[idx];
    LLIargs[argvSize + 1] = '\0';
    return executeProgram(LLIpath, LLIargs, envp);
  }
  return executeProgram(filename, argv, envp); 
}
