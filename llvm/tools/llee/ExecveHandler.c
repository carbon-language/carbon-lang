//===-- ExecveHandler.c - Replaces execve() to run LLVM files -------------===//
//
// This file implements a replacement execve() to spawn off LLVM programs to run
// transparently, without needing to be (JIT-)compiled manually by the user.
//
//===----------------------------------------------------------------------===//

#include "SysUtils.h"
#include <Config/dlfcn.h>
#include <Config/errno.h>
#include <Config/stdlib.h>
#include <stdio.h>
#include <string.h>

/*
 * This is the expected header for all valid LLVM bytecode files.
 * The first four characters must be exactly this.
 */
static const char llvmHeader[] = "llvm";

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

/*
 * This replacement execve() function first checks the file to be executed
 * to see if it is a valid LLVM bytecode file, and then either invokes our
 * execution environment or passes it on to the system execve() call.
 */
int execve(const char *filename, char *const argv[], char *const envp[])
{
  /* Open the file, test to see if first four characters are "llvm" */
  char header[4];
  FILE *file = fopen(filename, "r");
  /* Check validity of `file' */
  if (errno) { return errno; }
  /* Read the header from the file */
  size_t headerSize = strlen(llvmHeader) - 1; // ignore the NULL terminator
  size_t bytesRead = fread(header, sizeof(char), headerSize, file);
  fclose(file);
  if (bytesRead != headerSize) { 
    return EIO;
  }
  if (!strncmp(llvmHeader, header, headerSize)) {
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
    for (idx = 0; idx != argvSize; ++idx)
      LLIargs[idx+1] = argv[idx];
    LLIargs[argvSize + 1] = '\0';
    /*
    for (idx = 0; idx != argvSize+2; ++idx)
      printf("LLI args[%d] = \"%s\"\n", idx, LLIargs[idx]);
    */
    return executeProgram(LLIpath, LLIargs, envp);
  }
  executeProgram(filename, argv, envp); 
  return 0;
}
