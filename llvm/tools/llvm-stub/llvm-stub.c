/*===- llvm-stub.c - Stub executable to run llvm bytecode files -----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This tool is used by the gccld program to enable transparent execution of
// bytecode files by the user.  Specifically, gccld outputs two files when asked
// to compile a <program> file:
//    1. It outputs the LLVM bytecode file to <program>.bc
//    2. It outputs a stub executable that runs lli on <program>.bc
//
// This allows the end user to just say ./<program> and have the JIT executed
// automatically.  On unix, the stub executable emitted is actually a bourne
// shell script that does the forwarding.  Windows doesn't not like #!/bin/sh
// programs in .exe files, so we make it an actual program, defined here.
//
//===----------------------------------------------------------------------===*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Config/unistd.h"  /* provides definition of execve */

int main(int argc, char** argv) {
  const char *Interp = getenv("LLVMINTERP");
  const char **Args;
  if (Interp == 0) Interp = "lli";

  /* Set up the command line options to pass to the JIT. */
  Args = (const char**)malloc(sizeof(char*) * (argc+2));
  /* argv[0] is the JIT */
  Args[0] = Interp;
  /* argv[1] is argv[0] + ".bc". */
  Args[1] = strcat(strcpy((char*)malloc(strlen(argv[0])+4), argv[0]), ".bc");

  /* The rest of the args are as before. */
  memcpy(Args+2, argv+1, sizeof(char*)*argc);

  /* Run the JIT. */
  execvp(Interp, (char *const*)Args);

  /* if _execv returns, the JIT could not be started. */
  fprintf(stderr, "Could not execute the LLVM JIT.  Either add 'lli' to your"
          " path, or set the\ninterpreter you want to use in the LLVMINTERP "
          "environment variable.\n");
  return 1;
}
