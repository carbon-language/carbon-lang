//===-- Program.cpp - Implement OS Program Concept --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This header file implements the operating system Program concept.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Program.h"
#include "llvm/Config/config.h"
#include "llvm/Support/system_error.h"
using namespace llvm;
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

static bool Execute(void **Data, const Path &path, const char **args,
                    const char **env, const sys::Path **redirects,
                    unsigned memoryLimit, std::string *ErrMsg);

static int Wait(void *&Data, const Path &path, unsigned secondsToWait,
                std::string *ErrMsg);

int sys::ExecuteAndWait(StringRef path, const char **args, const char **env,
                        const StringRef **redirects, unsigned secondsToWait,
                        unsigned memoryLimit, std::string *ErrMsg,
                        bool *ExecutionFailed) {
  Path P(path);
  if (!redirects)
    return ExecuteAndWait(P, args, env, 0, secondsToWait, memoryLimit, ErrMsg,
                          ExecutionFailed);
  Path IO[3];
  const Path *IOP[3];
  for (int I = 0; I < 3; ++I) {
    if (redirects[I]) {
      IO[I] = *redirects[I];
      IOP[I] = &IO[I];
    } else {
      IOP[I] = 0;
   }
  }

  return ExecuteAndWait(P, args, env, IOP, secondsToWait, memoryLimit, ErrMsg,
                        ExecutionFailed);
}

int sys::ExecuteAndWait(const Path &path, const char **args, const char **envp,
                        const Path **redirects, unsigned secondsToWait,
                        unsigned memoryLimit, std::string *ErrMsg,
                        bool *ExecutionFailed) {
  void *Data = 0;
  if (Execute(&Data, path, args, envp, redirects, memoryLimit, ErrMsg)) {
    if (ExecutionFailed) *ExecutionFailed = false;
    return Wait(Data, path, secondsToWait, ErrMsg);
  }
  if (ExecutionFailed) *ExecutionFailed = true;
  return -1;
}

void sys::ExecuteNoWait(const Path &path, const char **args, const char **envp,
                        const Path **redirects, unsigned memoryLimit,
                        std::string *ErrMsg) {
  Execute(/*Data*/ 0, path, args, envp, redirects, memoryLimit, ErrMsg);
}

// Include the platform-specific parts of this class.
#ifdef LLVM_ON_UNIX
#include "Unix/Program.inc"
#endif
#ifdef LLVM_ON_WIN32
#include "Windows/Program.inc"
#endif
