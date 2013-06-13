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
#include "llvm/Support/PathV1.h"
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


static bool Execute(void **Data, StringRef Program, const char **args,
                    const char **env, const StringRef **Redirects,
                    unsigned memoryLimit, std::string *ErrMsg) {
  Path P(Program);
  if (!Redirects)
    return Execute(Data, P, args, env, 0, memoryLimit, ErrMsg);
  Path IO[3];
  const Path *IOP[3];
  for (int I = 0; I < 3; ++I) {
    if (Redirects[I]) {
      IO[I] = *Redirects[I];
      IOP[I] = &IO[I];
    } else {
      IOP[I] = 0;
    }
  }

  return Execute(Data, P, args, env, IOP, memoryLimit, ErrMsg);
}

static int Wait(void *&Data, StringRef Program, unsigned secondsToWait,
                std::string *ErrMsg) {
  Path P(Program);
  return Wait(Data, P, secondsToWait, ErrMsg);
}

int sys::ExecuteAndWait(StringRef Program, const char **args, const char **envp,
                        const StringRef **redirects, unsigned secondsToWait,
                        unsigned memoryLimit, std::string *ErrMsg,
                        bool *ExecutionFailed) {
  void *Data = 0;
  if (Execute(&Data, Program, args, envp, redirects, memoryLimit, ErrMsg)) {
    if (ExecutionFailed) *ExecutionFailed = false;
    return Wait(Data, Program, secondsToWait, ErrMsg);
  }
  if (ExecutionFailed) *ExecutionFailed = true;
  return -1;
}

void sys::ExecuteNoWait(StringRef Program, const char **args, const char **envp,
                        const StringRef **redirects, unsigned memoryLimit,
                        std::string *ErrMsg) {
  Execute(/*Data*/ 0, Program, args, envp, redirects, memoryLimit, ErrMsg);
}

// Include the platform-specific parts of this class.
#ifdef LLVM_ON_UNIX
#include "Unix/Program.inc"
#endif
#ifdef LLVM_ON_WIN32
#include "Windows/Program.inc"
#endif
