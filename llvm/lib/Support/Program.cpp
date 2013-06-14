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

static bool Execute(void **Data, StringRef Program, const char **args,
                    const char **env, const StringRef **Redirects,
                    unsigned memoryLimit, std::string *ErrMsg);

static int Wait(void *&Data, StringRef Program, unsigned secondsToWait,
                std::string *ErrMsg);

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
