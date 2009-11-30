//===-- cc1_main.cpp - Clang CC1 Driver -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the entry point to the clang -cc1 functionality.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream.h"

int cc1_main(const char **ArgBegin, const char **ArgEnd,
             const char *Argv0, void *MainAddr) {
  llvm::errs() << "cc1 argv:";
  for (const char **i = ArgBegin; i != ArgEnd; ++i)
    llvm::errs() << " \"" << *i << '"';
  llvm::errs() << "\n";

  return 0;
}
