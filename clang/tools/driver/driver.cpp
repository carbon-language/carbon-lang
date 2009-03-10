//===-- driver.cpp - Clang GCC-Compatible Driver --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the entry point to the clang driver; it is a thin wrapper
// for functionality in the Driver clang library.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Option.h"
#include "clang/Driver/Options.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/System/Path.h"
#include "llvm/System/Signals.h"
using namespace clang::driver;

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();

  // FIXME: We should use GetMainExecutable here, probably, but we may
  // want to handle symbolic links slightly differently. The problem
  // is that the path derived from this will influence search paths.
  llvm::sys::Path Path(argv[0]);

  llvm::OwningPtr<Driver> TheDriver(new Driver(Path.getBasename().c_str(),
                                               Path.getDirname().c_str()));

  llvm::OwningPtr<Compilation> C(TheDriver->BuildCompilation(argc, argv));

  return C->Execute();
}
