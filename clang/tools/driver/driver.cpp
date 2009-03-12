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

#include "clang/Frontend/TextDiagnosticPrinter.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/Config/config.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"
#include "llvm/System/Signals.h"
using namespace clang;
using namespace clang::driver;

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();

  llvm::OwningPtr<DiagnosticClient> 
    DiagClient(new TextDiagnosticPrinter(llvm::errs()));

  Diagnostic Diags(DiagClient.get());

  // FIXME: We should use GetMainExecutable here, probably, but we may
  // want to handle symbolic links slightly differently. The problem
  // is that the path derived from this will influence search paths.
  llvm::sys::Path Path(argv[0]);

  // FIXME: Use the triple of the host, not the triple that we were
  // compiled on.
  llvm::OwningPtr<Driver> TheDriver(new Driver(Path.getBasename().c_str(),
                                               Path.getDirname().c_str(),
                                               LLVM_HOSTTRIPLE,
                                               Diags));
                                               
  llvm::OwningPtr<Compilation> C(TheDriver->BuildCompilation(argc, argv));

  // If there were errors building the compilation, quit now.
  if (Diags.getNumErrors())
    return 1;

  return C->Execute();
}
