//===- tools/lld/lld.cpp - Linker Driver Dispatcher -----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// This is the entry point to the lld driver. This is a thin wrapper which
/// dispatches to the given platform specific driver.
///
//===----------------------------------------------------------------------===//

#include "lld/Core/LLVM.h"
#include "lld/Driver/Driver.h"

#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"

using namespace lld;


/// Universal linker main().  This linker eumulates the gnu, darwin, or
/// windows linker based on the tool name or if the first argument is
/// -flavor.
int main(int argc, const char *argv[]) {
  // Standard set up, so program fails gracefully.
  llvm::sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram stackPrinter(argc, argv);
  llvm::llvm_shutdown_obj shutdown;

  if (UniversalDriver::link(argc, argv))
    return 1;
  else
    return 0;
}
