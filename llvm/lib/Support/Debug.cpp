//===-- Debug.cpp - An easy way to add debug output to your code ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a handle way of adding debugging information to your
// code, without it being enabled all of the time, and without having to add
// command line options to enable it.
//
// In particular, just wrap your code with the DEBUG() macro, and it will be
// enabled automatically if you specify '-debug' on the command-line.
// Alternatively, you can also use the SET_DEBUG_TYPE("foo") macro to specify
// that your debug code belongs to class "foo".  Then, on the command line, you
// can specify '-debug-only=foo' to enable JUST the debug information for the
// foo class.
//
// When compiling in release mode, the -debug-* options and all code in DEBUG()
// statements disappears, so it does not effect the runtime of the code.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

// All Debug.h functionality is a no-op in NDEBUG mode.
#ifndef NDEBUG
bool llvm::DebugFlag;  // DebugFlag - Exported boolean set by the -debug option

// -debug - Command line option to enable the DEBUG statements in the passes.
// This flag may only be enabled in debug builds.
static cl::opt<bool, true>
Debug("debug", cl::desc("Enable debug output"), cl::Hidden,
      cl::location(DebugFlag));

static std::string CurrentDebugType;
static struct DebugOnlyOpt {
  void operator=(const std::string &Val) const {
    DebugFlag |= !Val.empty();
    CurrentDebugType = Val;
  }
} DebugOnlyOptLoc;

static cl::opt<DebugOnlyOpt, true, cl::parser<std::string> >
DebugOnly("debug-only", cl::desc("Enable a specific type of debug output"),
          cl::Hidden, cl::value_desc("debug string"),
          cl::location(DebugOnlyOptLoc), cl::ValueRequired);

// isCurrentDebugType - Return true if the specified string is the debug type
// specified on the command line, or if none was specified on the command line
// with the -debug-only=X option.
//
bool llvm::isCurrentDebugType(const char *DebugType) {
  return CurrentDebugType.empty() || DebugType == CurrentDebugType;
}
#else
// Avoid "has no symbols" warning.
int Debug_dummy = 0;
#endif
