//===- SystemUtils.cpp - Utilities for low-level system tasks -------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains functions used to do a variety of low-level, often
// system-specific, tasks.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SystemUtils.h"
#include "llvm/System/Program.h"

using namespace llvm;

/// isStandardOutAConsole - Return true if we can tell that the standard output
/// stream goes to a terminal window or console.
bool llvm::isStandardOutAConsole() {
#if HAVE_ISATTY
  return isatty(1);
#endif
  // If we don't have isatty, just return false.
  return false;
}


/// FindExecutable - Find a named executable, giving the argv[0] of program
/// being executed. This allows us to find another LLVM tool if it is built
/// into the same directory, but that directory is neither the current
/// directory, nor in the PATH.  If the executable cannot be found, return an
/// empty string.
///
#undef FindExecutable   // needed on windows :(
sys::Path llvm::FindExecutable(const std::string &ExeName,
                               const std::string &ProgramPath) {
  // First check the directory that the calling program is in.  We can do this  
  // if ProgramPath contains at least one / character, indicating that it is a
  // relative path to bugpoint itself.
  sys::Path Result ( ProgramPath );
  Result.elideFile();
  if (!Result.isEmpty()) {
    Result.appendFile(ExeName);
    if (Result.executable()) 
      return Result;
  }

  return sys::Program::FindProgramByName(ExeName);
}
