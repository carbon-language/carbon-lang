//===- SystemUtils.cpp - Utilities for low-level system tasks -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions used to do a variety of low-level, often
// system-specific, tasks.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SystemUtils.h"
#include "llvm/System/Process.h"
#include "llvm/System/Program.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

bool llvm::CheckBitcodeOutputToConsole(raw_ostream &stream_to_check,
                                       bool print_warning) {
  if (stream_to_check.is_displayed()) {
    if (print_warning) {
      errs() << "WARNING: You're attempting to print out a bitcode file.\n"
             << "This is inadvisable as it may cause display problems. If\n"
             << "you REALLY want to taste LLVM bitcode first-hand, you\n"
             << "can force output with the `-f' option.\n\n";
    }
    return true;
  }
  return false;
}

/// FindExecutable - Find a named executable, giving the argv[0] of program
/// being executed. This allows us to find another LLVM tool if it is built in
/// the same directory.  If the executable cannot be found, return an
/// empty string.
/// @brief Find a named executable.
#undef FindExecutable   // needed on windows :(
sys::Path llvm::FindExecutable(const std::string &ExeName,
                               const char *Argv0, void *MainAddr) {
  // Check the directory that the calling program is in.  We can do
  // this if ProgramPath contains at least one / character, indicating that it
  // is a relative path to the executable itself.
  sys::Path Result = sys::Path::GetMainExecutable(Argv0, MainAddr);
  Result.eraseComponent();
  if (!Result.isEmpty()) {
    Result.appendComponent(ExeName);
    if (Result.canExecute())
      return Result;
    // If the path is absolute (and it usually is), call FindProgramByName to
    // allow it to try platform-specific logic, such as appending a .exe suffix
    // on Windows. Don't do this if we somehow have a relative path, because
    // we don't want to go searching the PATH and accidentally find an unrelated
    // version of the program.
    if (Result.isAbsolute()) {
      Result = sys::Program::FindProgramByName(Result.str());
      if (!Result.empty())
        return Result;
    }
  }

  return sys::Path();
}
