//===- SystemUtils.h - Utilities to do low-level system stuff ---*- C++ -*-===//
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

#ifndef LLVM_SUPPORT_SYSTEMUTILS_H
#define LLVM_SUPPORT_SYSTEMUTILS_H

#include "llvm/System/Program.h"

namespace llvm {

/// isStandardOutAConsole - Return true if we can tell that the standard output
/// stream goes to a terminal window or console.
bool isStandardOutAConsole();

/// FindExecutable - Find a named executable, giving the argv[0] of program
/// being executed. This allows us to find another LLVM tool if it is built into
/// the same directory, but that directory is neither the current directory, nor
/// in the PATH.  If the executable cannot be found, return an empty string.
/// @brief Find a named executable.
sys::Path FindExecutable(const std::string &ExeName,
                         const std::string &ProgramPath);

/// RunProgramWithTimeout - This function provides an alternate interface to the
/// sys::Program::ExecuteAndWait interface.
/// @see sys:Program::ExecuteAndWait
inline int RunProgramWithTimeout(const sys::Path &ProgramPath,
                                const char **Args,
                                const sys::Path &StdInFile,
                                const sys::Path &StdOutFile,
                                const sys::Path &StdErrFile,
                                unsigned NumSeconds = 0) {
  const sys::Path* redirects[3];
  redirects[0] = &StdInFile;
  redirects[1] = &StdOutFile;
  redirects[2] = &StdErrFile;
  
  return 
    sys::Program::ExecuteAndWait(ProgramPath, Args, 0, redirects, NumSeconds);
}

} // End llvm namespace

#endif
