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

#ifndef SYSTEMUTILS_H
#define SYSTEMUTILS_H

#include <string>

namespace llvm {

/// isExecutableFile - This function returns true if the filename specified
/// exists and is executable.
///
bool isExecutableFile(const std::string &ExeFileName);

/// isStandardOutAConsole - Return true if we can tell that the standard output
/// stream goes to a terminal window or console.
bool isStandardOutAConsole();

/// FindExecutable - Find a named executable, giving the argv[0] of program
/// being executed. This allows us to find another LLVM tool if it is built into
/// the same directory, but that directory is neither the current directory, nor
/// in the PATH.  If the executable cannot be found, return an empty string.
/// 
std::string FindExecutable(const std::string &ExeName,
                           const std::string &ProgramPath);

/// RunProgramWithTimeout - This function executes the specified program, with
/// the specified null-terminated argument array, with the stdin/out/err fd's
/// redirected, with a timeout specified on the commandline.  This terminates
/// the calling program if there is an error executing the specified program.
/// It returns the return value of the program, or -1 if a timeout is detected.
///
int RunProgramWithTimeout(const std::string &ProgramPath, const char **Args,
                          const std::string &StdInFile = "",
                          const std::string &StdOutFile = "",
                          const std::string &StdErrFile = "");

/// ExecWait - Execute a program with the given arguments and environment and 
/// wait for it to terminate.
///
int ExecWait (const char * const argv[], const char * const envp[]);

/// AllocateRWXMemory - Allocate a slab of memory with read/write/execute
/// permissions.  This is typically used for JIT applications where we want
/// to emit code to the memory then jump to it.  Getting this type of memory
/// is very OS specific.
///
void *AllocateRWXMemory(unsigned NumBytes);

} // End llvm namespace

#endif
