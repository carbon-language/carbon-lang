//===--- Utility.cpp - The LLVM Compiler Driver -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Various helper and utility functions - implementation.
//
//===----------------------------------------------------------------------===//

#include "Utility.h"

#include "llvm/System/Program.h"

#include <stdexcept>

using namespace llvm;

int llvmcc::ExecuteProgram(const std::string& name,
                           const std::vector<std::string>& args) {
  sys::Path prog = sys::Program::FindProgramByName(name);

  if (prog.isEmpty())
    throw std::runtime_error("Can't find program '" + name + "'");
  if (!prog.canExecute())
    throw std::runtime_error("Program '" + name + "' is not executable.");

  // Invoke the program
  std::vector<const char*> argv((args.size()+2));
  argv[0] = name.c_str();
  for (unsigned i = 1; i <= args.size(); ++i)
    argv[i] = args[i-1].c_str();
  argv[args.size()+1] = 0;  // null terminate list.

  return sys::Program::ExecuteAndWait(prog, &argv[0]);
}
