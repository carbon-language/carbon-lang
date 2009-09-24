//===- not.cpp - The 'not' testing tool -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/System/Path.h"
#include "llvm/System/Program.h"
using namespace llvm;

int main(int argc, const char **argv) {
  sys::Path Program = sys::Program::FindProgramByName(argv[1]);
  return !sys::Program::ExecuteAndWait(Program, argv + 1);
}
