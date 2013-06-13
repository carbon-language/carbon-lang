//===- not.cpp - The 'not' testing tool -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

int main(int argc, const char **argv) {
  std::string Program = sys::FindProgramByName(argv[1]);

  std::string ErrMsg;
  int Result =
      sys::ExecuteAndWait(sys::Path(Program), argv + 1, 0, 0, 0, 0, &ErrMsg);
  if (Result < 0) {
    errs() << "Error: " << ErrMsg << "\n";
    return 1;
  }

  return Result == 0;
}
