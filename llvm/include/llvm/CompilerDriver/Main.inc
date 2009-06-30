//===--- Main.inc - The LLVM Compiler Driver --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This tool provides a single point of access to the LLVM
//  compilation tools.  It has many options. To discover the options
//  supported please refer to the tools' manual page or run the tool
//  with the --help option.
//
//  This file provides the default entry point for the driver executable.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INCLUDE_COMPILER_DRIVER_MAIN_INC
#define LLVM_INCLUDE_COMPILER_DRIVER_MAIN_INC

#include "llvm/CompilerDriver/ForceLinkage.h"

namespace llvmc {
  int Main(int argc, char** argv);
}

int main(int argc, char** argv) {
  llvmc::ForceLinkage();
  return llvmc::Main(argc, argv);
}

#endif // LLVM_INCLUDE_COMPILER_DRIVER_MAIN_INC
