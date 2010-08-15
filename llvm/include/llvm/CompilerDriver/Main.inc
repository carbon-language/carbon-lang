//===--- Main.inc - The LLVM Compiler Driver --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Default main() for the driver executable.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INCLUDE_COMPILER_DRIVER_MAIN_INC
#define LLVM_INCLUDE_COMPILER_DRIVER_MAIN_INC

#include "llvm/CompilerDriver/Main.h"

int main(int argc, char** argv) {
  return llvmc::Main(argc, argv);
}

#endif // LLVM_INCLUDE_COMPILER_DRIVER_MAIN_INC
