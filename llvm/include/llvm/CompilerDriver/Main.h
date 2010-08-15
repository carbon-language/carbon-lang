//===--- Main.h - The LLVM Compiler Driver ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Entry point for the driver executable.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INCLUDE_COMPILER_DRIVER_MAIN_H
#define LLVM_INCLUDE_COMPILER_DRIVER_MAIN_H

namespace llvmc {
  int Main(int argc, char** argv);
}

#endif // LLVM_INCLUDE_COMPILER_DRIVER_MAIN_H
