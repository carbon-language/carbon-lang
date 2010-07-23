//===--- Error.h - The LLVM Compiler Driver ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Error handling.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INCLUDE_COMPILER_DRIVER_ERROR_H
#define LLVM_INCLUDE_COMPILER_DRIVER_ERROR_H

#include "llvm/Support/raw_ostream.h"

#include <string>

namespace llvmc {

  inline void PrintError(const char* Err) {
    extern const char* ProgramName;
    llvm::errs() << ProgramName << ": " << Err << '\n';
  }

  inline void PrintError(const std::string& Err) {
    PrintError(Err.c_str());
  }
}

#endif // LLVM_INCLUDE_COMPILER_DRIVER_ERROR_H
