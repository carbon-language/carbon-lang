//===--- Error.h - The LLVM Compiler Driver ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Exception classes for llvmc.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INCLUDE_COMPILER_DRIVER_ERROR_H
#define LLVM_INCLUDE_COMPILER_DRIVER_ERROR_H

#include <stdexcept>

namespace llvmc {

  /// error_code - This gets thrown during the compilation process if a tool
  /// invocation returns a non-zero exit code.
  class error_code: public std::runtime_error {
    int Code_;
  public:
    error_code (int c)
      : std::runtime_error("Tool returned error code"), Code_(c)
    {}

    int code() const { return Code_; }
  };

}

#endif // LLVM_INCLUDE_COMPILER_DRIVER_ERROR_H
