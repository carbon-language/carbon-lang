//===-- llvm/Instrinsics.h - LLVM Intrinsic Function Handling ---*- C++ -*-===//
//
// This file defines a set of enums which allow processing of intrinsic
// functions.  Values of these enum types are returned by
// Function::getIntrinsicID.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INTRINSICS_H
#define LLVM_INTRINSICS_H

namespace LLVMIntrinsic {
  enum ID {
    not_intrinsic = 0,   // Must be zero
    va_start,            // Used to represent a va_start call in C
    va_end,              // Used to represent a va_end call in C
    va_copy,             // Used to represent a va_copy call in C
  };
}

#endif
