//===-- Exceptions.h - Helpers for processing C++ exceptions --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_FLO_EXCEPTIONS_H
#define LLVM_TOOLS_LLVM_FLO_EXCEPTIONS_H

#include "BinaryContext.h"
#include "llvm/ADT/ArrayRef.h"

namespace llvm {
namespace flo {

void readLSDA(ArrayRef<uint8_t> LSDAData, BinaryContext &BC);

} // namespace flo
} // namespace llvm

#endif
