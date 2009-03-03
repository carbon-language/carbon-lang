//===--- Util.h - Common Driver Utilities -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_UTIL_H_
#define CLANG_DRIVER_UTIL_H_

#include "llvm/ADT/SmallVector.h"

namespace clang {
namespace driver {
  /// ArgStringList - Type used for constructing argv lists for subprocesses.
  typedef llvm::SmallVector<const char*, 16> ArgStringList;

} // end namespace driver
} // end namespace clang

#endif
