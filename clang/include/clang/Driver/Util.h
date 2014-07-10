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

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {
  class Triple;
}

namespace clang {
class DiagnosticsEngine;

namespace driver {
  class Action;
  class JobAction;

  /// ArgStringMap - Type used to map a JobAction to its result file.
  typedef llvm::DenseMap<const JobAction*, const char*> ArgStringMap;

  /// ActionList - Type used for lists of actions.
  typedef SmallVector<Action*, 3> ActionList;

/// Get the (LLVM) name of the minimum ARM CPU for the arch we are targeting.
const char* getARMCPUForMArch(StringRef MArch, const llvm::Triple &Triple);

} // end namespace driver
} // end namespace clang

#endif
