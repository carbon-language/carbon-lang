//===--- Scope.h - Information about a semantic context -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines FunctionScopeInfo and its subclasses, which contain
// information about a single function, block, lambda, or method body.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_SEMA_SCOPEINFO_H
#define LLVM_FLANG_SEMA_SCOPEINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include <algorithm>

namespace flang {

namespace sema {

/// \brief Contains information about the PROGRAM statement currently
/// being parsed.
class ProgramScope {
public:
  ProgramScope();
  // TO BE IMPLEMETED
};

/// \brief Contains information about the PROGRAM statement currently
/// being parsed.
class FunctionScope {
public:
  FunctionScope();
  // TO BE IMPLEMETED
};

}  // end namespace sema
}  // end namespace flang

#endif
