//===--- Sema.h - Semantic Analysis & AST Building --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Sema class, which performs semantic analysis
// for Fortran.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMA_H
#define LLVM_CLANG_SEMA_SEMA_H

#include "flang/Basic/Version.h"
#include "flang/Sema/Scope.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TinyPtrVector.h"
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
// Put here the required forward declarations for LLVM
class APSInt;
template<typename ValueT> struct DenseMapInfo;
template<typename ValueT, typename ValueInfoT> class DenseSet;
class SmallBitVector;
class InlineAsmIdentifierInfo;
}  // namespace llvm

namespace flang {

// Put here the required forward declarations for flang
class SourceLocation;

namespace sema {
// Put here the forward declarations of the inner classes of the Sema library
class ProgramScope;
class FunctionScope;
}  // namespace sema

/// Sema - This implements semantic analysis for Fortran
class Sema {
public:
  Sema();
};  // end class Sema

}  // end namespace flang

#endif
