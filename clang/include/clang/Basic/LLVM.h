//===--- LLVM.h - Import various common LLVM datatypes ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file forward declares and imports various common LLVM datatypes that
// clang wants to use unqualified.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_BASIC_LLVM_H
#define CLANG_BASIC_LLVM_H

// This should be the only #include, force #includes of all the others on
// clients.
#include "llvm/Support/Casting.h"

namespace llvm {
  // ADT's.
  class StringRef;
  template<typename T, unsigned N> class SmallVector;
  template<typename T> class SmallVectorImpl;
  
  // TODO: Twine, raw_ostream, DenseMap, ...
}


namespace clang {
  // Casting operators.
  using llvm::isa;
  using llvm::cast;
  using llvm::dyn_cast;
  using llvm::dyn_cast_or_null;
  using llvm::cast_or_null;
  
  using llvm::StringRef;
  using llvm::SmallVector;
  using llvm::SmallVectorImpl;
  
} // end namespace clang.

#endif
