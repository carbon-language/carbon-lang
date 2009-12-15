//===-------------- TypeOrdering.h - Total ordering for types -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file provides a function objects and specializations that
//  allow QualType values to be sorted, used in std::maps, std::sets,
//  llvm::DenseMaps, and llvm::DenseSets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TYPE_ORDERING_H
#define LLVM_CLANG_TYPE_ORDERING_H

#include "clang/AST/Type.h"
#include <functional>

namespace clang {

/// QualTypeOrdering - Function object that provides a total ordering
/// on QualType values.
struct QualTypeOrdering : std::binary_function<QualType, QualType, bool> {
  bool operator()(QualType T1, QualType T2) const {
    return std::less<void*>()(T1.getAsOpaquePtr(), T2.getAsOpaquePtr());
  }
};

}

namespace llvm {
  template<class> struct DenseMapInfo;

  template<> struct DenseMapInfo<clang::QualType> {
    static inline clang::QualType getEmptyKey() { return clang::QualType(); }

    static inline clang::QualType getTombstoneKey() {
      using clang::QualType;
      return QualType::getFromOpaquePtr(reinterpret_cast<clang::Type *>(-1));
    }

    static unsigned getHashValue(clang::QualType Val) {
      return (unsigned)((uintptr_t)Val.getAsOpaquePtr()) ^
            ((unsigned)((uintptr_t)Val.getAsOpaquePtr() >> 9));
    }

    static bool isEqual(clang::QualType LHS, clang::QualType RHS) {
      return LHS == RHS;
    }
  };

  // FIXME: Move to Type.h
  template <>
  struct isPodLike<clang::QualType> { static const bool value = true; };
}

#endif
