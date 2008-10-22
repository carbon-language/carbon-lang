//===-------------- TypeOrdering.h - Total ordering for types -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file provides a function object that gives a total ordering
//  on QualType values, so that they can be sorted, used in std::maps
//  and std::sets, and so on.
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
  bool operator()(QualType T1, QualType T2) {
    return std::less<void*>()(T1.getAsOpaquePtr(), T2.getAsOpaquePtr());
  }
};

}

#endif
