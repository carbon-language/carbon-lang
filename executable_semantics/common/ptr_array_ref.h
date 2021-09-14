// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_COMMON_PTR_ARRAY_REF_H_
#define EXECUTABLE_SEMANTICS_COMMON_PTR_ARRAY_REF_H_

#include <vector>

#include "executable_semantics/common/ptr.h"
#include "llvm/ADT/ArrayRef.h"

namespace Carbon {

// Provides a constant reference to an underlying vector.
template <typename T>
class PtrArrayRef {
 public:
  explicit PtrArrayRef(llvm::ArrayRef<Ptr<T>> ref) : ref(ref) {}

  auto operator[](int i) -> Ptr<const T> { return ref[i]; }

 private:
  llvm::ArrayRef<Ptr<T>> ref;
};

}  // namespace Carbon

#endif  // #define EXECUTABLE_SEMANTICS_COMMON_PTR_ARRAY_REF_H_
