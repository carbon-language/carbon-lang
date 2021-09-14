// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_COMMON_PTR_ARRAY_REF_H_
#define EXECUTABLE_SEMANTICS_COMMON_PTR_ARRAY_REF_H_

#include <vector>

#include "executable_semantics/common/ptr.h"

namespace Carbon {

// Provides a constant reference to an underlying vector.
template <typename T>
class PtrArrayRef {
 public:
  explicit PtrArrayRef(const std::vector<Ptr<T>>* v) : v(v) {}
  auto operator[](int i) -> Ptr<const T> { return (*v)[i]; }

 private:
  Ptr<const std::vector<Ptr<T>>> v;
};

}  // namespace Carbon

#endif  // #define EXECUTABLE_SEMANTICS_COMMON_PTR_ARRAY_REF_H_
