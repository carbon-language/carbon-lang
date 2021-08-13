// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_COMMON_PTR_H_
#define EXECUTABLE_SEMANTICS_COMMON_PTR_H_

#include <memory>
#include <vector>

#include "common/check.h"

namespace Carbon {

// A non-nullable pointer. Written as `Ptr<T>` instead of `T*`.
template <typename T>
class Ptr {
 public:
  explicit Ptr(T* ptr) : ptr(ptr) { CHECK(ptr != nullptr); }

  auto operator*() const -> T& { return *ptr; }
  auto operator->() const -> T* { return ptr; }

  template <typename OtherT>
  operator Ptr<OtherT>() const {
    return Ptr<OtherT>(ptr);
  }

 private:
  T* ptr;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_COMMON_PTR_H_
