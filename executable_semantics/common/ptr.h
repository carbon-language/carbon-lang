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
  Ptr() = default;
  
  explicit Ptr(T* ptr) : ptr(ptr) { CHECK(ptr != nullptr); }

  template <typename OtherT,
            std::enable_if_t<std::is_convertible_v<OtherT*, T*>>* = nullptr>
  Ptr(Ptr<OtherT> other) {
    return Ptr<OtherT>(other.ptr);
  }

  Ptr(std::nullptr_t) = delete;

  Ptr(const Ptr& other) = default;
  Ptr& operator=(const Ptr& rhs) = default;

  auto operator*() const -> T& { return *ptr; }
  auto operator->() const -> T* { return ptr; }

  T* Get() const { return ptr; }

 private:
  T* ptr;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_COMMON_PTR_H_
