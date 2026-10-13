// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_SET_AND_RESTORE_H_
#define CARBON_COMMON_SET_AND_RESTORE_H_

#include "common/check.h"

namespace Carbon {

// Sets a variable to a new value, and restores the original value when
// the SetAndRestore is destroyed. CHECKs that the variable is still holding the
// new value when restoring the original value, to catch any data loss.
//
// The value must be copyable in order to compare with the new value at the time
// the SetAndRestore is destroyed.
template <typename T>
class [[nodiscard]] SetAndRestore {
 public:
  SetAndRestore(T& var, const T& new_value)
      : var_(&var),
        new_value_(new_value),
        orig_value_(std::exchange(var, new_value_)) {}

  ~SetAndRestore() {
    CARBON_CHECK(*var_ == new_value_);
    *var_ = std::move(orig_value_);
  }

 private:
  T* var_;
  T new_value_;
  T orig_value_;
};

template <typename T>
SetAndRestore(T&, const T&) -> SetAndRestore<T>;
template <typename T>
SetAndRestore(T&, T&&) -> SetAndRestore<T>;

}  // namespace Carbon

#endif  // CARBON_COMMON_SET_AND_RESTORE_H_
