// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_CHECKED_SAVE_AND_RESTORE_H_
#define CARBON_COMMON_CHECKED_SAVE_AND_RESTORE_H_

#include <utility>

#include "common/check.h"

namespace Carbon {

// Sets a variable to a new value, and restores the original value when
// the SetAndRestore is destroyed. CHECKs that the variable is still holding the
// new value when restoring the original value, to catch any data loss.
//
// The value must be copyable in order to compare with the new value at the time
// the SetAndRestore is destroyed.
template <typename T>
class [[nodiscard]] CheckedSaveAndRestore {
 public:
  CheckedSaveAndRestore(T& var, const T& new_value)
      : var_(&var),
        new_value_(new_value),
        orig_value_(std::exchange(var, new_value_)) {}

  ~CheckedSaveAndRestore() {
    CARBON_CHECK(*var_ == new_value_);
    *var_ = std::move(orig_value_);
  }

  CheckedSaveAndRestore(const CheckedSaveAndRestore&) = delete;
  auto operator=(const CheckedSaveAndRestore&)
      -> CheckedSaveAndRestore& = delete;

 private:
  T* var_;
  T new_value_;
  T orig_value_;
};

template <typename T>
CheckedSaveAndRestore(T&, const T&) -> CheckedSaveAndRestore<T>;
template <typename T>
CheckedSaveAndRestore(T&, T&&) -> CheckedSaveAndRestore<T>;

}  // namespace Carbon

#endif  // CARBON_COMMON_CHECKED_SAVE_AND_RESTORE_H_
