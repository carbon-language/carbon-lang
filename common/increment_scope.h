// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_INCREMENT_SCOPE_H_
#define CARBON_COMMON_INCREMENT_SCOPE_H_

#include <cstdint>

#include "common/check.h"

namespace Carbon {

// An RAII object that increments a value while it remains in scope,
// decrementing it when it is destroyed.
class [[nodiscard]] IncrementScope {
 public:
  explicit IncrementScope(int32_t& value [[clang::lifetimebound]])
      : value_(&value), before_(value) {
    (*value_)++;
  }
  ~IncrementScope() {
    (*value_)--;
    CARBON_CHECK(*value_ == before_,
                 "IncrementScope value not restored - incorrect bracketing of "
                 "increments and decrements?");
  }

 private:
  int32_t* value_;
  int32_t before_;
};

}  // namespace Carbon

#endif  // CARBON_COMMON_INCREMENT_SCOPE_H_
