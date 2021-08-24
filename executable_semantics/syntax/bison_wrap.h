// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_SYNTAX_BISON_WRAP_H_
#define EXECUTABLE_SEMANTICS_SYNTAX_BISON_WRAP_H_

#include <optional>

#include "common/check.h"

namespace Carbon {

// Bison requires that types be default initializable for use with its variant
// semantics. This wraps arbitrary types to support a default constructor, while
// still requiring they be properly initialized.
template <typename T>
class BisonWrap {
 public:
  // Assigning a value initializes the wrapper.
  BisonWrap& operator=(T&& rhs) {
    val = std::move(rhs);
    return *this;
  }

  // Support transparent conversion to the wrapped type, erroring if not
  // initialized.
  operator T() {
    CHECK(val.has_value());
    T ret = std::move(*val);
    val.reset();
    return ret;
  }

 private:
  std::optional<T> val;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_SYNTAX_BISON_WRAP_H_
