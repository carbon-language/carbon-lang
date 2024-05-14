// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_SYNTAX_BISON_WRAP_H_
#define CARBON_EXPLORER_SYNTAX_BISON_WRAP_H_

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
  auto operator=(T&& rhs) -> BisonWrap& {
    val_ = std::move(rhs);
    return *this;
  }

  // Support transparent conversion to the wrapped type.
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator T() { return Release(); }

  // Access the contained object, which must already have been set.
  auto Unwrap() & -> T& { return *val_; }

  // Deliberately releases the contained value. Errors if not initialized.
  // Called directly in parser.ypp when releasing pairs.
  auto Release() -> T {
    CARBON_CHECK(val_.has_value());
    T ret = std::move(*val_);
    val_.reset();
    return ret;
  }

 private:
  std::optional<T> val_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_SYNTAX_BISON_WRAP_H_
