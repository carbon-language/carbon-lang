// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_EXPRESSION_CATEGORY_H_
#define CARBON_EXPLORER_AST_EXPRESSION_CATEGORY_H_

#include "llvm/ADT/StringRef.h"

namespace Carbon {

// The category of a Carbon expression indicates whether it evaluates
// to a value, reference, or initialization.
enum class ExpressionCategory {
  // A "value expression" produces a value (with no associated location).
  Value,
  // A "reference expression" produces a location of an existing value.
  Reference,
  // An "initializing expression" takes a location and initialize it.
  Initializing,
};

auto ExpressionCategoryToString(ExpressionCategory cat) -> llvm::StringRef;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_EXPRESSION_CATEGORY_H_
