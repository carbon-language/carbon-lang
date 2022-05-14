// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_AST_H_
#define CARBON_EXPLORER_AST_AST_H_

#include <vector>

#include "explorer/ast/declaration.h"
#include "explorer/ast/library_name.h"
#include "explorer/common/nonnull.h"

namespace Carbon {

// A Carbon file's AST.
struct AST {
  // The package directive's library.
  LibraryName package;
  // The package directive's API or impl state.
  bool is_api;
  // Import directives.
  std::vector<LibraryName> imports;
  // The file's ordered declarations.
  std::vector<Nonnull<Declaration*>> declarations;
  // Synthesized call to `Main`. Injected after parsing.
  std::optional<Nonnull<CallExpression*>> main_call;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_AST_H_
