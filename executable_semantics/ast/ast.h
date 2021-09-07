// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_AST_H_
#define EXECUTABLE_SEMANTICS_AST_AST_H_

#include <vector>

#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/ast/library_name.h"
#include "executable_semantics/common/ptr.h"

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
  std::vector<Ptr<const Declaration>> declarations;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_AST_H_
