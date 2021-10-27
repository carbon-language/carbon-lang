// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/ast.h"

namespace Carbon {

void AST::Print(llvm::raw_ostream& out) const {
  out << "package " << package;
  if (is_api) {
    out << "api";
  }
  out << ";\n";

  for (const LibraryName& import : imports) {
    out << "import " << import << ";\n";
  }

  for (Nonnull<const Declaration*> declaration : declarations) {
    out << *declaration << "\n";
  }
}

}  // namespace Carbon
