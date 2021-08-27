// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_AST_H_
#define EXECUTABLE_SEMANTICS_AST_AST_H_

#include <list>

#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/ast/library.h"
#include "executable_semantics/common/ptr.h"

namespace Carbon {

struct AST {
  Library package;
  std::vector<Library> imports;
  std::list<Ptr<const Declaration>> declarations;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_AST_H_
