// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_LIBRARY_H_
#define EXECUTABLE_SEMANTICS_AST_LIBRARY_H_

#include <string>

namespace Carbon {

struct Library {
  std::string package;
  std::optional<std::string> path;
  bool is_api;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_LIBRARY_H_
