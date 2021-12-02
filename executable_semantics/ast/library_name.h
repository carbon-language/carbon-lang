// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_LIBRARY_NAME_H_
#define EXECUTABLE_SEMANTICS_AST_LIBRARY_NAME_H_

#include <string>

namespace Carbon {

// Identifies a particular library. For example, "Geometry//Objects/FourSides"
// will have package="Geometry" and path="Objects/FourSides".
struct LibraryName {
  // The library's package.
  std::string package;

  // The package-relative path of the library. This defaults to the empty
  // string.
  std::string path;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_LIBRARY_NAME_H_
