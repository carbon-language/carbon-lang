// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_VALUE_CATEGORY_H_
#define EXECUTABLE_SEMANTICS_AST_VALUE_CATEGORY_H_

namespace Carbon {

// The value category of a Carbon expression indicates whether it evaluates
// to a variable or a value. A variable can be mutated, and can have its
// address taken, whereas a value cannot.
enum class ValueCategory {
  // A variable. This roughly corresponds to a C/C++ lvalue.
  Var,
  // A value. This roughly corresponds to a C/C++ rvalue.
  Let,
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_VALUE_CATEGORY_H_
