// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 0

package ExplorerTest api;

class Class {}
class GenericClass(T:! Type) {}

alias ClassAlias = Class;
alias GenericClassAlias = GenericClass;
alias ClassSpecializationAlias = GenericClassAlias(i32);

fn Main() -> i32 {
  var a: Class = {};
  var b: ClassAlias = a;

  var c: GenericClass(i32) = {};
  var d: GenericClassAlias(i32) = c;
  var e: ClassSpecializationAlias = c;
  return 0;
}
