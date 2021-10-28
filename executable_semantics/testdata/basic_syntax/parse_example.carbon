// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: split-file %s %t
// RUN: executable_semantics --dump_ast %t/a.carbon 2>&1 > %t.a.ast
// RUN: executable_semantics --dump_ast %t/b.carbon 2>&1 | diff %t.a.ast -

//--- a.carbon

package ExecutableSemanticsTest api;

fn main() -> i32 {
  return a __unimplemented_example_infix b + c;
}

//--- b.carbon

package ExecutableSemanticsTest api;

fn main() -> i32 {
  return (a __unimplemented_example_infix b) + c;
}
