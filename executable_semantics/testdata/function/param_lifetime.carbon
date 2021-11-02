// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: executable_semantics %s
// CHECK: result: 0

package ExecutableSemanticsTest api;

// This tests the call-by-value aspect of parameter passing.
// This makes sure that when the value in `x` dies,
// it does not cause the value in `a` to also die.

fn f(x: i32) -> i32 {
  return 0;
}

fn Main() -> i32 {
  var a: i32 = 0; var b: i32 = 1;
  f(a);
  b = a;
  return b;
}
