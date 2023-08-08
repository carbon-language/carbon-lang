// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package Foo api;
fn Main() -> i32 {
  return {.m = 1, .n = 2}.({.n: i32, .m: i32}.n);
}

// CHECK:STDOUT: result: 2
