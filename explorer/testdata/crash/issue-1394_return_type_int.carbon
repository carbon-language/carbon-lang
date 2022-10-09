// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 2>&1 | %{FileCheck} %s
// AUTOUPDATE: %{explorer} %s

package EmptyIdentifier impl;

fn test() -> i32 {
  "a"["1"];
}
fn foo() -> test();

fn Main() -> i32 {
  return 0;
}
