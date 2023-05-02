// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDERR: RUNTIME ERROR: overflow:1: stack overflow: too many interpreter actions on stack

package EmptyIdentifier impl;

fn A() {
  A();
}

fn Main() -> i32 {
  A();
  return 0;
}
