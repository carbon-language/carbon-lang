// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn F() -> () {
  return ();
}

fn Main() -> i32 {
  F();
  return 0;
}

// CHECK:STDOUT: result: 0
