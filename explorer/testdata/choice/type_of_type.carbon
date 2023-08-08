// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

choice C {}

fn Main() -> i32 {
  var a: auto = C;
  // OK: type of `C` is `type`.
  a = i32;
  return 0;
}

// CHECK:STDOUT: result: 0
