// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: Printing 1.10
// CHECK:STDOUT: Printing 1.01
// CHECK:STDOUT: result: 0

package ExplorerTest api;

fn Main() -> i32 {
  Print("Printing {0}", 1.1);
  Print("Printing {0}", 1.01);
  return 0;
}
