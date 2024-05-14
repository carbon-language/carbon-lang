// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class C {
  destructor[self: Self] {
    Print("Destructor");
  }
}

fn Main() -> i32 {
  let c: C = {};
  return 0;
}

// CHECK:STDOUT: Destructor
// CHECK:STDOUT: result: 0
