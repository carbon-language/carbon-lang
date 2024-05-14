// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  var t: auto = 5;
  match (t) {
    case 5 => {
      return 0;
    }
    default => {
      return 1;
    }
  }
}

// CHECK:STDOUT: result: 0
