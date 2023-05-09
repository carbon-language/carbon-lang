// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDERR: RUNTIME ERROR: explorer/data/prelude.carbon:{{.*}}: Integer overflow

package ExplorerTest api;

fn Main() -> i32 {
  var a: auto = 5 << 75;
  return 0;
}
