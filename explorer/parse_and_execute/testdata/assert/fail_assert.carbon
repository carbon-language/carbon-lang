// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDERR: RUNTIME ERROR: explorer/data/prelude.carbon:{{.*}}: "HALLO WELT"

package ExplorerTest api;

fn Main() -> i32 {
  Assert(false, "HALLO WELT");
  return 0;
}
