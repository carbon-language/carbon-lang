// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDERR: RUNTIME ERROR: explorer/data/prelude.carbon:{{.*}}: Rand inputs must be ordered for a non-empty range: 0 must be less than 0

package ExplorerTest api;

fn Main() -> i32 {
  Rand(0, 0);
  return 0;
}
