// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: 8
// CHECK:STDOUT: 4
// CHECK:STDOUT: result: 0

package ExplorerTest api;

fn Main() -> i32 {
  if (1 << 0 != 1) { return 1; }
  if (1 << 3 != 8) { return 2; }
  if (0 << 3 != 0) { return 3; }
  if (3 << 1 != 6) { return 4; }
  if (-1 << 2 != -4) { return 5; }
  if (1 >> 0 != 1) { return 6; }
  if (1 >> 1 != 0) { return 7; }
  if (3 >> 1 != 1) { return 8; }
  if (-1 >> 1 != -1) { return 9; }
  if (-2 >> 1 != -1) { return 10; }

  var n: i32 = 1;
  n <<= 3;
  Print("{0}", n);
  n >>= 1;
  Print("{0}", n);

  return 0;
}
