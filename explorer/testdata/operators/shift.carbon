// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 0

package ExplorerTest api;

fn Main() -> i32 {
  if (not (1 << 0 == 1)) { return 1; }
  if (not (1 << 3 == 8)) { return 2; }
  if (not (0 << 3 == 0)) { return 3; }
  if (not (3 << 1 == 6)) { return 4; }
  if (not (-1 << 2 == -4)) { return 5; }
  if (not (1 >> 0 == 1)) { return 6; }
  if (not (1 >> 1 == 0)) { return 7; }
  if (not (3 >> 1 == 1)) { return 8; }
  if (not (-1 >> 1 == -1)) { return 9; }
  if (not (-2 >> 1 == -1)) { return 10; }

  return 0;
}
