// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// NOAUTOUPDATE
// RUN: %{not} %{explorer-run}
// CHECK:STDERR: RUNTIME ERROR: overflow:1: Possible infinite loop: too many interpreter steps executed

package ExplorerTest impl;

fn Main() -> i32 {
  while (true) { }
  return 0;
}
