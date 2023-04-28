// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 12

package ExplorerTest api;

fn Main() -> i32 {
  var cond: if true then bool else i32 = true;
  if (if cond then true else false) {}
  while (if cond then false else true) {}
  return if if cond then true or false else false and true
         then if not cond then 1 + 2 else 3 * 4
         else if not cond then 5 + 6 else 7 * 8;
}
