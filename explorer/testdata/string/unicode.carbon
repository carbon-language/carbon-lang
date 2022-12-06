// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

fn CompareStr(s: String) -> i32 {
  if (s == "HELLO WORLD!!") {
    return 0;
  }
  return 1;
}

fn Main() -> i32 {
  return CompareStr("\u{0048}\u{0045}\u{004C}\u{004C}\u{004F} \u{0057}\u{004F}\u{0052}\u{004C}\u{0044}\u{0021}\u{21}");
}
