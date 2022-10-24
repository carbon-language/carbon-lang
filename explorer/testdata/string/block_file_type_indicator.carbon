// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

fn Main() -> i32 {
  var s: String = """filetype
    A "block" ""string"" literal
      with file type indicator.
    """;
  if (s == "A \"block\" \"\"string\"\" literal\n  with file type indicator.\n") {
    return 0;
  } else {
    return 1;
  }
}
