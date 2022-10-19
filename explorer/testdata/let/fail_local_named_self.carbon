// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main() -> i32 {
  // Error: Can't use keyword `Self` as the name of a local.
  // CHECK:STDERR: SYNTAX ERROR: {{.*}}/explorer/testdata/let/fail_local_named_self.carbon:[[@LINE+1]]: syntax error, unexpected COLON, expecting SLASH or binary *
  let Self: auto = 10;
  return 0;
}
