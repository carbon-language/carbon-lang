// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

choice Ch {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/basic_syntax/fail_alternative_uses_choice.carbon:[[@LINE+1]]: 'Ch' is not usable until after it has been completely declared
  Opt(Ch)
}

fn Main() -> i32 {
  return 0;
}
