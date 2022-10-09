// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

class C {
  // CHECK:SYNTAX ERROR: {{.*}}/explorer/testdata/addr/fail_method_me_misspelled.carbon:[[@LINE+1]]: illegal binding pattern in implicit parameter list
  fn F[addr mew: Self*]() {}
}

fn Main() -> i32 { return 0; }
