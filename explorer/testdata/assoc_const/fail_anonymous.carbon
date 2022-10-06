// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

interface Iface {
  // CHECK:SYNTAX ERROR: {{.*}}/explorer/testdata/assoc_const/fail_anonymous.carbon:[[@LINE+1]]: syntax error, unexpected UNDERSCORE, expecting identifier
  let _:! Type;
}

fn Main() -> i32 { return 0; }
