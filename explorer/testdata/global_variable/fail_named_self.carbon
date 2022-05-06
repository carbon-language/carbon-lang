// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

// Error: global variable may not be named with keyword `Self`.
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/global_variable/fail_named_self.carbon:[[@LINE+1]]: syntax error, unexpected SELF, expecting identifier
var Self: i32 = 0;

fn Main() -> i32 {
  return Self;
}
