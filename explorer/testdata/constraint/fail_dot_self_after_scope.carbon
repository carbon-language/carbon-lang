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

interface FrobWith(T:! Type) {}

fn F[T:! FrobWith(.Self)]
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/constraint/fail_dot_self_after_scope.carbon:[[@LINE+1]]: could not resolve '.Self'
  (U: .Self) {
}

fn Main() -> i32 {
  return 0;
}
