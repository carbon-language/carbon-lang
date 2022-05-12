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

fn NoReturn() -> auto {
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/function/auto_return/fail_no_return.carbon:[[@LINE+1]]: control-flow reaches end of function that provides a `->` return type without reaching a return statement
}

fn Main() {
  NoReturn();
}
