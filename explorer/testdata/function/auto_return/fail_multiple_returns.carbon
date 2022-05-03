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

fn Add(x: i32, y: i32) -> auto {
  if (x == 0) {
    return x;
  } else if (y == 0) {
    // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/function/auto_return/fail_multiple_returns.carbon:[[@LINE+1]]: Only one return is allowed in a function with an `auto` return type.
    return y;
  } else {
    return x + y;
  }
}

fn Main() -> i32 {
  return Add(1, 2) - 3;
}
