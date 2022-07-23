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

class C {
  fn F() {}
  fn G[me: Self]() {}
}

fn H[T:! Type](x: T) {}

fn Main() -> i32 {
  H(C.F);
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_method_deduced.carbon:[[@LINE+1]]: Expected a type, but got member name G
  H(C.G);
  return 0;
}
