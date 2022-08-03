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

class A {
  fn F() {}
}

class B extends A {
  fn F() {}
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_class_extends.carbon:[[@LINE+1]]: Class extension with `extends` is not supported yet
}

fn Main() -> i32 {
  return 0;
}
