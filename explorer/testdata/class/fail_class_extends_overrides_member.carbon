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
  var var_c: i32;
}

class D extends C {
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_class_extends_overrides_member.carbon:[[@LINE+1]]: A member named `var_c` already exists in this class or its parents
  var var_c: i32;
}

fn Main() -> i32 {
  return 0;
}
