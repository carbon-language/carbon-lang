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

class C extends i32 {
  var x: i32;
  var y: i32;
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_extends_non_class.carbon:[[@LINE+1]]: Attempting to extend non-identifier expression `int type literal`
}

fn Main() -> i32 {
  return 0;
}
