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

class HasI32Param(V:! i32) {}

fn F(x: HasI32Param(1)) {}

fn Main() -> i32 {
  var n: HasI32Param(0) = {};
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/fail_value_param_mismatch.carbon:[[@LINE+1]]: mismatch in non-type values, `0` != `1`
  F(n);
  return 0;
}
