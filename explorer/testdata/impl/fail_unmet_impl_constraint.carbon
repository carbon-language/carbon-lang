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

interface A {}
interface B {}

// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/impl/fail_unmet_impl_constraint.carbon:[[@LINE+1]]: could not find implementation of interface B for i32
external impl i32 as A where i32 is B {}

fn Main() -> i32 {
  return 0;
}
