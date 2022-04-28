// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{executable_semantics} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s

package ExecutableSemanticsTest api;

// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/impl/impl_as.carbon:[[@LINE+1]]: expected constraint after `as`, found value of type i32
external impl i32 as i32 {}

fn Main() -> i32 {
}
