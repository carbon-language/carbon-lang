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

class X(T:! Type) {}

// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/constraint/fail_nontype_self.carbon:[[@LINE+1]]: `.Self` used in type of non-type binding `T`
fn F[T:! X(.Self)](x: T) {}

fn Main() -> i32 { return 0; }
