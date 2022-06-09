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

// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/constraint/fail_where_equals_different_types.carbon:[[@LINE+3]]: type error in values in `where ==` constraint
// CHECK: expected: i32
// CHECK: actual: Type
alias B = A where 4 == i32;

fn Main() -> i32 { return 0; }
