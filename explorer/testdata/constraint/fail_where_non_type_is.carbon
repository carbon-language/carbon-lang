// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

interface A {}

// CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/constraint/fail_where_non_type_is.carbon:[[@LINE+1]]: Expected a type, but got 4
alias B = A where 4 is A;

fn Main() -> i32 { return 0; }
