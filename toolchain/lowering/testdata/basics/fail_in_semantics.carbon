// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This validates that earlier errors prevent lowering, without crashing.
// AUTOUPDATE
// RUN: %{not} %{carbon-run-lowering}

// CHECK:STDERR: {{.*}}/toolchain/lowering/testdata/basics/fail_in_semantics.carbon:[[@LINE+1]]:17: Type mismatch: lhs is node2, rhs is node3
var x: i32 = 1.0;
