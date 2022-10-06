// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

interface A(T:! Type) {}
interface B {}

// CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/impl/fail_nondeducible_parameter.carbon:[[@LINE+1]]: parameter `T:! Type` is not deducible from `impl i32 as interface B`
external impl forall [T:! Type] i32 as A(T) & B {}

fn Main() -> i32 { return 0; }
