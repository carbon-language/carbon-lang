// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface A(T:! Type) {}
interface B {}

// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/impl/fail_nondeducible_parameter.carbon:[[@LINE+1]]: parameter `T:! Type` is not deducible from `impl i32 as interface B`
external impl forall [T:! Type] i32 as A(T) & B {}

fn Main() -> i32 { return 0; }
