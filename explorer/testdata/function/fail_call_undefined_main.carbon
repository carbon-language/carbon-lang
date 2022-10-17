// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}
// CHECK:STDERR: RUNTIME ERROR: <Main()>:0: attempt to call function `Main` that has not been defined

package ExplorerTest api;

fn Main() -> i32;
