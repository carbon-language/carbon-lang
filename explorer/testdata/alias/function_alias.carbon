// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 7

package ExplorerTest api;

alias TypeAlias = i32;

fn Function(a: i32, b: TypeAlias) -> i32 { return a + b; }
fn GenericFunction[T:! type](x: T) -> T { return x; }

alias FunctionAlias = Function;
alias GenericFunctionAlias = GenericFunction;

fn Main() -> i32 {
  return FunctionAlias(1, 2) + GenericFunctionAlias(4);
}
