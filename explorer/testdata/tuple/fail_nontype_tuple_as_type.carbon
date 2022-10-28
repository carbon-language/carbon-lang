// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/tuple/fail_nontype_tuple_as_type.carbon:[[@LINE+1]]: type error in type of name binding: '(i32, i32)' is not implicitly convertible to 'Type'
fn F[T:! (i32, i32)](x: T);

fn Main() -> i32;
