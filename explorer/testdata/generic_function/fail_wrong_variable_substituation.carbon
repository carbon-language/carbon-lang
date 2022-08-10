// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s

package Foo api;

class X{
}

fn Get[T:! ()](n: T) -> i32 {  return 2; }

fn Main() -> i32 {
  var x : X = {};
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_function/fail_wrong_variable_substituation.carbon:[[@LINE+1]]: cannot convert deduced value class X for T to parameter type ()
  Get(x);
  return 0;
}
