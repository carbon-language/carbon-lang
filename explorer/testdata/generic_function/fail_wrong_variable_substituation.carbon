// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package Foo api;

class X{
}

// CHECK:STDERR: COMPILATION ERROR: fail_wrong_variable_substituation.carbon:[[@LINE+1]]: value of generic binding T is not known
fn Get[T:! ()](n: T) -> i32 {  return 2; }

fn Main() -> i32 {
  var x : X = {};
  Get(x);
  return 0;
}
