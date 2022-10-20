// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}

package ExplorerTest api;

interface A(T:! Type) { let AResult:! Type; }
interface B(T:! Type) { let BResult:! A(T); }
interface I(T:! Type) {
  let X:! B(T);
  // The constraints introduced by X should be in scope here.
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/interface/assoc_constant_constraints_in_scope.carbon:[[@LINE+1]]: could not find implementation of interface B(T = T) for (Self).(I(T = T).X)
  let Y:! X.BResult.AResult;
}

class CA {
  impl as A(i32) where .AResult = i32 {}
}
class CB {
  impl as B(i32) where .BResult = CA {}
}
class CI {
  impl as I(i32) where .X = CB and .Y = 5 {}
}

fn Main() -> i32 {
  var v: CI = {};
  return v.(I(i32).Y);
}
