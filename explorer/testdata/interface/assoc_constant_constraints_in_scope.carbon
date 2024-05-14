// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface A(T:! type) { let AResult:! type; }
interface B(T:! type) { let BResult:! A(T); }
interface I(T:! type) {
  let X:! B(T);
  // The constraints introduced by X should be in scope here.
  let Y:! X.BResult.AResult;
}

class CA {
  extend impl as A(i32) where .AResult = i32 {}
}
class CB {
  extend impl as B(i32) where .BResult = CA {}
}
class CI {
  extend impl as I(i32) where .X = CB and .Y = 5 {}
}

fn Main() -> i32 {
  var v: CI = {};
  return v.(I(i32).Y);
}

// CHECK:STDOUT: result: 5
