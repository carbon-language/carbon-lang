// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package ExplorerTest api;

fn Main() -> i32 {
  var a: f64 = 1.0;
  var b: f64 = 2.0;
  a += b;
  a -= b;
  a *= b;
  a /= b;

  var c: f64 = -a;

  var d: f64 = a + b;
  var e: f64 = a - b;
  var f: f64 = a * b;
  var g: f64 = a / b;

  var h: bool = a == b;
  var i: bool = a != b;
  var j: bool = a < b;
  var k: bool = a <= b;
  var l: bool = a > b;
  var m: bool = a >= b;

  return 0;
}
