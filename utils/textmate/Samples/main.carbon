// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// More single line comments

package Carbon api;

interface HasValueParam(T:! Type, V:! T) {
  fn Go[me: Self]() -> T;
}

impl () as HasValueParam(i32, 5) {
  fn Go[me: Self]() -> i32 { return 42; }
}

class Point {
  fn Origin() -> Self {
    return {.x = 0, .y = 0};
  }

  var x: i32;
  var y: i32;
}

fn Procedure() -> i32 {
  returned var zoop: i32 = 0;
  while (DoSomeJob() {
    zoop += 1;
  }

  return var;
}

fn Main() -> i32 {
  let str = "Hello world";
  let hex = 0xABCDEF1234567890;
  let bin = 0b0000111001010010;
  let dec = 123456789012345678;

  let big: Carbon.Int(1024) = 1234;
  let aaa: auto = "Carbon";
  let view: StringView = "Carbon";

  return Procedure();
}
