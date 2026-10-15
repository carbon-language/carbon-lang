// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Cpp library "<functional>";

class C {
  fn F(unused self: Self, i: i32) -> i32 {
    return i + 1;
  }
}

fn Run() -> i32 {
  var c: C = {};
  var n: i32 = Cpp.std.invoke(Cpp.std.bind(C.F, ref c, 1));
  return if n == 2 then 0 else 1;
}
