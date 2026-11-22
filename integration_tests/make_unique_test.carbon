// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Cpp library "<memory>";

class C {
  fn C(i: i32) -> Self {
    return {.i = i};
  }

  var i: i32;
}

fn Run() -> i32 {
  var ptr: Cpp.std.unique_ptr(C) = Cpp.std.make_unique(C, 42);
  if (not ptr.get().HasValue()) {
    return 1;
  }
  let ref c: C = *ptr.get().Get();
  return if c.i == 42 then 0 else 1;
}
