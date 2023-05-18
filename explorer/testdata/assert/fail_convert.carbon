// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDERR: RUNTIME ERROR: explorer/data/prelude.carbon:{{.*}}: "Fail"

package ExplorerTest api;

class ConvertTo(T:! type) {
  var v: T;
  impl as ImplicitAs(T) {
    fn Convert[self: Self]() -> T { return self.v; }
  }
}

fn Main() -> i32 {
  Assert({.v = false} as ConvertTo(bool), {.v = "Fail"} as ConvertTo(String));
  return 0;
}
