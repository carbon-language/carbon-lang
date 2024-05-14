// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class Foo {
    fn Bar[self: Foo*]() {
        self->x += 1;
    }
    var x: i32;
}

fn Main() -> i32 {
  var foo: Foo = {.x = 0};
  // CHECK:STDERR: COMPILATION ERROR: fail_missing_addr.carbon:[[@LINE+1]]: method foo.Bar does not match the target function's self pattern (did you forget an `addr`?)
  foo.Bar();
  return 0;
}
