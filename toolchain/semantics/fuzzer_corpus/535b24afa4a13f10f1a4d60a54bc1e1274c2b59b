// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class Point {
  var x: i32;
  var y: i32;
}

fn Main() -> i32 {
  var p: Point;
  if (1 == 0) {
    p = {.x = 0, .y = 0};
  }
  // CHECK:STDERR: RUNTIME ERROR: fail_store_to_uninitialized_class.carbon:[[@LINE+1]]: undefined behavior: store to subobject of uninitialized value Uninit<Placeholder<p>>
  p.x = 1;
  p.y = 2;
  return p.x;
}
