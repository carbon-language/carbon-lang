// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface Z {
  fn Zero() -> Self;
}

class Point {
  var x: i32;
  var y: i32;
  // `Self` not allowed after `extend impl`.
  // CHECK:STDERR: SYNTAX ERROR: fail_extend_with_self.carbon:[[@LINE+1]]: syntax error, unexpected SELF, expecting AS
  extend impl Self as Z {
    fn Zero() -> Self {
      return {.x = 0, .y = 0};
    }
  }
}

fn Main() -> i32 {
  var a: Point = Point.Zero();
  return p.x;
}
