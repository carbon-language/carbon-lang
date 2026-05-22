// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

class Point {
  var x: i32;

  fn GetX(self: Point) -> i32 { return self.x; }
  fn GetY(self) -> i32 { return 0; }
  fn SetX(ref self, new_x: i32) {}
}
