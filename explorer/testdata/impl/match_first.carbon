// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: XYZ: 1
// CHECK:STDOUT: XY: 1
// CHECK:STDOUT: XZ: 1
// CHECK:STDOUT: YZ: 2
// CHECK:STDOUT: JustX: 1
// CHECK:STDOUT: JustY: 2
// CHECK:STDOUT: JustZ: 3
// CHECK:STDOUT: None: 4
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface A {
  fn Which() -> i32;
}

interface X {}
interface Y {}
interface Z {}

__match_first {
  external impl forall [T:! X] T as A {
    fn Which() -> i32 { return 1; }
  }
  external impl forall [T:! Y] T as A {
    fn Which() -> i32 { return 2; }
  }
  external impl forall [T:! Z] T as A {
    fn Which() -> i32 { return 3; }
  }
  external impl forall [T:! type] T as A {
    fn Which() -> i32 { return 4; }
  }
}

class XYZ {}
class XY {}
class XZ {}
class YZ {}
class JustX {}
class JustY {}
class JustZ {}
class None {}

impl XYZ as X & Y & Z {}
impl XY as X & Y {}
impl XZ as X & Z {}
impl YZ as Y & Z {}
impl JustX as X {}
impl JustY as Y {}
impl JustZ as Z {}

fn Main() -> i32 {
  Print("XYZ: {0}", XYZ.(A.Which)());
  Print("XY: {0}", XY.(A.Which)());
  Print("XZ: {0}", XZ.(A.Which)());
  Print("YZ: {0}", YZ.(A.Which)());
  Print("JustX: {0}", JustX.(A.Which)());
  Print("JustY: {0}", JustY.(A.Which)());
  Print("JustZ: {0}", JustZ.(A.Which)());
  Print("None: {0}", None.(A.Which)());
  return 0;
}
