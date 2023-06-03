// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: 1
// CHECK:STDOUT: 2
// CHECK:STDOUT: 3
// CHECK:STDOUT: 4
// CHECK:STDOUT: 4
// CHECK:STDOUT: 6
// CHECK:STDOUT: 5
// CHECK:STDOUT: 6
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class A(T:! type, U:! type) {}
interface B(T:! type, U:! type) { fn F(); }

external impl forall [T:! type] A(T, i32) as B(i32, i32) {
  fn F() { Print("1"); }
}
external impl forall [T:! type] A(i32, T) as B(i32, i32) {
  fn F() { Print("2"); }
}
// Intentionally out of order so that explorer can't get the right answer by
// chance, by just picking the first or last matching impl.
external impl forall [T:! type] A(i32, i32) as B(i32, T) {
  fn F() { Print("4"); }
}
external impl forall [T:! type] A(i32, i32) as B(T, i32) {
  fn F() { Print("3"); }
}

external impl forall [T:! type, U:! type] A(T, i32*) as B(i32*, U) {
  fn F() { Print("5"); }
}
external impl forall [T:! type, U:! type] A(i32*, T) as B(U, i32*) {
  fn F() { Print("6"); }
}

fn Main() -> i32 {
  A((), i32).(B(i32, i32).F)();
  A(i32, ()).(B(i32, i32).F)();
  A(i32, i32).(B((), i32).F)();
  A(i32, i32).(B(i32, ()).F)();

  A(i32, i32).(B(i32, i32).F)();

  A(i32*, i32*).(B(i32, i32*).F)();
  A(i32*, i32*).(B(i32*, i32).F)();
  A(i32*, i32*).(B(i32*, i32*).F)();
  return 0;
}
