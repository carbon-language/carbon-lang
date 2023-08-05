// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 25

package ExplorerTest api;

class List {
  choice Node {
    Nil,
    Cons(i32, Self*)
  }
  var node: Node;

  // Creates a list of `Cons(n, Cons(n - 1, ... Cons(1, Nil) ... ))`.
  fn Make(n: i32) -> Self {
    return {
      .node = if n == 0 then Node.Nil else Node.Cons(n, heap.New(Make(n - 1)))
    };
  }

  // Returns the sum of values in the list plus the value of `a`.
  fn Sum[self: Self](a: i32) -> i32 {
    match (self.node) {
      case Node.Nil => { return a; }
      case Node.Cons(b: i32, rest: Self*) => { return rest->Sum(a + b); }
    }
  }
}

fn Main() -> i32 {
  var l: List = List.Make(5);
  return l.Sum(10);
}
