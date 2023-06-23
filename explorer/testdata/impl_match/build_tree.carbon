// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class S(T:! type) {}
class Z {}

class Tree(L:! type, R:! type) {}
class Leaf {}

interface Foo(T:! type) {}

impl forall [U:! type, T:! Foo(Tree(U, U))] S(T) as Foo(U) {}

alias Level1 = Tree(Leaf, Leaf);
alias Level2 = Tree(Level1, Level1);
alias Level3 = Tree(Level2, Level2);
alias Level4 = Tree(Level3, Level3);
alias Level5 = Tree(Level4, Level4);

impl Z as Foo(Level5) {}

fn F[T:! Foo(Leaf)](x: T) {}
fn Main() -> i32 {
  var n: S(S(S(S(S(Z))))) = {};
  F(n);
  return 0;
}
