// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 12345

package ExplorerTest api;

class Class { var a: i32; }
class GenericClass(T:! Type) { var a: T; }
interface Interface { fn Make() -> Self; }
interface GenericInterface(T:! Type) { fn Make() -> (Self, T); }

alias ClassAlias = Class;
alias GenericClassAlias = GenericClass;
alias ClassSpecializationAlias = GenericClassAlias(i32);
alias InterfaceAlias = Interface;
alias GenericInterfaceAlias = GenericInterface;
alias InterfaceSpecializationAlias = GenericInterfaceAlias(i32);

impl ClassAlias as InterfaceAlias {
  fn Make() -> Self { return {.a = 1}; }
}
impl ClassSpecializationAlias as InterfaceSpecializationAlias {
  fn Make() -> (Self, i32) { return ({.a = 2}, 3); }
}
alias S = {.a: i32};
impl GenericClassAlias(S) as GenericInterfaceAlias(S) {
  fn Make() -> (Self, {.a: i32}) { return ({.a = {.a = 4}}, {.a = 5}); }
}

fn CheckImplementsInterface[T:! Interface](x: T) -> T {
  return T.Make();
}
fn CheckImplementsGenericInterface_i32
    [T:! GenericInterface(i32)](x: T) -> (T, i32) {
  return T.Make();
}
fn CheckImplementsGenericInterface_struct
    [T:! GenericInterface(S)](x: T) -> (T, S) {
  return T.Make();
}

fn Main() -> i32 {
  var a: Class = {.a = 0};
  a = CheckImplementsInterface(a);

  var b: GenericClass(i32) = {.a = 0};
  var (b2: GenericClass(i32), n: i32) =
    CheckImplementsGenericInterface_i32(b);

  var c: GenericClass({.a: i32}) = {.a = {.a = 0}};
  var (c2: GenericClass({.a: i32}), s: S) =
    CheckImplementsGenericInterface_struct(c);

  return 10000 * a.a + 1000 * b2.a + 100 * n + 10 * c2.a.a + s.a;
}
