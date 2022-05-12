// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 7

package ExplorerTest api;

alias TypeAlias = i32;

fn Function(a: i32, b: TypeAlias) -> i32 { return a + b; }
fn GenericFunction[T:! Type](x: T) -> T { return x; }

alias FunctionAlias = Function;
alias GenericFunctionAlias = GenericFunction;

fn CheckFunctionAliases() -> i32 {
  return FunctionAlias(1, 2) + GenericFunctionAlias(4);
}


class Class {}
class GenericClass(T:! Type) {}

alias ClassAlias = Class;
alias GenericClassAlias = GenericClass;
alias ClassSpecializationAlias = GenericClassAlias(i32);

fn CheckClassAliases() {
  var a: Class = {};
  var b: ClassAlias = a;

  var c: GenericClass(i32) = {};
  var d: GenericClassAlias(i32) = c;
  var e: ClassSpecializationAlias = c;
}


interface Interface { fn Make() -> Self; }
interface GenericInterface(T:! Type) { fn Make() -> (Self, T); }

alias InterfaceAlias = Interface;
alias GenericInterfaceAlias = GenericInterface;
alias InterfaceSpecializationAlias = GenericInterfaceAlias(i32);

impl ClassAlias as InterfaceAlias { fn Make() -> Self { return {}; } }
impl ClassSpecializationAlias as InterfaceSpecializationAlias {
  fn Make() -> (Self, i32) { return ({}, 1); }
}
impl GenericClassAlias({.a: i32}) as GenericInterfaceAlias({.a: i32}) {
  fn Make() -> (Self, {.a: i32}) { return ({}, {.a = 0}); }
}

fn CheckImplementsInterface[T:! Interface](x: T) {}
fn CheckImplementsGenericInterface_i32[T:! GenericInterface(i32)](x: T) {}
fn CheckImplementsGenericInterface_struct[T:! GenericInterface({.a: i32})](x: T) {}

fn CheckInterfaceAliases() {
  var a: Class = {};
  CheckImplementsInterface(a);

  var b: GenericClass(i32) = {};
  CheckImplementsGenericInterface_i32(b);

  var c: GenericClass({.a: i32}) = {};
  CheckImplementsGenericInterface_struct(c);
}


fn Main() -> i32 {
  return CheckFunctionAliases();
}
