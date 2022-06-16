// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 5

package Foo api;

choice Choice { Alternative(i32) }
class Class { fn F(n: i32) -> i32 { return n + 1; } }
interface Interface { fn G[me: Self]() -> Self; }
interface AnotherInterface {}

impl i32 as Interface { fn G[me: i32]() -> i32 { return me + 1; } }
impl i32 as AnotherInterface {}

fn GetStruct() -> Type { return {.n: i32}; }
fn GetChoice() -> Type { return Choice; }
fn GetClass() -> Type { return Class; }
fn GetInterface() -> Type { return Interface; }
fn GetConstraint() -> Type { return Interface & AnotherInterface; }

fn Main() -> i32 {
  var s: GetStruct() = {.n = 1};

  var n: i32 = s.(GetStruct().n);
  var c: GetChoice() = GetChoice().Alternative(n + 1);
  match (c) {
    case GetChoice().Alternative(m: i32) => {
      return GetClass().F(m).(GetInterface().G)().(GetConstraint().G)();
    }
  }
  // TODO: The above pattern is exhaustive; this should not be necessary.
  return -1;
}
