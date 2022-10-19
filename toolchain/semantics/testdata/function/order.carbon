// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{carbon} dump semantics-ir %s 2>&1 | %{FileCheck-strict} %s
// CHECK:{
// CHECK:  Function(
// CHECK:      %[[ID0:[0-9]+]],
// CHECK:      {
// CHECK:      }),
// CHECK:  SetName(`Foo`, %[[ID0]]),
// CHECK:  Function(
// CHECK:      %[[ID1:[0-9]+]],
// CHECK:      {
// CHECK:      }),
// CHECK:  SetName(`Bar`, %[[ID1]]),
// CHECK:  Function(
// CHECK:      %[[ID2:[0-9]+]],
// CHECK:      {
// CHECK:      }),
// CHECK:  SetName(`Baz`, %[[ID2]]),
// CHECK:}

fn Foo() {}
fn Bar() {}
fn Baz() {}
