// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{carbon} dump semantics-ir %s 2>&1 | %{FileCheckStrict} %s
// CHECK:{
// CHECK:  Function(
// CHECK:      %[[ID0:[0-9]+]],
// CHECK:      {
// CHECK:      }),
// CHECK:  SetName(`Foo`, %[[ID0]]),
// CHECK:}

fn Foo() {}
