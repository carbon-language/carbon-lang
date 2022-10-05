// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{carbon} dump semantics-ir %s 2>&1 | %{FileCheckStrict} %s
// CHECK:{
// CHECK:  Function(
// CHECK:      %[[ID0:[0-9]+]],
// CHECK:      {
// CHECK:        IntegerLiteral(%[[ID1:[0-9]+]], 12),
// CHECK:        IntegerLiteral(%[[ID2:[0-9]+]], 34),
// CHECK:        BinaryOperator(%[[ID3:[0-9]+]], +, %[[ID1]], %[[ID2]]),
// CHECK:        Return(%[[ID3]]),
// CHECK:      }),
// CHECK:  SetName(`Main`, %[[ID0]]),
// CHECK:}

fn Main() {
  return 12 + 34;
}
