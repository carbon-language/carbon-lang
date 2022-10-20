// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDOUT: {
// CHECK:STDOUT:   Function(
// CHECK:STDOUT:       %4,
// CHECK:STDOUT:       {
// CHECK:STDOUT:         IntegerLiteral(%1, 12),
// CHECK:STDOUT:         IntegerLiteral(%2, 34),
// CHECK:STDOUT:         BinaryOperator(%3, +, %1, %2),
// CHECK:STDOUT:         Return(%3),
// CHECK:STDOUT:       }),
// CHECK:STDOUT:   SetName(`Main`, %4),
// CHECK:STDOUT: }

fn Main() {
  return 12 + 34;
}
