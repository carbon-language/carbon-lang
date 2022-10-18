// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-semantics}
// CHECK:STDOUT: {
// CHECK:STDOUT:   Function(
// CHECK:STDOUT:       %1,
// CHECK:STDOUT:       {
// CHECK:STDOUT:         Return(None),
// CHECK:STDOUT:       }),
// CHECK:STDOUT:   SetName(`Main`, %1),
// CHECK:STDOUT: }

fn Main() {
  return;
}
