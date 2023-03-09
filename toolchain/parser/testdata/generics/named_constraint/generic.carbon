// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: 5
// CHECK:STDOUT: 5
// CHECK:STDOUT: 5
// CHECK:STDOUT: 5
// CHECK:STDOUT: 5
// CHECK:STDOUT: result: 0

constraint IndirectMaker(T:! type) {
  extends Maker(T*);
}
