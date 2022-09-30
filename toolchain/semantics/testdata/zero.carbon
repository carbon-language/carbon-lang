// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{carbon} dump semantics-ir %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// CHECK: Function(%0, {IntegerLiteral(%1, 0), Return(%1)})
// CHECK: SetName(`Main`, %0)

fn Main() {
  return 0;
}
