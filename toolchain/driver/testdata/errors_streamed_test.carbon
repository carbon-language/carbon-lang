// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{carbon} dump-tokens --print-errors=streamed %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s

fn run(String program) {
  return True;

var x = 3a;

// CHECK: {{.*}}/errors_streamed_test.carbon:11:10: Invalid digit 'a' in decimal numeric literal.
// CHECK: {{.*}}/errors_streamed_test.carbon:8:24: Closing symbol does not match most recent opening symbol.
// CHECK-COUNT-17: token: {{.*}}
