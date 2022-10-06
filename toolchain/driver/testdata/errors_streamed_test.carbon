// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{carbon} --print-errors=streamed dump tokens %s 2>&1 | \
// RUN:   %{FileCheck-strict} %s

fn run(String program) {
  return True;

var x = 3a;

// CHECK:{{.*}}/errors_streamed_test.carbon:[[@LINE-2]]:10: Invalid digit 'a' in decimal numeric literal.
// CHECK:{{.*}}/errors_streamed_test.carbon:[[@LINE-6]]:24: Closing symbol does not match most recent opening symbol.
// CHECK-COUNT-17:token: {{.*}}
