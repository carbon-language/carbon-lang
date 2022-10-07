// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{carbon} dump tokens %s 2>&1 | %{FileCheck-strict} %s

// CHECK:{{.*}}/errors_sorted_test.carbon:[[@LINE+1]]:24: Closing symbol does not match most recent opening symbol.
fn run(String program) {
  return True;

// CHECK:{{.*}}/errors_sorted_test.carbon:[[@LINE+1]]:10: Invalid digit 'a' in decimal numeric literal.
var x = 3a;
// CHECK-COUNT-17:token: {{.*}}
