// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: dump tokens %s
//
// TODO: Disable token output, it's not interesting for these tests.
// AUTOUPDATE

// CHECK:STDERR: fail_errors_sorted.carbon:[[@LINE+3]]:24: Closing symbol does not match most recent opening symbol.
// CHECK:STDERR: fn run(String program) {
// CHECK:STDERR:                        ^
fn run(String program) {
  return True;

// CHECK:STDERR: fail_errors_sorted.carbon:[[@LINE+3]]:10: Invalid digit 'a' in decimal numeric literal.
// CHECK:STDERR: var x = 3a;
// CHECK:STDERR:          ^
var x = 3a;

// CHECK:STDOUT: [
// CHECK:STDOUT: {{.*}}
// CHECK:STDOUT: {{.*}}
// CHECK:STDOUT: {{.*}}
// CHECK:STDOUT: {{.*}}
// CHECK:STDOUT: {{.*}}
// CHECK:STDOUT: {{.*}}
// CHECK:STDOUT: {{.*}}
// CHECK:STDOUT: {{.*}}
// CHECK:STDOUT: {{.*}}
// CHECK:STDOUT: {{.*}}
// CHECK:STDOUT: {{.*}}
// CHECK:STDOUT: {{.*}}
// CHECK:STDOUT: {{.*}}
// CHECK:STDOUT: {{.*}}
// CHECK:STDOUT: {{.*}}
// CHECK:STDOUT: {{.*}}
// CHECK:STDOUT: {{.*}}
// CHECK:STDOUT: ]
