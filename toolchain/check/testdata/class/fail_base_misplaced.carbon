// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class B {}

// CHECK:STDERR: fail_base_misplaced.carbon:[[@LINE+3]]:1: ERROR: `base` declaration can only be used in a class.
// CHECK:STDERR: extend base: B;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
extend base: B;

fn F() {
  // CHECK:STDERR: fail_base_misplaced.carbon:[[@LINE+6]]:3: ERROR: Expected expression.
  // CHECK:STDERR:   extend base: B;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_base_misplaced.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `HandleInvalidParse`.
  // CHECK:STDERR:   extend base: B;
  // CHECK:STDERR:   ^~~~~~
  extend base: B;
}

// CHECK:STDOUT: --- fail_base_misplaced.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT: }
// CHECK:STDOUT:
