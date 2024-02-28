// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface I {}

class C {
  // CHECK:STDERR: fail_extend_impl_type_as.carbon:[[@LINE+3]]:3: ERROR: Cannot `extend` an `impl` with an explicit self type.
  // CHECK:STDERR:   extend impl i32 as I {}
  // CHECK:STDERR:   ^~~~~~
  extend impl i32 as I {}
}

class D {
  // CHECK:STDERR: fail_extend_impl_type_as.carbon:[[@LINE+6]]:3: ERROR: Cannot `extend` an `impl` with an explicit self type.
  // CHECK:STDERR:   extend impl D as I;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_extend_impl_type_as.carbon:[[@LINE+3]]:15: Remove the explicit `Self` type here.
  // CHECK:STDERR:   extend impl D as I;
  // CHECK:STDERR:               ^
  extend impl D as I;
}

class E {
  // CHECK:STDERR: fail_extend_impl_type_as.carbon:[[@LINE+6]]:3: ERROR: Cannot `extend` an `impl` with an explicit self type.
  // CHECK:STDERR:   extend impl Self as I {}
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_extend_impl_type_as.carbon:[[@LINE+3]]:15: Remove the explicit `Self` type here.
  // CHECK:STDERR:   extend impl Self as I {}
  // CHECK:STDERR:               ^~~~
  extend impl Self as I {}
}

// CHECK:STDOUT: --- fail_extend_impl_type_as.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %E: type = class_type @E [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %I.decl = interface_decl @I, () [template = constants.%.1]
// CHECK:STDOUT:   %C.decl = class_decl @C, () [template = constants.%C]
// CHECK:STDOUT:   %D.decl = class_decl @D, () [template = constants.%D]
// CHECK:STDOUT:   %E.decl = class_decl @E, () [template = constants.%E]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: i32 as I {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: D as I;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: E as I {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl.1, (<unexpected instref inst+5>)
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   impl_decl @impl.2, (<unexpected instref inst+10>, <unexpected instref inst+11>)
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   impl_decl @impl.3, (<unexpected instref inst+15>, <unexpected instref inst+16>)
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
