// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface I {}

class C {
  // CHECK:STDERR: fail_extend_impl_type_as.carbon:[[@LINE+4]]:3: ERROR: Cannot `extend` an `impl` with an explicit self type.
  // CHECK:STDERR:   extend impl i32 as I {}
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  extend impl i32 as I {}
}

class D {
  // CHECK:STDERR: fail_extend_impl_type_as.carbon:[[@LINE+7]]:3: ERROR: Cannot `extend` an `impl` with an explicit self type.
  // CHECK:STDERR:   extend impl D as I;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_extend_impl_type_as.carbon:[[@LINE+4]]:15: Remove the explicit `Self` type here.
  // CHECK:STDERR:   extend impl D as I;
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
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
// CHECK:STDOUT:   %.2: <witness> = interface_witness () [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %E: type = class_type @E [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.1] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %E.decl: type = class_decl @E [template = constants.%E] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: I = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: i32 as I {
// CHECK:STDOUT:   %.1: <witness> = interface_witness () [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: D as I;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: E as I {
// CHECK:STDOUT:   %.1: <witness> = interface_witness () [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl.1 {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   impl_decl @impl.2 {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   impl_decl @impl.3 {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [template = constants.%E]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
