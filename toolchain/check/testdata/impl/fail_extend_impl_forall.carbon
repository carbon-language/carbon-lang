// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_extend_impl_forall.carbon:[[@LINE+3]]:1: ERROR: Semantics TODO: `generic interface`.
// CHECK:STDERR: interface GenericInterface(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
interface GenericInterface(T:! type) {
  fn F(x: T);
}

class C {
  // CHECK:STDERR: fail_extend_impl_forall.carbon:[[@LINE+6]]:3: ERROR: Cannot `extend` a parameterized `impl`.
  // CHECK:STDERR:   extend impl forall [T:! type] as GenericInterface(T) {
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_extend_impl_forall.carbon:[[@LINE+3]]:36: ERROR: Value of type `type` is not callable.
  // CHECK:STDERR:   extend impl forall [T:! type] as GenericInterface(T) {
  // CHECK:STDERR:                                    ^~~~~~~~~~~~~~~~~
  extend impl forall [T:! type] as GenericInterface(T) {
    fn F(x: T) {}
  }
}

// CHECK:STDOUT: --- fail_extend_impl_forall.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @GenericInterface [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.GenericInterface = %GenericInterface.decl, .C = %C.decl} [template]
// CHECK:STDOUT:   %GenericInterface.decl = interface_decl @GenericInterface, (<unexpected instref inst+1>, <unexpected instref inst+2>) [template = constants.%.1]
// CHECK:STDOUT:   %C.decl = class_decl @C, () [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @GenericInterface {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as <error> {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl, (<unexpected instref inst+11>, <unexpected instref inst+12>, <unexpected instref inst+13>, <unexpected instref inst+14>)
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1(%x: T);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%x: T) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
