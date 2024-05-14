// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_extend_impl_forall.carbon:[[@LINE+4]]:1: ERROR: Semantics TODO: `generic interface`.
// CHECK:STDERR: interface GenericInterface(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface GenericInterface(T:! type) {
  fn F(x: T);
}

class C {
  // CHECK:STDERR: fail_extend_impl_forall.carbon:[[@LINE+11]]:3: ERROR: Cannot `extend` a parameterized `impl`.
  // CHECK:STDERR:   extend impl forall [T:! type] as GenericInterface(T) {
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_extend_impl_forall.carbon:[[@LINE+7]]:3: ERROR: Semantics TODO: `impl as non-interface`.
  // CHECK:STDERR:   extend impl forall [T:! type] as GenericInterface(T) {
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.1: type = interface_type @GenericInterface [template]
// CHECK:STDOUT:   %Self: GenericInterface = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %F.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @GenericInterface, F [template]
// CHECK:STDOUT:   %.4: <associated F in GenericInterface> = assoc_entity element0, @GenericInterface.%F.decl [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %F.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .GenericInterface = %GenericInterface.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %GenericInterface.decl: type = interface_decl @GenericInterface [template = constants.%.1] {
// CHECK:STDOUT:     %T.loc11_28.1: type = param T
// CHECK:STDOUT:     %T.loc11_28.2: type = bind_symbolic_name T 0, %T.loc11_28.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @GenericInterface {
// CHECK:STDOUT:   %Self: GenericInterface = bind_symbolic_name Self 1 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.1 [template = constants.%struct.1] {
// CHECK:STDOUT:     %T.ref: type = name_ref T, file.%T.loc11_28.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %x.loc12_8.1: T = param x
// CHECK:STDOUT:     %x.loc12_8.2: T = bind_name x, %x.loc12_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: <associated F in GenericInterface> = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc12
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as <error> {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.2 [template = constants.%struct.2] {
// CHECK:STDOUT:     %T.ref: type = name_ref T, @C.%T.loc27_23.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %x.loc28_10.1: T = param x
// CHECK:STDOUT:     %x.loc28_10.2: T = bind_name x, %x.loc28_10.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %T.loc27_23.1: type = param T
// CHECK:STDOUT:     %T.loc27_23.2: type = bind_symbolic_name T 0, %T.loc27_23.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %GenericInterface.ref: type = name_ref GenericInterface, file.%GenericInterface.decl [template = constants.%.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc27_23.2 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1(@GenericInterface.%x.loc12_8.2: T);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(@impl.%x.loc28_10.2: T) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
