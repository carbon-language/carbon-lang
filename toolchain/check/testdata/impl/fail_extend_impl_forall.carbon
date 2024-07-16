// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/fail_extend_impl_forall.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/fail_extend_impl_forall.carbon

interface GenericInterface(T:! type) {
  fn F(x: T);
}

class C {
  // CHECK:STDERR: fail_extend_impl_forall.carbon:[[@LINE+3]]:3: ERROR: Cannot `extend` a parameterized `impl`.
  // CHECK:STDERR:   extend impl forall [T:! type] as GenericInterface(T) {
  // CHECK:STDERR:   ^~~~~~
  extend impl forall [T:! type] as GenericInterface(T) {
    fn F(x: T) {}
  }
}

// CHECK:STDOUT: --- fail_extend_impl_forall.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %GenericInterface.type: type = generic_interface_type @GenericInterface [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %GenericInterface: %GenericInterface.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = interface_type @GenericInterface, file.%GenericInterface.decl(%T) [symbolic]
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @GenericInterface, %F.type.1 [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @GenericInterface.%F.decl [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %.6: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .GenericInterface = %GenericInterface.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Core: <namespace> = namespace %Core.import, [template] {}
// CHECK:STDOUT:   %GenericInterface.decl: %GenericInterface.type = interface_decl @GenericInterface [template = constants.%GenericInterface] {
// CHECK:STDOUT:     %T.loc11_28.1: type = param T
// CHECK:STDOUT:     %T.loc11_28.2: type = bind_symbolic_name T 0, %T.loc11_28.1 [symbolic = %T.loc11_28.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @GenericInterface
// CHECK:STDOUT:     generic [file.%T.loc11_28.2: type] {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %T.ref: type = name_ref T, file.%T.loc11_28.2 [symbolic = %T.ref (constants.%T)]
// CHECK:STDOUT:     %x.loc12_8.1: @GenericInterface.%T.ref (%T) = param x
// CHECK:STDOUT:     %x.loc12_8.2: @GenericInterface.%T.ref (%T) = bind_name x, %x.loc12_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: %.3 = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc12
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C as %.2 {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %T.ref: type = name_ref T, @C.%T.loc19_23.2 [symbolic = %T.ref (constants.%T)]
// CHECK:STDOUT:     %x.loc20_10.1: @impl.%T.ref (%T) = param x
// CHECK:STDOUT:     %x.loc20_10.2: @impl.%T.ref (%T) = bind_name x, %x.loc20_10.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F.decl) [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc19_54.1: type = value_of_initializer %.loc19_52 [symbolic = constants.%.2]
// CHECK:STDOUT:   %.loc19_54.2: type = converted %.loc19_52, %.loc19_54.1 [symbolic = constants.%.2]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %T.loc19_23.1: type = param T
// CHECK:STDOUT:     %T.loc19_23.2: type = bind_symbolic_name T 0, %T.loc19_23.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %GenericInterface.ref: %GenericInterface.type = name_ref GenericInterface, file.%GenericInterface.decl [template = constants.%GenericInterface]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc19_23.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %.loc19_52: init type = call %GenericInterface.ref(%T.ref) [symbolic = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1(@GenericInterface.%x.loc12_8.2: @GenericInterface.%T.ref (%T))
// CHECK:STDOUT:     generic [file.%T.loc11_28.2: type, @GenericInterface.%Self: %.2];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(@impl.%x.loc20_10.2: @impl.%T.ref (%T))
// CHECK:STDOUT:     generic [@C.%T.loc19_23.2: type] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific file.%GenericInterface.decl(constants.%T) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   file.%T.loc11_28.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface.%F.decl(constants.%T, constants.%Self) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   @GenericInterface.%T.ref => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.%F.decl(constants.%T) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   @impl.%T.ref => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
