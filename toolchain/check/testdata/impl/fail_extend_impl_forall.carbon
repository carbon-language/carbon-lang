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
  // CHECK:STDERR: fail_extend_impl_forall.carbon:[[@LINE+3]]:3: error: cannot `extend` a parameterized `impl`
  // CHECK:STDERR:   extend impl forall [T:! type] as GenericInterface(T) {
  // CHECK:STDERR:   ^~~~~~
  extend impl forall [T:! type] as GenericInterface(T) {
    fn F(x: T) {}
  }
}

// CHECK:STDOUT: --- fail_extend_impl_forall.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %GenericInterface.type.1: type = generic_interface_type @GenericInterface [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %GenericInterface: %GenericInterface.type.1 = struct_value () [template]
// CHECK:STDOUT:   %GenericInterface.type.2: type = interface_type @GenericInterface, @GenericInterface(%T) [symbolic]
// CHECK:STDOUT:   %Self: %GenericInterface.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @GenericInterface(%T) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %GenericInterface.type.2, %F.type.1 [symbolic]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @GenericInterface.%F.decl [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %.5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .GenericInterface = %GenericInterface.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %GenericInterface.decl: %GenericInterface.type.1 = interface_decl @GenericInterface [template = constants.%GenericInterface] {
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericInterface(%T.loc11: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericInterface.type: type = interface_type @GenericInterface, @GenericInterface(%T.1) [symbolic = %GenericInterface.type (constants.%GenericInterface.type.2)]
// CHECK:STDOUT:   %Self.2: %GenericInterface.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @GenericInterface(%T.1) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @GenericInterface.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @GenericInterface.%GenericInterface.type (%GenericInterface.type.2), @GenericInterface.%F.type (%F.type.1) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %.2: @GenericInterface.%.1 (%.2) = assoc_entity element0, %F.decl [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @GenericInterface.%GenericInterface.type (%GenericInterface.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @GenericInterface.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = @GenericInterface.%F (constants.%F.1)] {
// CHECK:STDOUT:       %x.patt: @F.1.%T (%T) = binding_pattern x
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @GenericInterface.%T.loc11 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %x.param: @F.1.%T (%T) = param x, runtime_param0
// CHECK:STDOUT:       %x: @F.1.%T (%T) = bind_name x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc12: @GenericInterface.%.1 (%.2) = assoc_entity element0, %F.decl [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %.loc12
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc19: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %GenericInterface.type.1: type = interface_type @GenericInterface, @GenericInterface(%T.1) [symbolic = %GenericInterface.type.1 (constants.%GenericInterface.type.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %GenericInterface.type.loc19 {
// CHECK:STDOUT:     %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:       %x.patt: @F.2.%T (%T) = binding_pattern x
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @impl.%T.loc19 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %x.param: @F.2.%T (%T) = param x, runtime_param0
// CHECK:STDOUT:       %x: @F.2.%T (%T) = bind_name x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc19: <witness> = interface_witness (%F.decl) [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = %.loc19
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl [template] {
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc19: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %GenericInterface.ref: %GenericInterface.type.1 = name_ref GenericInterface, file.%GenericInterface.decl [template = constants.%GenericInterface]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc19 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %GenericInterface.type.loc19: type = interface_type @GenericInterface, @GenericInterface(constants.%T) [symbolic = %GenericInterface.type.1 (constants.%GenericInterface.type.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22: <witness> = complete_type_witness %.5 [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@GenericInterface.%T.loc11: type, @GenericInterface.%Self.1: @GenericInterface.%GenericInterface.type (%GenericInterface.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x: @F.1.%T (%T));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@impl.%T.loc19: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x: @F.2.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericInterface.type => constants.%GenericInterface.type.2
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.1
// CHECK:STDOUT:   %F => constants.%F.1
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT:   %.2 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(@GenericInterface.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(@impl.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %GenericInterface.type.1 => constants.%GenericInterface.type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%C) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
