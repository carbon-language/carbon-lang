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
  // CHECK:STDERR: fail_extend_impl_forall.carbon:[[@LINE+3]]:3: error: cannot `extend` a parameterized `impl` [ExtendImplForall]
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
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %GenericInterface.type.1: type = generic_interface_type @GenericInterface [template]
// CHECK:STDOUT:   %GenericInterface: %GenericInterface.type.1 = struct_value () [template]
// CHECK:STDOUT:   %GenericInterface.type.2: type = facet_type <@GenericInterface, @GenericInterface(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %GenericInterface.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @GenericInterface(%T) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %GenericInterface.type.2, %F.type.1 [symbolic]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @GenericInterface.%F.decl [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: %GenericInterface.type.2 = facet_value %C, %C [symbolic]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%F.2) [symbolic]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %.5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
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
// CHECK:STDOUT:     %T.patt.loc11_28.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_28.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_28.1, runtime_param<invalid> [symbolic = %T.patt.loc11_28.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_28.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_28.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericInterface(%T.loc11_28.1: type) {
// CHECK:STDOUT:   %T.loc11_28.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_28.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_28.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_28.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericInterface.type: type = facet_type <@GenericInterface, @GenericInterface(%T.loc11_28.2)> [symbolic = %GenericInterface.type (constants.%GenericInterface.type.2)]
// CHECK:STDOUT:   %Self.2: %GenericInterface.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @GenericInterface(%T.loc11_28.2) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @GenericInterface.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:   %.loc12_13.2: type = assoc_entity_type @GenericInterface.%GenericInterface.type (%GenericInterface.type.2), @GenericInterface.%F.type (%F.type.1) [symbolic = %.loc12_13.2 (constants.%.1)]
// CHECK:STDOUT:   %.loc12_13.3: @GenericInterface.%.loc12_13.2 (%.1) = assoc_entity element0, %F.decl [symbolic = %.loc12_13.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @GenericInterface.%GenericInterface.type (%GenericInterface.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @GenericInterface.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = @GenericInterface.%F (constants.%F.1)] {
// CHECK:STDOUT:       %x.patt: @F.1.%T (%T) = binding_pattern x
// CHECK:STDOUT:       %x.param_patt: @F.1.%T (%T) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @GenericInterface.%T.loc11_28.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %x.param: @F.1.%T (%T) = value_param runtime_param0
// CHECK:STDOUT:       %x: @F.1.%T (%T) = bind_name x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc12_13.1: @GenericInterface.%.loc12_13.2 (%.1) = assoc_entity element0, %F.decl [symbolic = %.loc12_13.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %.loc12_13.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc19_23.1: type) {
// CHECK:STDOUT:   %T.loc19_23.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc19_23.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc19_23.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc19_23.2 (constants.%T.patt)]
// CHECK:STDOUT:   %GenericInterface.type.loc19_52.2: type = facet_type <@GenericInterface, @GenericInterface(%T.loc19_23.2)> [symbolic = %GenericInterface.type.loc19_52.2 (constants.%GenericInterface.type.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%T.loc19_23.2) [symbolic = %F.type (constants.%F.type.2)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.2) = struct_value () [symbolic = %F (constants.%F.2)]
// CHECK:STDOUT:   %.loc19_56.2: <witness> = interface_witness (%F) [symbolic = %.loc19_56.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %GenericInterface.type.loc19_52.1 {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.2) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.2)] {
// CHECK:STDOUT:       %x.patt: @F.2.%T (%T) = binding_pattern x
// CHECK:STDOUT:       %x.param_patt: @F.2.%T (%T) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @impl.%T.loc19_23.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %x.param: @F.2.%T (%T) = value_param runtime_param0
// CHECK:STDOUT:       %x: @F.2.%T (%T) = bind_name x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc19_56.1: <witness> = interface_witness (%F.decl) [symbolic = %.loc19_56.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = %.loc19_56.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl [template] {
// CHECK:STDOUT:     %T.patt.loc19_23.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc19_23.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc19_23.1, runtime_param<invalid> [symbolic = %T.patt.loc19_23.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %GenericInterface.ref: %GenericInterface.type.1 = name_ref GenericInterface, file.%GenericInterface.decl [template = constants.%GenericInterface]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc19_23.1 [symbolic = %T.loc19_23.2 (constants.%T)]
// CHECK:STDOUT:     %GenericInterface.type.loc19_52.1: type = facet_type <@GenericInterface, @GenericInterface(constants.%T)> [symbolic = %GenericInterface.type.loc19_52.2 (constants.%GenericInterface.type.2)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc19_23.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc19_23.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19: type = specific_constant @impl.%GenericInterface.type.loc19_52.1, @impl(constants.%T) [symbolic = constants.%GenericInterface.type.2]
// CHECK:STDOUT:   %.loc22: <witness> = complete_type_witness %.5 [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@GenericInterface.%T.loc11_28.1: type, @GenericInterface.%Self.1: @GenericInterface.%GenericInterface.type (%GenericInterface.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param_patt: @F.1.%T (%T));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@impl.%T.loc19_23.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param_patt: @F.2.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(constants.%T) {
// CHECK:STDOUT:   %T.loc11_28.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_28.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericInterface.type => constants.%GenericInterface.type.2
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.1
// CHECK:STDOUT:   %F => constants.%F.1
// CHECK:STDOUT:   %.loc12_13.2 => constants.%.1
// CHECK:STDOUT:   %.loc12_13.3 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(%T.loc11_28.2) {
// CHECK:STDOUT:   %T.loc11_28.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_28.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(@impl.%T.loc19_23.2) {
// CHECK:STDOUT:   %T.loc11_28.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_28.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc19_23.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc19_23.2 => constants.%T
// CHECK:STDOUT:   %GenericInterface.type.loc19_52.2 => constants.%GenericInterface.type.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F => constants.%F.2
// CHECK:STDOUT:   %.loc19_56.2 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%.3) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc19_23.2) {
// CHECK:STDOUT:   %T.loc19_23.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc19_23.2 => constants.%T
// CHECK:STDOUT:   %GenericInterface.type.loc19_52.2 => constants.%GenericInterface.type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
