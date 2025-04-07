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
  // CHECK:STDERR: fail_extend_impl_forall.carbon:[[@LINE+4]]:3: error: cannot `extend` a parameterized `impl` [ExtendImplForall]
  // CHECK:STDERR:   extend impl forall [T:! type] as GenericInterface(T) {
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  extend impl forall [T:! type] as GenericInterface(T) {
    fn F(x: T) {}
  }
}

// CHECK:STDOUT: --- fail_extend_impl_forall.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %GenericInterface.type.c92: type = generic_interface_type @GenericInterface [concrete]
// CHECK:STDOUT:   %GenericInterface.generic: %GenericInterface.type.c92 = struct_value () [concrete]
// CHECK:STDOUT:   %GenericInterface.type.3fe: type = facet_type <@GenericInterface, @GenericInterface(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %GenericInterface.type.3fe = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.a4b: type = fn_type @F.1, @GenericInterface(%T) [symbolic]
// CHECK:STDOUT:   %F.3d9: %F.type.a4b = struct_value () [symbolic]
// CHECK:STDOUT:   %GenericInterface.assoc_type: type = assoc_entity_type @GenericInterface, @GenericInterface(%T) [symbolic]
// CHECK:STDOUT:   %assoc0: %GenericInterface.assoc_type = assoc_entity element0, @GenericInterface.%F.decl [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %require_complete.70f: <witness> = require_complete_type %GenericInterface.type.3fe [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (invalid), @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.type.2cb: type = fn_type @F.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.4b1: %F.type.2cb = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .GenericInterface = %GenericInterface.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %GenericInterface.decl: %GenericInterface.type.c92 = interface_decl @GenericInterface [concrete = constants.%GenericInterface.generic] {
// CHECK:STDOUT:     %T.patt.loc11_28.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_28.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc11_28.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_28.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericInterface(%T.loc11_28.1: type) {
// CHECK:STDOUT:   %T.loc11_28.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_28.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_28.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_28.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericInterface.type: type = facet_type <@GenericInterface, @GenericInterface(%T.loc11_28.2)> [symbolic = %GenericInterface.type (constants.%GenericInterface.type.3fe)]
// CHECK:STDOUT:   %Self.2: @GenericInterface.%GenericInterface.type (%GenericInterface.type.3fe) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @GenericInterface(%T.loc11_28.2) [symbolic = %F.type (constants.%F.type.a4b)]
// CHECK:STDOUT:   %F: @GenericInterface.%F.type (%F.type.a4b) = struct_value () [symbolic = %F (constants.%F.3d9)]
// CHECK:STDOUT:   %GenericInterface.assoc_type: type = assoc_entity_type @GenericInterface, @GenericInterface(%T.loc11_28.2) [symbolic = %GenericInterface.assoc_type (constants.%GenericInterface.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc12_13.2: @GenericInterface.%GenericInterface.assoc_type (%GenericInterface.assoc_type) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc12_13.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @GenericInterface.%GenericInterface.type (%GenericInterface.type.3fe) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @GenericInterface.%F.type (%F.type.a4b) = fn_decl @F.1 [symbolic = @GenericInterface.%F (constants.%F.3d9)] {
// CHECK:STDOUT:       %x.patt: @F.1.%T (%T) = binding_pattern x
// CHECK:STDOUT:       %x.param_patt: @F.1.%T (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @F.1.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref: type = name_ref T, @GenericInterface.%T.loc11_28.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %x: @F.1.%T (%T) = bind_name x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc12_13.1: @GenericInterface.%GenericInterface.assoc_type (%GenericInterface.assoc_type) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc12_13.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc12_13.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc20_23.1: type) {
// CHECK:STDOUT:   %T.loc20_23.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc20_23.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc20_23.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc20_23.2 (constants.%T.patt)]
// CHECK:STDOUT:   %GenericInterface.type.loc20_54.2: type = facet_type <@GenericInterface, @GenericInterface(%T.loc20_23.2)> [symbolic = %GenericInterface.type.loc20_54.2 (constants.%GenericInterface.type.3fe)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%GenericInterface.type.loc20_54.2 (%GenericInterface.type.3fe) [symbolic = %require_complete (constants.%require_complete.70f)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (invalid), @impl(%T.loc20_23.2) [symbolic = %impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%T.loc20_23.2) [symbolic = %F.type (constants.%F.type.2cb)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.2cb) = struct_value () [symbolic = %F (constants.%F.4b1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %GenericInterface.type.loc20_54.1 {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.2cb) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.4b1)] {
// CHECK:STDOUT:       %x.patt: @F.2.%T (%T) = binding_pattern x
// CHECK:STDOUT:       %x.param_patt: @F.2.%T (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @F.2.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref: type = name_ref T, @impl.%T.loc20_23.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %x: @F.2.%T (%T) = bind_name x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc20_23.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc20_23.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %GenericInterface.ref: %GenericInterface.type.c92 = name_ref GenericInterface, file.%GenericInterface.decl [concrete = constants.%GenericInterface.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc20_23.1 [symbolic = %T.loc20_23.2 (constants.%T)]
// CHECK:STDOUT:     %GenericInterface.type.loc20_54.1: type = facet_type <@GenericInterface, @GenericInterface(constants.%T)> [symbolic = %GenericInterface.type.loc20_54.2 (constants.%GenericInterface.type.3fe)]
// CHECK:STDOUT:     %T.loc20_23.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc20_23.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (invalid), @impl(constants.%T) [symbolic = @impl.%impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:   %.loc20: type = specific_constant @impl.%GenericInterface.type.loc20_54.1, @impl(constants.%T) [symbolic = constants.%GenericInterface.type.3fe]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .GenericInterface = <poisoned>
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@GenericInterface.%T.loc11_28.1: type, @GenericInterface.%Self.1: @GenericInterface.%GenericInterface.type (%GenericInterface.type.3fe)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param_patt: @F.1.%T (%T));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@impl.%T.loc20_23.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.2.%T (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param_patt: @F.2.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(constants.%T) {
// CHECK:STDOUT:   %T.loc11_28.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_28.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericInterface.type => constants.%GenericInterface.type.3fe
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.a4b
// CHECK:STDOUT:   %F => constants.%F.3d9
// CHECK:STDOUT:   %GenericInterface.assoc_type => constants.%GenericInterface.assoc_type
// CHECK:STDOUT:   %assoc0.loc12_13.2 => constants.%assoc0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(%T.loc11_28.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc20_23.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc20_23.2 => constants.%T.patt
// CHECK:STDOUT:   %GenericInterface.type.loc20_54.2 => constants.%GenericInterface.type.3fe
// CHECK:STDOUT:   %require_complete => constants.%require_complete.70f
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.2cb
// CHECK:STDOUT:   %F => constants.%F.4b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(@impl.%T.loc20_23.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc20_23.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
