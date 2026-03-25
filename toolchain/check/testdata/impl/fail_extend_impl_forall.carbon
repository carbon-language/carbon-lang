// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
    fn F(unused x: T) {}
  }
}

// CHECK:STDOUT: --- fail_extend_impl_forall.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %GenericInterface.type.21d: type = generic_interface_type @GenericInterface [concrete]
// CHECK:STDOUT:   %GenericInterface.generic: %GenericInterface.type.21d = struct_value () [concrete]
// CHECK:STDOUT:   %GenericInterface.type.8bc: type = facet_type <@GenericInterface, @GenericInterface(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %GenericInterface.type.8bc = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %GenericInterface.WithSelf.F.type: type = fn_type @GenericInterface.WithSelf.F, @GenericInterface.WithSelf(%T, %Self) [symbolic]
// CHECK:STDOUT:   %GenericInterface.WithSelf.F: %GenericInterface.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %GenericInterface.assoc_type: type = assoc_entity_type @GenericInterface, @GenericInterface(%T) [symbolic]
// CHECK:STDOUT:   %assoc0: %GenericInterface.assoc_type = assoc_entity element0, @GenericInterface.WithSelf.%GenericInterface.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %GenericInterface.impl_witness: <witness> = impl_witness @C.as.GenericInterface.impl.%GenericInterface.impl_witness_table, @C.as.GenericInterface.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.GenericInterface.impl.F.type: type = fn_type @C.as.GenericInterface.impl.F, @C.as.GenericInterface.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.GenericInterface.impl.F: %C.as.GenericInterface.impl.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .GenericInterface = %GenericInterface.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericInterface.decl: %GenericInterface.type.21d = interface_decl @GenericInterface [concrete = constants.%GenericInterface.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_32.1: type = splice_block %.loc15_32.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc15_32.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_28.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_28.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @GenericInterface(%T.loc15_28.2: type) {
// CHECK:STDOUT:   %T.loc15_28.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_28.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericInterface.type: type = facet_type <@GenericInterface, @GenericInterface(%T.loc15_28.1)> [symbolic = %GenericInterface.type (constants.%GenericInterface.type.8bc)]
// CHECK:STDOUT:   %Self.loc15_38.2: @GenericInterface.%GenericInterface.type (%GenericInterface.type.8bc) = symbolic_binding Self, 1 [symbolic = %Self.loc15_38.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc15_38.1: @GenericInterface.%GenericInterface.type (%GenericInterface.type.8bc) = symbolic_binding Self, 1 [symbolic = %Self.loc15_38.2 (constants.%Self)]
// CHECK:STDOUT:     %GenericInterface.WithSelf.decl = interface_with_self_decl @GenericInterface [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %GenericInterface.WithSelf.F.decl: @GenericInterface.WithSelf.%GenericInterface.WithSelf.F.type (%GenericInterface.WithSelf.F.type) = fn_decl @GenericInterface.WithSelf.F [symbolic = @GenericInterface.WithSelf.%GenericInterface.WithSelf.F (constants.%GenericInterface.WithSelf.F)] {
// CHECK:STDOUT:       %x.patt: @GenericInterface.WithSelf.F.%pattern_type (%pattern_type.51d) = value_binding_pattern x [concrete]
// CHECK:STDOUT:       %x.param_patt: @GenericInterface.WithSelf.F.%pattern_type (%pattern_type.51d) = value_param_pattern %x.patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @GenericInterface.WithSelf.F.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref: type = name_ref T, @GenericInterface.%T.loc15_28.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %x: @GenericInterface.WithSelf.F.%T (%T) = value_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc16_13.1: @GenericInterface.WithSelf.%GenericInterface.assoc_type (%GenericInterface.assoc_type) = assoc_entity element0, %GenericInterface.WithSelf.F.decl [symbolic = %assoc0.loc16_13.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc15_38.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = @GenericInterface.WithSelf.%assoc0.loc16_13.1
// CHECK:STDOUT:     witness = (@GenericInterface.WithSelf.%GenericInterface.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.GenericInterface.impl(%T.loc24_23.2: type) {
// CHECK:STDOUT:   %T.loc24_23.1: type = symbolic_binding T, 0 [symbolic = %T.loc24_23.1 (constants.%T)]
// CHECK:STDOUT:   %GenericInterface.type.loc24_54.1: type = facet_type <@GenericInterface, @GenericInterface(%T.loc24_23.1)> [symbolic = %GenericInterface.type.loc24_54.1 (constants.%GenericInterface.type.8bc)]
// CHECK:STDOUT:   %GenericInterface.impl_witness.loc24_56.2: <witness> = impl_witness %GenericInterface.impl_witness_table, @C.as.GenericInterface.impl(%T.loc24_23.1) [symbolic = %GenericInterface.impl_witness.loc24_56.2 (constants.%GenericInterface.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.GenericInterface.impl.F.type: type = fn_type @C.as.GenericInterface.impl.F, @C.as.GenericInterface.impl(%T.loc24_23.1) [symbolic = %C.as.GenericInterface.impl.F.type (constants.%C.as.GenericInterface.impl.F.type)]
// CHECK:STDOUT:   %C.as.GenericInterface.impl.F: @C.as.GenericInterface.impl.%C.as.GenericInterface.impl.F.type (%C.as.GenericInterface.impl.F.type) = struct_value () [symbolic = %C.as.GenericInterface.impl.F (constants.%C.as.GenericInterface.impl.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %GenericInterface.type.loc24_54.2 {
// CHECK:STDOUT:     %C.as.GenericInterface.impl.F.decl: @C.as.GenericInterface.impl.%C.as.GenericInterface.impl.F.type (%C.as.GenericInterface.impl.F.type) = fn_decl @C.as.GenericInterface.impl.F [symbolic = @C.as.GenericInterface.impl.%C.as.GenericInterface.impl.F (constants.%C.as.GenericInterface.impl.F)] {
// CHECK:STDOUT:       %x.patt: @C.as.GenericInterface.impl.F.%pattern_type (%pattern_type.51d) = value_binding_pattern x [concrete]
// CHECK:STDOUT:       %x.param_patt: @C.as.GenericInterface.impl.F.%pattern_type (%pattern_type.51d) = value_param_pattern %x.patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @C.as.GenericInterface.impl.F.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref: type = name_ref T, @C.as.GenericInterface.impl.%T.loc24_23.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %x: @C.as.GenericInterface.impl.F.%T (%T) = value_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %GenericInterface.impl_witness_table = impl_witness_table (), @C.as.GenericInterface.impl [concrete]
// CHECK:STDOUT:     %GenericInterface.impl_witness.loc24_56.1: <witness> = impl_witness %GenericInterface.impl_witness_table, @C.as.GenericInterface.impl(constants.%T) [symbolic = %GenericInterface.impl_witness.loc24_56.2 (constants.%GenericInterface.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %C.as.GenericInterface.impl.F.decl
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @C.as.GenericInterface.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %GenericInterface.ref: %GenericInterface.type.21d = name_ref GenericInterface, file.%GenericInterface.decl [concrete = constants.%GenericInterface.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc24_23.2 [symbolic = %T.loc24_23.1 (constants.%T)]
// CHECK:STDOUT:     %GenericInterface.type.loc24_54.2: type = facet_type <@GenericInterface, @GenericInterface(constants.%T)> [symbolic = %GenericInterface.type.loc24_54.1 (constants.%GenericInterface.type.8bc)]
// CHECK:STDOUT:     %.loc24_27.1: type = splice_block %.loc24_27.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc24_27.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc24_23.2: type = symbolic_binding T, 0 [symbolic = %T.loc24_23.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .GenericInterface = <poisoned>
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GenericInterface.WithSelf.F(@GenericInterface.%T.loc15_28.2: type, @GenericInterface.%Self.loc15_38.1: @GenericInterface.%GenericInterface.type (%GenericInterface.type.8bc)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @GenericInterface.WithSelf.F.%T (%T));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.GenericInterface.impl.F(@C.as.GenericInterface.impl.%T.loc24_23.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @C.as.GenericInterface.impl.F.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface(constants.%T) {
// CHECK:STDOUT:   %T.loc15_28.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface.WithSelf(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericInterface.WithSelf.F(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.GenericInterface.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc24_23.1 => constants.%T
// CHECK:STDOUT:   %GenericInterface.type.loc24_54.1 => constants.%GenericInterface.type.8bc
// CHECK:STDOUT:   %GenericInterface.impl_witness.loc24_56.2 => constants.%GenericInterface.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.GenericInterface.impl.F.type => constants.%C.as.GenericInterface.impl.F.type
// CHECK:STDOUT:   %C.as.GenericInterface.impl.F => constants.%C.as.GenericInterface.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.GenericInterface.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
