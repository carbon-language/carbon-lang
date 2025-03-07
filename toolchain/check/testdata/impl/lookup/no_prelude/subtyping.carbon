// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/no_prelude/subtyping.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/no_prelude/subtyping.carbon

// This is testing `FindWitnessInFacet` from impl_lookup.cpp

// --- equivalent.carbon
library "[[@TEST_NAME]]";

interface A {}

fn TakesA[T:! A](x: T) {}

fn WithExtraWhere[U:! A where .Self impls type](y: U) {
  TakesA(y);
}

// CHECK:STDOUT: --- equivalent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: %A.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %A.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %TakesA.type: type = fn_type @TakesA [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TakesA: %TakesA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.cf4: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.Self: %A.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %A_where.type: type = facet_type <@A where TODO> [concrete]
// CHECK:STDOUT:   %U: %A_where.type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: %A_where.type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %WithExtraWhere.type: type = fn_type @WithExtraWhere [concrete]
// CHECK:STDOUT:   %WithExtraWhere: %WithExtraWhere.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.732: <witness> = require_complete_type %U.as_type [symbolic]
// CHECK:STDOUT:   %U.as_wit: <witness> = facet_access_witness %U [symbolic]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %U.as_type, %U.as_wit [symbolic]
// CHECK:STDOUT:   %TakesA.specific_fn: <specific function> = specific_function %TakesA, @TakesA(%A.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .TakesA = %TakesA.decl
// CHECK:STDOUT:     .WithExtraWhere = %WithExtraWhere.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %TakesA.decl: %TakesA.type = fn_decl @TakesA [concrete = constants.%TakesA] {
// CHECK:STDOUT:     %T.patt.loc5_11.1: %A.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_11.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @TakesA.%T.as_type.loc5_21.2 (%T.as_type) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @TakesA.%T.as_type.loc5_21.2 (%T.as_type) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     %T.loc5_11.1: %A.type = bind_symbolic_name T, 0 [symbolic = %T.loc5_11.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @TakesA.%T.as_type.loc5_21.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_21.1: type = splice_block %.loc5_21.2 [symbolic = %T.as_type.loc5_21.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc5_11.1 [symbolic = %T.loc5_11.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc5_21.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc5_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc5_21.2: type = converted %T.ref, %T.as_type.loc5_21.1 [symbolic = %T.as_type.loc5_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @TakesA.%T.as_type.loc5_21.2 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithExtraWhere.decl: %WithExtraWhere.type = fn_decl @WithExtraWhere [concrete = constants.%WithExtraWhere] {
// CHECK:STDOUT:     %U.patt.loc7_19.1: %A_where.type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc7_19.2 (constants.%U.patt)]
// CHECK:STDOUT:     %y.patt: @WithExtraWhere.%U.as_type.loc7_52.2 (%U.as_type) = binding_pattern y
// CHECK:STDOUT:     %y.param_patt: @WithExtraWhere.%U.as_type.loc7_52.2 (%U.as_type) = value_param_pattern %y.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_25.1: type = splice_block %.loc7_25.2 [concrete = constants.%A_where.type] {
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %.Self: %A.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %A.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc7_31: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc7_25.2: type = where_expr %.Self [concrete = constants.%A_where.type] {
// CHECK:STDOUT:         requirement_impls %.loc7_31, type
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_19.1: %A_where.type = bind_symbolic_name U, 0 [symbolic = %U.loc7_19.2 (constants.%U)]
// CHECK:STDOUT:     %y.param: @WithExtraWhere.%U.as_type.loc7_52.2 (%U.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_52.1: type = splice_block %.loc7_52.2 [symbolic = %U.as_type.loc7_52.2 (constants.%U.as_type)] {
// CHECK:STDOUT:       %U.ref: %A_where.type = name_ref U, %U.loc7_19.1 [symbolic = %U.loc7_19.2 (constants.%U)]
// CHECK:STDOUT:       %U.as_type.loc7_52.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc7_52.2 (constants.%U.as_type)]
// CHECK:STDOUT:       %.loc7_52.2: type = converted %U.ref, %U.as_type.loc7_52.1 [symbolic = %U.as_type.loc7_52.2 (constants.%U.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: @WithExtraWhere.%U.as_type.loc7_52.2 (%U.as_type) = bind_name y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesA(%T.loc5_11.1: %A.type) {
// CHECK:STDOUT:   %T.loc5_11.2: %A.type = bind_symbolic_name T, 0 [symbolic = %T.loc5_11.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc5_11.2: %A.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_11.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc5_21.2: type = facet_access_type %T.loc5_11.2 [symbolic = %T.as_type.loc5_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @TakesA.%T.as_type.loc5_21.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete.cf4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc5_11.1: %A.type](%x.param_patt: @TakesA.%T.as_type.loc5_21.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WithExtraWhere(%U.loc7_19.1: %A_where.type) {
// CHECK:STDOUT:   %U.loc7_19.2: %A_where.type = bind_symbolic_name U, 0 [symbolic = %U.loc7_19.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc7_19.2: %A_where.type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc7_19.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.as_type.loc7_52.2: type = facet_access_type %U.loc7_19.2 [symbolic = %U.as_type.loc7_52.2 (constants.%U.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @WithExtraWhere.%U.as_type.loc7_52.2 (%U.as_type) [symbolic = %require_complete (constants.%require_complete.732)]
// CHECK:STDOUT:   %U.as_wit.loc8_11.3: <witness> = facet_access_witness %U.loc7_19.2 [symbolic = %U.as_wit.loc8_11.3 (constants.%U.as_wit)]
// CHECK:STDOUT:   %A.facet.loc8_11.3: %A.type = facet_value %U.as_type.loc7_52.2, %U.as_wit.loc8_11.3 [symbolic = %A.facet.loc8_11.3 (constants.%A.facet)]
// CHECK:STDOUT:   %TakesA.specific_fn.loc8_3.2: <specific function> = specific_function constants.%TakesA, @TakesA(%A.facet.loc8_11.3) [symbolic = %TakesA.specific_fn.loc8_3.2 (constants.%TakesA.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%U.patt.loc7_19.1: %A_where.type](%y.param_patt: @WithExtraWhere.%U.as_type.loc7_52.2 (%U.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %TakesA.ref: %TakesA.type = name_ref TakesA, file.%TakesA.decl [concrete = constants.%TakesA]
// CHECK:STDOUT:     %y.ref: @WithExtraWhere.%U.as_type.loc7_52.2 (%U.as_type) = name_ref y, %y
// CHECK:STDOUT:     %U.as_wit.loc8_11.1: <witness> = facet_access_witness constants.%U [symbolic = %U.as_wit.loc8_11.3 (constants.%U.as_wit)]
// CHECK:STDOUT:     %A.facet.loc8_11.1: %A.type = facet_value constants.%U.as_type, %U.as_wit.loc8_11.1 [symbolic = %A.facet.loc8_11.3 (constants.%A.facet)]
// CHECK:STDOUT:     %.loc8_11.1: %A.type = converted constants.%U.as_type, %A.facet.loc8_11.1 [symbolic = %A.facet.loc8_11.3 (constants.%A.facet)]
// CHECK:STDOUT:     %U.as_wit.loc8_11.2: <witness> = facet_access_witness constants.%U [symbolic = %U.as_wit.loc8_11.3 (constants.%U.as_wit)]
// CHECK:STDOUT:     %A.facet.loc8_11.2: %A.type = facet_value constants.%U.as_type, %U.as_wit.loc8_11.2 [symbolic = %A.facet.loc8_11.3 (constants.%A.facet)]
// CHECK:STDOUT:     %.loc8_11.2: %A.type = converted constants.%U.as_type, %A.facet.loc8_11.2 [symbolic = %A.facet.loc8_11.3 (constants.%A.facet)]
// CHECK:STDOUT:     %TakesA.specific_fn.loc8_3.1: <specific function> = specific_function %TakesA.ref, @TakesA(constants.%A.facet) [symbolic = %TakesA.specific_fn.loc8_3.2 (constants.%TakesA.specific_fn)]
// CHECK:STDOUT:     %TakesA.call: init %empty_tuple.type = call %TakesA.specific_fn.loc8_3.1(%y.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesA(constants.%T) {
// CHECK:STDOUT:   %T.loc5_11.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc5_11.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc5_21.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithExtraWhere(constants.%U) {
// CHECK:STDOUT:   %U.loc7_19.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc7_19.2 => constants.%U
// CHECK:STDOUT:   %U.as_type.loc7_52.2 => constants.%U.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesA(constants.%A.facet) {
// CHECK:STDOUT:   %T.loc5_11.2 => constants.%A.facet
// CHECK:STDOUT:   %T.patt.loc5_11.2 => constants.%A.facet
// CHECK:STDOUT:   %T.as_type.loc5_21.2 => constants.%U.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.732
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesA(@WithExtraWhere.%A.facet.loc8_11.3) {}
// CHECK:STDOUT:
