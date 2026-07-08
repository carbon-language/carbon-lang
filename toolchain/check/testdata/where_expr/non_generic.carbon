// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/non_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/non_generic.carbon

interface I { let T:! type; }

// Ensure that we don't crash when checking this `where` in a non-generic context.
//@dump-sem-ir-begin
fn NotGenericF(unused U: I where .T == i32) {}
//@dump-sem-ir-end

// CHECK:STDOUT: --- non_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%T [concrete]
// CHECK:STDOUT:   %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.ced: <witness> = lookup_impl_witness %.Self.frozen, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.125: type = impl_witness_access %I.lookup_impl_witness.ced, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.78b: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e55: type = impl_witness_access %I.lookup_impl_witness.78b, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where TODO> [concrete]
// CHECK:STDOUT:   %pattern_type.603: type = pattern_type %I_where.type [concrete]
// CHECK:STDOUT:   %U.param_patt: %pattern_type.603 = value_param_pattern [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.603 = at_binding_pattern U, %U.param_patt [concrete]
// CHECK:STDOUT:   %NotGenericF.type: type = fn_type @NotGenericF [concrete]
// CHECK:STDOUT:   %NotGenericF: %NotGenericF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %NotGenericF.decl: %NotGenericF.type = fn_decl @NotGenericF [concrete = constants.%NotGenericF] {
// CHECK:STDOUT:     %U.param_patt: %pattern_type.603 = value_param_pattern [concrete = constants.%U.param_patt]
// CHECK:STDOUT:     %U.patt: %pattern_type.603 = at_binding_pattern U, %U.param_patt [concrete = constants.%U.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.param: %I_where.type = value_param call_param0
// CHECK:STDOUT:     %.loc17_28.1: type = splice_block %.loc17_28.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %.Self.ref: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc17_34: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %T.ref: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc17_34.1: type = impl_witness_access constants.%I.lookup_impl_witness.ced, element0 [symbolic_self = constants.%impl.elem0.125]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %equiv.loc17_37.1 = requirement_equivalent %impl.elem0.loc17_34.1, %i32 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc17_34.2: type = impl_witness_access constants.%I.lookup_impl_witness.78b, element0 [symbolic_self = constants.%impl.elem0.e55]
// CHECK:STDOUT:       %equiv.loc17_37.2 = requirement_equivalent %impl.elem0.loc17_34.2, %i32 [concrete]
// CHECK:STDOUT:       %.loc17_28.2: type = where_expr [concrete = constants.%I_where.type] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:         %equiv.loc17_37.2 = requirement_equivalent %impl.elem0.loc17_34.2, %i32 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U: %I_where.type = wrapper_binding U, %U.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotGenericF(%U.param: %I_where.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
