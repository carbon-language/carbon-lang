// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/min_prelude/runtime_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/min_prelude/runtime_value.carbon

// --- facet_value_copy_from_reference.carbon
library "[[@TEST_NAME]]";

interface I {}

class C {
  var i: I;
}

fn F(c: C) {
  // Member access produces a reference, which is copied into the value
  // binding.
  let a: I = c.i;
}

// --- fail_todo_facet_copy_narrowing_from_reference.carbon
library "[[@TEST_NAME]]";

interface I {}
interface J {}

class C {
  var ij: I & J;
}

fn F(c: C) {
  // Member access produces a reference, which is copied into the value
  // binding.
  // CHECK:STDERR: fail_todo_facet_copy_narrowing_from_reference.carbon:[[@LINE+4]]:14: error: semantics TODO: `conversion of runtime facet value` [SemanticsTodo]
  // CHECK:STDERR:   let a: I = c.ij;
  // CHECK:STDERR:              ^~~~
  // CHECK:STDERR:
  let a: I = c.ij;
}

// --- fail_todo_member_lookup_in_runtime_value.carbon

interface Z(T:! type) {
    let X:! type;
}

class C {}

final impl forall [T:! type] T as Z(C) where .X = () {}

// CHECK:STDERR: fail_todo_member_lookup_in_runtime_value.carbon:[[@LINE+4]]:18: error: semantics TODO: `associated value lookup on runtime facet value` [SemanticsTodo]
// CHECK:STDERR: fn F(T: Z(C)) -> T.(Z(C).X) {
// CHECK:STDERR:                  ^~~~~~~~~~
// CHECK:STDERR:
fn F(T: Z(C)) -> T.(Z(C).X) {
    return ();
}

// CHECK:STDOUT: --- facet_value_copy_from_reference.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %I.type [concrete]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %I.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.i [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.2b5: type = pattern_type %I.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.c48 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %.loc6: %C.elem = field_decl i, element0 [concrete]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %I.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.i [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .i = %.loc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.2b5 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %i.ref: %C.elem = name_ref i, @C.%.loc6 [concrete = @C.%.loc6]
// CHECK:STDOUT:   %.loc12_15.1: ref %I.type = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc12_15.2: %I.type = bind_value %.loc12_15.1
// CHECK:STDOUT:   %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %a: %I.type = bind_name a, %.loc12_15.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_facet_copy_narrowing_from_reference.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.f99: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.05a: %Op.type.f99 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAnd.impl_witness.0e5: <witness> = impl_witness imports.%BitAnd.impl_witness_table, @impl(type) [concrete]
// CHECK:STDOUT:   %Op.type.eb8: type = fn_type @Op.2, @impl(type) [concrete]
// CHECK:STDOUT:   %Op.444: %Op.type.eb8 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value type, (%BitAnd.impl_witness.0e5) [concrete]
// CHECK:STDOUT:   %.518: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %I.type, %Op.444 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.444, @Op.2(type) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %I.type, %Op.specific_fn [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@I & @J> [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %facet_type [concrete]
// CHECK:STDOUT:   %struct_type.ij: type = struct_type {.ij: %facet_type} [concrete]
// CHECK:STDOUT:   %complete_type.559: <witness> = complete_type_witness %struct_type.ij [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.2b5: type = pattern_type %I.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAnd: type = import_ref Core//prelude, BitAnd, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.1e6: @impl.%Op.type (%Op.type.f99) = import_ref Core//prelude, loc22_42, loaded [symbolic = @impl.%Op (constants.%Op.05a)]
// CHECK:STDOUT:   %BitAnd.impl_witness_table = impl_witness_table (%Core.import_ref.1e6), @impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.c48 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   %impl.elem0: %.518 = impl_witness_access constants.%BitAnd.impl_witness.0e5, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:   %bound_method.loc7_13.1: <bound method> = bound_method %I.ref, %impl.elem0 [concrete = constants.%Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_13.2: <bound method> = bound_method %I.ref, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %type.and: init type = call %bound_method.loc7_13.2(%I.ref, %J.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc7_13.1: type = value_of_initializer %type.and [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc7_13.2: type = converted %type.and, %.loc7_13.1 [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc7_9: %C.elem = field_decl ij, element0 [concrete]
// CHECK:STDOUT:   %struct_type.ij: type = struct_type {.ij: %facet_type} [concrete = constants.%struct_type.ij]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.ij [concrete = constants.%complete_type.559]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   .ij = %.loc7_9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.2b5 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %ij.ref: %C.elem = name_ref ij, @C.%.loc7_9 [concrete = @C.%.loc7_9]
// CHECK:STDOUT:   %.loc17_15.1: ref %facet_type = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc17_15.2: %facet_type = bind_value %.loc17_15.1
// CHECK:STDOUT:   %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %a: %I.type = bind_name a, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_member_lookup_in_runtime_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Z.type.9fb: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.9fb = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.a61: type = facet_type <@Z, @Z(%T)> [symbolic]
// CHECK:STDOUT:   %Self.2bc: %Z.type.a61 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.26e: type = assoc_entity_type @Z, @Z(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.1cd: %Z.assoc_type.26e = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Z.type.049: type = facet_type <@Z, @Z(%C)> [concrete]
// CHECK:STDOUT:   %.Self: %Z.type.049 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Self.a38: %Z.type.049 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.252: type = assoc_entity_type @Z, @Z(%C) [concrete]
// CHECK:STDOUT:   %assoc0.ea4: %Z.assoc_type.252 = assoc_entity element0, @Z.%X [concrete]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @Z, @Z(%C) [symbolic_self]
// CHECK:STDOUT:   %Z.facet: %Z.type.049 = facet_value %.Self.as_type, (%Z.lookup_impl_witness) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(%C) where %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table, @impl(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.c1f: type = pattern_type %Z.type.049 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Z.decl: %Z.type.9fb = interface_decl @Z [concrete = constants.%Z.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc2_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc2_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_20.1 [symbolic = %T.loc8_20.2 (constants.%T)]
// CHECK:STDOUT:     %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.049]
// CHECK:STDOUT:     %.Self: %Z.type.049 = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %Z.type.049 = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.loc8_46.1: %Z.assoc_type.252 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type.252 = name_ref X, %.loc8_46.1 [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc8_46.2: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc8_52.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_52.2: type = converted %.loc8_52.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc8_40: type = where_expr %.Self [concrete = constants.%Z_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc8_52.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_20.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_20.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @impl [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table, @impl(constants.%T) [symbolic = @impl.%Z.impl_witness (constants.%Z.impl_witness)]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.c1f = binding_pattern T [concrete]
// CHECK:STDOUT:     %T.param_patt: %pattern_type.c1f = value_param_pattern %T.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %Z.type.049 = name_ref T, %T
// CHECK:STDOUT:     %Z.ref.loc14_21: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %C.ref.loc14_23: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.type.loc14_24: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.049]
// CHECK:STDOUT:     %.loc14_25: %Z.assoc_type.252 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type.252 = name_ref X, %.loc14_25 [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %T.param: %Z.type.049 = value_param call_param0
// CHECK:STDOUT:     %.loc14_12: type = splice_block %Z.type.loc14_12 [concrete = constants.%Z.type.049] {
// CHECK:STDOUT:       %Z.ref.loc14_9: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:       %C.ref.loc14_11: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Z.type.loc14_12: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.049]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T: %Z.type.049 = bind_name T, %T.param
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param1
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(%T.loc2_13.1: type) {
// CHECK:STDOUT:   %T.loc2_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc2_13.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc2_13.2)> [symbolic = %Z.type (constants.%Z.type.a61)]
// CHECK:STDOUT:   %Self.2: @Z.%Z.type (%Z.type.a61) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.2bc)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%T.loc2_13.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.26e)]
// CHECK:STDOUT:   %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.26e) = assoc_entity element0, %X [symbolic = %assoc0 (constants.%assoc0.1cd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Z.%Z.type (%Z.type.a61) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.2bc)]
// CHECK:STDOUT:     %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.26e) = assoc_entity element0, @Z.%X [symbolic = @Z.%assoc0 (constants.%assoc0.1cd)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (%X)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@Z.%T.loc2_13.1: type, @Z.%Self.1: @Z.%Z.type (%Z.type.a61)) {
// CHECK:STDOUT:   assoc_const X:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc8_20.1: type) {
// CHECK:STDOUT:   %T.loc8_20.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_20.2 (constants.%T)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table, @impl(%T.loc8_20.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %.loc8_40 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Z.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%T.param: %Z.type.049) -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T) {
// CHECK:STDOUT:   %T.loc2_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T, constants.%Self.2bc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%C) {
// CHECK:STDOUT:   %T.loc2_13.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.049
// CHECK:STDOUT:   %Self.2 => constants.%Self.a38
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.252
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ea4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%Z.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc8_20.2 => constants.%T
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
