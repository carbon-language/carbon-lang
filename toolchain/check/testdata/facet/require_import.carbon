// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/require_import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/require_import.carbon

// --- a.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  fn F();
}

interface Y {
  extend require impls Z(Self);
}

constraint X {
  extend require impls Z(Self);
}

// --- a.impl.carbon
//@include-in-dumps
impl library "[[@TEST_NAME]]";

fn F(A:! X, B:! Y) {
  A.F();
}

// --- fail_todo_symbolic_facet_interface_requires_interface.carbon
library "[[@TEST_NAME]]";

import library "a";

fn F(B:! Y) {
  // TODO: We find the name F, but then fail to do impl lookup for `F` in `B:!
  // Y` since the identified facet type doesn't include Z. We need to be able to
  // find a symbolic facet from the require decl in Y during impl lookup.
  //
  // CHECK:STDERR: fail_todo_symbolic_facet_interface_requires_interface.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Z(B)` in type `B` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   B.F();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  B.F();
}

// CHECK:STDOUT: --- a.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %X.type: type = facet_type <@X> [concrete]
// CHECK:STDOUT:   %Self.f45: %X.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.f60: type = symbolic_binding_type Self, 0, %Self.f45 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Z.type.0ed: type = facet_type <@Z, @Z(%T)> [symbolic]
// CHECK:STDOUT:   %Self.984: %Z.type.0ed = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.0bf: type = assoc_entity_type @Z, @Z(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.f88: %Z.assoc_type.0bf = assoc_entity element0, imports.%Main.import_ref.f14 [symbolic]
// CHECK:STDOUT:   %Z.F.type.6cb: type = fn_type @Z.F, @Z(%T) [symbolic]
// CHECK:STDOUT:   %Z.F.6cc: %Z.F.type.6cb = struct_value () [symbolic]
// CHECK:STDOUT:   %Z.type.9f3244.1: type = facet_type <@Z, @Z(%Self.binding.as_type.f60)> [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.449133.1: type = assoc_entity_type @Z, @Z(%Self.binding.as_type.f60) [symbolic]
// CHECK:STDOUT:   %assoc0.179: %Z.assoc_type.449133.1 = assoc_entity element0, imports.%Main.import_ref.5847cb.1 [symbolic]
// CHECK:STDOUT:   %Z.F.type.0ab8e8.1: type = fn_type @Z.F, @Z(%Self.binding.as_type.f60) [symbolic]
// CHECK:STDOUT:   %Z.F.d97fe5.1: %Z.F.type.0ab8e8.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.fcd60c.1: %Z.type.9f3244.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %A: %X.type = symbolic_binding A, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type %X.type [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.d4d: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.63e: type = symbolic_binding_type Self, 0, %Self.d4d [symbolic]
// CHECK:STDOUT:   %Z.type.d16: type = facet_type <@Z, @Z(%Self.binding.as_type.63e)> [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.df3: type = assoc_entity_type @Z, @Z(%Self.binding.as_type.63e) [symbolic]
// CHECK:STDOUT:   %assoc0.e61: %Z.assoc_type.df3 = assoc_entity element0, imports.%Main.import_ref.5847cb.2 [symbolic]
// CHECK:STDOUT:   %Z.F.type.388: type = fn_type @Z.F, @Z(%Self.binding.as_type.63e) [symbolic]
// CHECK:STDOUT:   %Z.F.791: %Z.F.type.388 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.890: %Z.type.d16 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %B: %Y.type = symbolic_binding B, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %Y.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.binding.as_type: type = symbolic_binding_type A, 0, %A [symbolic]
// CHECK:STDOUT:   %Z.type.9f3244.2: type = facet_type <@Z, @Z(%A.binding.as_type)> [symbolic]
// CHECK:STDOUT:   %Self.fcd60c.2: %Z.type.9f3244.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.F.type.0ab8e8.2: type = fn_type @Z.F, @Z(%A.binding.as_type) [symbolic]
// CHECK:STDOUT:   %Z.F.d97fe5.2: %Z.F.type.0ab8e8.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.449133.2: type = assoc_entity_type @Z, @Z(%A.binding.as_type) [symbolic]
// CHECK:STDOUT:   %assoc0.f48: %Z.assoc_type.449133.2 = assoc_entity element0, imports.%Main.import_ref.f14 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z.type.9f3244.2 [symbolic]
// CHECK:STDOUT:   %assoc0.d5a: %Z.assoc_type.0bf = assoc_entity element0, imports.%Main.import_ref.5847cb.3 [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %A, @Z, @Z(%A.binding.as_type) [symbolic]
// CHECK:STDOUT:   %Z.facet: %Z.type.9f3244.2 = facet_value %A.binding.as_type, (%Z.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.168: type = fn_type_with_self_type %Z.F.type.0ab8e8.2, %Z.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.168 = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Z.F(%A.binding.as_type, %Z.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Z = import_ref Main//a, Z, unloaded
// CHECK:STDOUT:   %Main.Y: type = import_ref Main//a, Y, loaded [concrete = constants.%Y.type]
// CHECK:STDOUT:   %Main.X: type = import_ref Main//a, X, loaded [concrete = constants.%X.type]
// CHECK:STDOUT:   %Main.import_ref.145 = import_ref Main//a, loc3_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.f4a: @Z.%Z.assoc_type (%Z.assoc_type.0bf) = import_ref Main//a, loc4_9, loaded [symbolic = @Z.%assoc0 (constants.%assoc0.d5a)]
// CHECK:STDOUT:   %Main.F = import_ref Main//a, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.1: type = import_ref Main//a, loc3_13, loaded [symbolic = @Z.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.f14: @Z.%Z.F.type (%Z.F.type.6cb) = import_ref Main//a, loc4_9, loaded [symbolic = @Z.%Z.F (constants.%Z.F.6cc)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//a, loc3_13, loaded [symbolic = @Z.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.1dc: @Z.%Z.type (%Z.type.0ed) = import_ref Main//a, loc3_23, loaded [symbolic = @Z.%Self (constants.%Self.984)]
// CHECK:STDOUT:   %Main.import_ref.5847cb.1 = import_ref Main//a, loc4_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.67f: type = import_ref Main//a, loc12_18, loaded [symbolic = @X.Self.binding.as_type.impls.Z.type.require.%Self.binding.as_type (constants.%Self.binding.as_type.f60)]
// CHECK:STDOUT:   %Main.import_ref.8d6: type = import_ref Main//a, loc12_30, loaded [symbolic = @X.Self.binding.as_type.impls.Z.type.require.%Z.type (constants.%Z.type.9f3244.1)]
// CHECK:STDOUT:   %Main.import_ref.e33cd1.1: %X.type = import_ref Main//a, loc11_14, loaded [symbolic = constants.%Self.f45]
// CHECK:STDOUT:   %Main.import_ref.65b = import_ref Main//a, loc11_14, unloaded
// CHECK:STDOUT:   %Main.import_ref.f88: type = import_ref Main//a, loc12_31, loaded [symbolic = constants.%Z.type.9f3244.1]
// CHECK:STDOUT:   %Main.import_ref.5847cb.2 = import_ref Main//a, loc4_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.7bb: type = import_ref Main//a, loc8_18, loaded [symbolic = @Y.Self.binding.as_type.impls.Z.type.require.%Self.binding.as_type (constants.%Self.binding.as_type.63e)]
// CHECK:STDOUT:   %Main.import_ref.5ab: type = import_ref Main//a, loc8_30, loaded [symbolic = @Y.Self.binding.as_type.impls.Z.type.require.%Z.type (constants.%Z.type.d16)]
// CHECK:STDOUT:   %Main.import_ref.8c7: %Y.type = import_ref Main//a, loc7_13, loaded [symbolic = constants.%Self.d4d]
// CHECK:STDOUT:   %Main.import_ref.3e2c97.1 = import_ref Main//a, loc7_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.80a = import_ref Main//a, loc8_31, unloaded
// CHECK:STDOUT:   %Main.import_ref.5847cb.3 = import_ref Main//a, loc4_9, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Z = imports.%Main.Z
// CHECK:STDOUT:     .Y = imports.%Main.Y
// CHECK:STDOUT:     .X = imports.%Main.X
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_17.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_17.2 = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %A.patt: %pattern_type.9a5 = symbolic_binding_pattern A, 0 [concrete]
// CHECK:STDOUT:     %B.patt: %pattern_type.95b = symbolic_binding_pattern B, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_10: type = splice_block %X.ref [concrete = constants.%X.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %X.ref: type = name_ref X, imports.%Main.X [concrete = constants.%X.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %A.loc4_6.2: %X.type = symbolic_binding A, 0 [symbolic = %A.loc4_6.1 (constants.%A)]
// CHECK:STDOUT:     %.loc4_17: type = splice_block %Y.ref [concrete = constants.%Y.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Y.ref: type = name_ref Y, imports.%Main.Y [concrete = constants.%Y.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc4_13.2: %Y.type = symbolic_binding B, 1 [symbolic = %B.loc4_13.1 (constants.%B)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(imports.%Main.import_ref.b3bc94.1: type) [from "a.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T)> [symbolic = %Z.type (constants.%Z.type.0ed)]
// CHECK:STDOUT:   %Self: @Z.%Z.type (%Z.type.0ed) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.984)]
// CHECK:STDOUT:   %Z.F.type: type = fn_type @Z.F, @Z(%T) [symbolic = %Z.F.type (constants.%Z.F.type.6cb)]
// CHECK:STDOUT:   %Z.F: @Z.%Z.F.type (%Z.F.type.6cb) = struct_value () [symbolic = %Z.F (constants.%Z.F.6cc)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%T) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.0bf)]
// CHECK:STDOUT:   %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.0bf) = assoc_entity element0, imports.%Main.import_ref.f14 [symbolic = %assoc0 (constants.%assoc0.f88)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.145
// CHECK:STDOUT:     .F = imports.%Main.import_ref.f4a
// CHECK:STDOUT:     witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.3e2c97.1
// CHECK:STDOUT:   extend imports.%Main.import_ref.80a
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Y.Self.binding.as_type.impls.Z.type.require {
// CHECK:STDOUT:     require imports.%Main.import_ref.7bb impls imports.%Main.import_ref.5ab
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @X [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.65b
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend imports.%Main.import_ref.f88
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @X.Self.binding.as_type.impls.Z.type.require {
// CHECK:STDOUT:     require imports.%Main.import_ref.67f impls imports.%Main.import_ref.8d6
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @X.Self.binding.as_type.impls.Z.type.require(imports.%Main.import_ref.e33cd1.1: %X.type) [from "a.carbon"] {
// CHECK:STDOUT:   %Self: %X.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.f45)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.f60)]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%Self.binding.as_type)> [symbolic = %Z.type (constants.%Z.type.9f3244.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Y.Self.binding.as_type.impls.Z.type.require(imports.%Main.import_ref.8c7: %Y.type) [from "a.carbon"] {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.d4d)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.63e)]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%Self.binding.as_type)> [symbolic = %Z.type (constants.%Z.type.d16)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Z.F(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.1dc: @Z.%Z.type (%Z.type.0ed)) [from "a.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%A.loc4_6.2: %X.type, %B.loc4_13.2: %Y.type) {
// CHECK:STDOUT:   %A.loc4_6.1: %X.type = symbolic_binding A, 0 [symbolic = %A.loc4_6.1 (constants.%A)]
// CHECK:STDOUT:   %B.loc4_13.1: %Y.type = symbolic_binding B, 1 [symbolic = %B.loc4_13.1 (constants.%B)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.binding.as_type: type = symbolic_binding_type A, 0, %A.loc4_6.1 [symbolic = %A.binding.as_type (constants.%A.binding.as_type)]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%A.binding.as_type)> [symbolic = %Z.type (constants.%Z.type.9f3244.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%A.binding.as_type) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.449133.2)]
// CHECK:STDOUT:   %assoc0: @F.%Z.assoc_type (%Z.assoc_type.449133.2) = assoc_entity element0, imports.%Main.import_ref.f14 [symbolic = %assoc0 (constants.%assoc0.f48)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %A.loc4_6.1, @Z, @Z(%A.binding.as_type) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness)]
// CHECK:STDOUT:   %Z.F.type: type = fn_type @Z.F, @Z(%A.binding.as_type) [symbolic = %Z.F.type (constants.%Z.F.type.0ab8e8.2)]
// CHECK:STDOUT:   %Z.facet: @F.%Z.type (%Z.type.9f3244.2) = facet_value %A.binding.as_type, (%Z.lookup_impl_witness) [symbolic = %Z.facet (constants.%Z.facet)]
// CHECK:STDOUT:   %.loc5_4.3: type = fn_type_with_self_type %Z.F.type, %Z.facet [symbolic = %.loc5_4.3 (constants.%.168)]
// CHECK:STDOUT:   %impl.elem0.loc5_4.2: @F.%.loc5_4.3 (%.168) = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc5_4.2: <specific function> = specific_impl_function %impl.elem0.loc5_4.2, @Z.F(%A.binding.as_type, %Z.facet) [symbolic = %specific_impl_fn.loc5_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %A.ref: %X.type = name_ref A, %A.loc4_6.2 [symbolic = %A.loc4_6.1 (constants.%A)]
// CHECK:STDOUT:     %A.as_type: type = facet_access_type %A.ref [symbolic = %A.binding.as_type (constants.%A.binding.as_type)]
// CHECK:STDOUT:     %.loc5_4.1: type = converted %A.ref, %A.as_type [symbolic = %A.binding.as_type (constants.%A.binding.as_type)]
// CHECK:STDOUT:     %.loc5_4.2: @F.%Z.assoc_type (%Z.assoc_type.449133.2) = specific_constant imports.%Main.import_ref.f4a, @Z(constants.%A.binding.as_type) [symbolic = %assoc0 (constants.%assoc0.f48)]
// CHECK:STDOUT:     %F.ref: @F.%Z.assoc_type (%Z.assoc_type.449133.2) = name_ref F, %.loc5_4.2 [symbolic = %assoc0 (constants.%assoc0.f48)]
// CHECK:STDOUT:     %impl.elem0.loc5_4.1: @F.%.loc5_4.3 (%.168) = impl_witness_access constants.%Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc5_4.1: <specific function> = specific_impl_function %impl.elem0.loc5_4.1, @Z.F(constants.%A.binding.as_type, constants.%Z.facet) [symbolic = %specific_impl_fn.loc5_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %Z.F.call: init %empty_tuple.type = call %specific_impl_fn.loc5_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.F(constants.%T, constants.%Self.984) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self.binding.as_type.f60) {
// CHECK:STDOUT:   %T => constants.%Self.binding.as_type.f60
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.9f3244.1
// CHECK:STDOUT:   %Self => constants.%Self.fcd60c.1
// CHECK:STDOUT:   %Z.F.type => constants.%Z.F.type.0ab8e8.1
// CHECK:STDOUT:   %Z.F => constants.%Z.F.d97fe5.1
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.449133.1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.179
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.Self.binding.as_type.impls.Z.type.require(constants.%Self.f45) {
// CHECK:STDOUT:   %Self => constants.%Self.f45
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.f60
// CHECK:STDOUT:   %Z.type => constants.%Z.type.9f3244.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self.binding.as_type.63e) {
// CHECK:STDOUT:   %T => constants.%Self.binding.as_type.63e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.d16
// CHECK:STDOUT:   %Self => constants.%Self.890
// CHECK:STDOUT:   %Z.F.type => constants.%Z.F.type.388
// CHECK:STDOUT:   %Z.F => constants.%Z.F.791
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.df3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.Self.binding.as_type.impls.Z.type.require(constants.%Self.d4d) {
// CHECK:STDOUT:   %Self => constants.%Self.d4d
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.63e
// CHECK:STDOUT:   %Z.type => constants.%Z.type.d16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%A, constants.%B) {
// CHECK:STDOUT:   %A.loc4_6.1 => constants.%A
// CHECK:STDOUT:   %B.loc4_13.1 => constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%A.binding.as_type) {
// CHECK:STDOUT:   %T => constants.%A.binding.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.9f3244.2
// CHECK:STDOUT:   %Self => constants.%Self.fcd60c.2
// CHECK:STDOUT:   %Z.F.type => constants.%Z.F.type.0ab8e8.2
// CHECK:STDOUT:   %Z.F => constants.%Z.F.d97fe5.2
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.449133.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.Self.binding.as_type.impls.Z.type.require(constants.%A) {
// CHECK:STDOUT:   %Self => constants.%A
// CHECK:STDOUT:   %Self.binding.as_type => constants.%A.binding.as_type
// CHECK:STDOUT:   %Z.type => constants.%Z.type.9f3244.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.F(constants.%A.binding.as_type, constants.%Z.facet) {}
// CHECK:STDOUT:
