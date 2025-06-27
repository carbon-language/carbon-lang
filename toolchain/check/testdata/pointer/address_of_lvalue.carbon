// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/address_of_lvalue.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/address_of_lvalue.carbon

fn F() {
  var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};

  var p: {.a: i32, .b: i32}* = &s;
  var q: i32* = &s.a;
  var r: i32* = &s.b;

  var t: (i32, i32) = (1, 2);
  var t0: i32* = &t.0;
  var t1: i32* = &t.1;
}

// CHECK:STDOUT: --- address_of_lvalue.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.501: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %ptr.3ee: type = ptr_type %struct_type.a.b.501 [concrete]
// CHECK:STDOUT:   %pattern_type.851: type = pattern_type %struct_type.a.b.501 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.501 = struct_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %pattern_type.8dd: type = pattern_type %ptr.3ee [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %ptr.235 [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.511: type = pattern_type %tuple.type.d07 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.3, @impl.49c(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.711: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%ptr.235) [concrete]
// CHECK:STDOUT:   %Op.type.ba2: type = fn_type @Op.3, @impl.49c(%ptr.235) [concrete]
// CHECK:STDOUT:   %Op.649: %Op.type.ba2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.2bf: %Destroy.type = facet_value %ptr.235, (%Destroy.impl_witness.711) [concrete]
// CHECK:STDOUT:   %.a79: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.2bf [concrete]
// CHECK:STDOUT:   %Op.specific_fn.829: <specific function> = specific_function %Op.649, @Op.3(%ptr.235) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.5a7: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%tuple.type.d07) [concrete]
// CHECK:STDOUT:   %Op.type.a4a: type = fn_type @Op.3, @impl.49c(%tuple.type.d07) [concrete]
// CHECK:STDOUT:   %Op.7c6: %Op.type.a4a = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.167: %Destroy.type = facet_value %tuple.type.d07, (%Destroy.impl_witness.5a7) [concrete]
// CHECK:STDOUT:   %.b68: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.167 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.bbb: <specific function> = specific_function %Op.7c6, @Op.3(%tuple.type.d07) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.ff6: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%ptr.3ee) [concrete]
// CHECK:STDOUT:   %Op.type.757: type = fn_type @Op.3, @impl.49c(%ptr.3ee) [concrete]
// CHECK:STDOUT:   %Op.e22: %Op.type.757 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.ed3: %Destroy.type = facet_value %ptr.3ee, (%Destroy.impl_witness.ff6) [concrete]
// CHECK:STDOUT:   %.f93: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.ed3 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.ce1: <specific function> = specific_function %Op.e22, @Op.3(%ptr.3ee) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.ad1: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%struct_type.a.b.501) [concrete]
// CHECK:STDOUT:   %Op.type.da3: type = fn_type @Op.3, @impl.49c(%struct_type.a.b.501) [concrete]
// CHECK:STDOUT:   %Op.9db: %Op.type.da3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.807: %Destroy.type = facet_value %struct_type.a.b.501, (%Destroy.impl_witness.ad1) [concrete]
// CHECK:STDOUT:   %.8e0: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.807 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.1e0: <specific function> = specific_function %Op.9db, @Op.3(%struct_type.a.b.501) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.49c.%Op.type (%Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_23, loaded [symbolic = @impl.49c.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl.49c [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.851 = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.851 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %struct_type.a.b.501 = var %s.var_patt
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc16: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc16_46.1: %struct_type.a.b.cfd = struct_literal (%int_1.loc16, %int_2.loc16)
// CHECK:STDOUT:   %impl.elem0.loc16_46.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc16_46.1: <bound method> = bound_method %int_1.loc16, %impl.elem0.loc16_46.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc16_46.1: <specific function> = specific_function %impl.elem0.loc16_46.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_46.2: <bound method> = bound_method %int_1.loc16, %specific_fn.loc16_46.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc16_46.1: init %i32 = call %bound_method.loc16_46.2(%int_1.loc16) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc16_46.2: init %i32 = converted %int_1.loc16, %int.convert_checked.loc16_46.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc16_46.3: ref %i32 = struct_access %s.var, element0
// CHECK:STDOUT:   %.loc16_46.4: init %i32 = initialize_from %.loc16_46.2 to %.loc16_46.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc16_46.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc16_46.3: <bound method> = bound_method %int_2.loc16, %impl.elem0.loc16_46.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc16_46.2: <specific function> = specific_function %impl.elem0.loc16_46.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_46.4: <bound method> = bound_method %int_2.loc16, %specific_fn.loc16_46.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc16_46.2: init %i32 = call %bound_method.loc16_46.4(%int_2.loc16) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc16_46.5: init %i32 = converted %int_2.loc16, %int.convert_checked.loc16_46.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc16_46.6: ref %i32 = struct_access %s.var, element1
// CHECK:STDOUT:   %.loc16_46.7: init %i32 = initialize_from %.loc16_46.5 to %.loc16_46.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc16_46.8: init %struct_type.a.b.501 = struct_init (%.loc16_46.4, %.loc16_46.7) to %s.var [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc16_3.1: init %struct_type.a.b.501 = converted %.loc16_46.1, %.loc16_46.8 [concrete = constants.%struct]
// CHECK:STDOUT:   assign %s.var, %.loc16_3.1
// CHECK:STDOUT:   %.loc16_27: type = splice_block %struct_type.a.b.loc16 [concrete = constants.%struct_type.a.b.501] {
// CHECK:STDOUT:     %int_32.loc16_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc16_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.loc16: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.501]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %struct_type.a.b.501 = bind_name s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.8dd = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.8dd = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.3ee = var %p.var_patt
// CHECK:STDOUT:   %s.ref.loc18: ref %struct_type.a.b.501 = name_ref s, %s
// CHECK:STDOUT:   %addr.loc18: %ptr.3ee = addr_of %s.ref.loc18
// CHECK:STDOUT:   assign %p.var, %addr.loc18
// CHECK:STDOUT:   %.loc18_28: type = splice_block %ptr.loc18 [concrete = constants.%ptr.3ee] {
// CHECK:STDOUT:     %int_32.loc18_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc18_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.loc18: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.501]
// CHECK:STDOUT:     %ptr.loc18: type = ptr_type %struct_type.a.b.loc18 [concrete = constants.%ptr.3ee]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.3ee = bind_name p, %p.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %q.patt: %pattern_type.fe8 = binding_pattern q [concrete]
// CHECK:STDOUT:     %q.var_patt: %pattern_type.fe8 = var_pattern %q.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %q.var: ref %ptr.235 = var %q.var_patt
// CHECK:STDOUT:   %s.ref.loc19: ref %struct_type.a.b.501 = name_ref s, %s
// CHECK:STDOUT:   %.loc19_19: ref %i32 = struct_access %s.ref.loc19, element0
// CHECK:STDOUT:   %addr.loc19: %ptr.235 = addr_of %.loc19_19
// CHECK:STDOUT:   assign %q.var, %addr.loc19
// CHECK:STDOUT:   %.loc19_13: type = splice_block %ptr.loc19 [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc19: type = ptr_type %i32.loc19 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %q: ref %ptr.235 = bind_name q, %q.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r.patt: %pattern_type.fe8 = binding_pattern r [concrete]
// CHECK:STDOUT:     %r.var_patt: %pattern_type.fe8 = var_pattern %r.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %r.var: ref %ptr.235 = var %r.var_patt
// CHECK:STDOUT:   %s.ref.loc20: ref %struct_type.a.b.501 = name_ref s, %s
// CHECK:STDOUT:   %.loc20_19: ref %i32 = struct_access %s.ref.loc20, element1
// CHECK:STDOUT:   %addr.loc20: %ptr.235 = addr_of %.loc20_19
// CHECK:STDOUT:   assign %r.var, %addr.loc20
// CHECK:STDOUT:   %.loc20_13: type = splice_block %ptr.loc20 [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc20: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc20: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc20: type = ptr_type %i32.loc20 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %r: ref %ptr.235 = bind_name r, %r.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type.511 = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.var_patt: %pattern_type.511 = var_pattern %t.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %tuple.type.d07 = var %t.var_patt
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc22: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc22_28.1: %tuple.type.f94 = tuple_literal (%int_1.loc22, %int_2.loc22)
// CHECK:STDOUT:   %impl.elem0.loc22_28.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc22_28.1: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_28.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc22_28.1: <specific function> = specific_function %impl.elem0.loc22_28.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_28.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22_28.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc22_28.1: init %i32 = call %bound_method.loc22_28.2(%int_1.loc22) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc22_28.2: init %i32 = converted %int_1.loc22, %int.convert_checked.loc22_28.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem0.loc22_28: ref %i32 = tuple_access %t.var, element0
// CHECK:STDOUT:   %.loc22_28.3: init %i32 = initialize_from %.loc22_28.2 to %tuple.elem0.loc22_28 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc22_28.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc22_28.3: <bound method> = bound_method %int_2.loc22, %impl.elem0.loc22_28.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc22_28.2: <specific function> = specific_function %impl.elem0.loc22_28.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_28.4: <bound method> = bound_method %int_2.loc22, %specific_fn.loc22_28.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc22_28.2: init %i32 = call %bound_method.loc22_28.4(%int_2.loc22) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc22_28.4: init %i32 = converted %int_2.loc22, %int.convert_checked.loc22_28.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.elem1.loc22_28: ref %i32 = tuple_access %t.var, element1
// CHECK:STDOUT:   %.loc22_28.5: init %i32 = initialize_from %.loc22_28.4 to %tuple.elem1.loc22_28 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc22_28.6: init %tuple.type.d07 = tuple_init (%.loc22_28.3, %.loc22_28.5) to %t.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc22_3.1: init %tuple.type.d07 = converted %.loc22_28.1, %.loc22_28.6 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %t.var, %.loc22_3.1
// CHECK:STDOUT:   %.loc22_19.1: type = splice_block %.loc22_19.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:     %int_32.loc22_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc22_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc22_19.2: %tuple.type.24b = tuple_literal (%i32.loc22_11, %i32.loc22_16)
// CHECK:STDOUT:     %.loc22_19.3: type = converted %.loc22_19.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %tuple.type.d07 = bind_name t, %t.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t0.patt: %pattern_type.fe8 = binding_pattern t0 [concrete]
// CHECK:STDOUT:     %t0.var_patt: %pattern_type.fe8 = var_pattern %t0.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t0.var: ref %ptr.235 = var %t0.var_patt
// CHECK:STDOUT:   %t.ref.loc23: ref %tuple.type.d07 = name_ref t, %t
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0.loc23: ref %i32 = tuple_access %t.ref.loc23, element0
// CHECK:STDOUT:   %addr.loc23: %ptr.235 = addr_of %tuple.elem0.loc23
// CHECK:STDOUT:   assign %t0.var, %addr.loc23
// CHECK:STDOUT:   %.loc23_14: type = splice_block %ptr.loc23 [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc23: type = ptr_type %i32.loc23 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t0: ref %ptr.235 = bind_name t0, %t0.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t1.patt: %pattern_type.fe8 = binding_pattern t1 [concrete]
// CHECK:STDOUT:     %t1.var_patt: %pattern_type.fe8 = var_pattern %t1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t1.var: ref %ptr.235 = var %t1.var_patt
// CHECK:STDOUT:   %t.ref.loc24: ref %tuple.type.d07 = name_ref t, %t
// CHECK:STDOUT:   %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc24: ref %i32 = tuple_access %t.ref.loc24, element1
// CHECK:STDOUT:   %addr.loc24: %ptr.235 = addr_of %tuple.elem1.loc24
// CHECK:STDOUT:   assign %t1.var, %addr.loc24
// CHECK:STDOUT:   %.loc24_14: type = splice_block %ptr.loc24 [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc24: type = ptr_type %i32.loc24 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t1: ref %ptr.235 = bind_name t1, %t1.var
// CHECK:STDOUT:   %impl.elem0.loc24: %.a79 = impl_witness_access constants.%Destroy.impl_witness.711, element0 [concrete = constants.%Op.649]
// CHECK:STDOUT:   %bound_method.loc24_3.1: <bound method> = bound_method %t1.var, %impl.elem0.loc24
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24, @Op.3(constants.%ptr.235) [concrete = constants.%Op.specific_fn.829]
// CHECK:STDOUT:   %bound_method.loc24_3.2: <bound method> = bound_method %t1.var, %specific_fn.loc24
// CHECK:STDOUT:   %.loc24_3: %ptr.235 = bind_value %t1.var
// CHECK:STDOUT:   %no_op.loc24: init %empty_tuple.type = call %bound_method.loc24_3.2(%.loc24_3)
// CHECK:STDOUT:   %impl.elem0.loc23: %.a79 = impl_witness_access constants.%Destroy.impl_witness.711, element0 [concrete = constants.%Op.649]
// CHECK:STDOUT:   %bound_method.loc23_3.1: <bound method> = bound_method %t0.var, %impl.elem0.loc23
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Op.3(constants.%ptr.235) [concrete = constants.%Op.specific_fn.829]
// CHECK:STDOUT:   %bound_method.loc23_3.2: <bound method> = bound_method %t0.var, %specific_fn.loc23
// CHECK:STDOUT:   %.loc23_3: %ptr.235 = bind_value %t0.var
// CHECK:STDOUT:   %no_op.loc23: init %empty_tuple.type = call %bound_method.loc23_3.2(%.loc23_3)
// CHECK:STDOUT:   %impl.elem0.loc22_3: %.b68 = impl_witness_access constants.%Destroy.impl_witness.5a7, element0 [concrete = constants.%Op.7c6]
// CHECK:STDOUT:   %bound_method.loc22_3.1: <bound method> = bound_method %t.var, %impl.elem0.loc22_3
// CHECK:STDOUT:   %specific_fn.loc22_3: <specific function> = specific_function %impl.elem0.loc22_3, @Op.3(constants.%tuple.type.d07) [concrete = constants.%Op.specific_fn.bbb]
// CHECK:STDOUT:   %bound_method.loc22_3.2: <bound method> = bound_method %t.var, %specific_fn.loc22_3
// CHECK:STDOUT:   %tuple.elem0.loc22_3: ref %i32 = tuple_access %t.var, element0
// CHECK:STDOUT:   %.loc22_3.2: %i32 = bind_value %tuple.elem0.loc22_3
// CHECK:STDOUT:   %tuple.elem1.loc22_3: ref %i32 = tuple_access %t.var, element1
// CHECK:STDOUT:   %.loc22_3.3: %i32 = bind_value %tuple.elem1.loc22_3
// CHECK:STDOUT:   %tuple: %tuple.type.d07 = tuple_value (%.loc22_3.2, %.loc22_3.3)
// CHECK:STDOUT:   %.loc22_3.4: %tuple.type.d07 = converted %t.var, %tuple
// CHECK:STDOUT:   %no_op.loc22: init %empty_tuple.type = call %bound_method.loc22_3.2(%.loc22_3.4)
// CHECK:STDOUT:   %impl.elem0.loc20: %.a79 = impl_witness_access constants.%Destroy.impl_witness.711, element0 [concrete = constants.%Op.649]
// CHECK:STDOUT:   %bound_method.loc20_3.1: <bound method> = bound_method %r.var, %impl.elem0.loc20
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Op.3(constants.%ptr.235) [concrete = constants.%Op.specific_fn.829]
// CHECK:STDOUT:   %bound_method.loc20_3.2: <bound method> = bound_method %r.var, %specific_fn.loc20
// CHECK:STDOUT:   %.loc20_3: %ptr.235 = bind_value %r.var
// CHECK:STDOUT:   %no_op.loc20: init %empty_tuple.type = call %bound_method.loc20_3.2(%.loc20_3)
// CHECK:STDOUT:   %impl.elem0.loc19: %.a79 = impl_witness_access constants.%Destroy.impl_witness.711, element0 [concrete = constants.%Op.649]
// CHECK:STDOUT:   %bound_method.loc19_3.1: <bound method> = bound_method %q.var, %impl.elem0.loc19
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @Op.3(constants.%ptr.235) [concrete = constants.%Op.specific_fn.829]
// CHECK:STDOUT:   %bound_method.loc19_3.2: <bound method> = bound_method %q.var, %specific_fn.loc19
// CHECK:STDOUT:   %.loc19_3: %ptr.235 = bind_value %q.var
// CHECK:STDOUT:   %no_op.loc19: init %empty_tuple.type = call %bound_method.loc19_3.2(%.loc19_3)
// CHECK:STDOUT:   %impl.elem0.loc18: %.f93 = impl_witness_access constants.%Destroy.impl_witness.ff6, element0 [concrete = constants.%Op.e22]
// CHECK:STDOUT:   %bound_method.loc18_3.1: <bound method> = bound_method %p.var, %impl.elem0.loc18
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Op.3(constants.%ptr.3ee) [concrete = constants.%Op.specific_fn.ce1]
// CHECK:STDOUT:   %bound_method.loc18_3.2: <bound method> = bound_method %p.var, %specific_fn.loc18
// CHECK:STDOUT:   %.loc18_3: %ptr.3ee = bind_value %p.var
// CHECK:STDOUT:   %no_op.loc18: init %empty_tuple.type = call %bound_method.loc18_3.2(%.loc18_3)
// CHECK:STDOUT:   %impl.elem0.loc16_3: %.8e0 = impl_witness_access constants.%Destroy.impl_witness.ad1, element0 [concrete = constants.%Op.9db]
// CHECK:STDOUT:   %bound_method.loc16_3.1: <bound method> = bound_method %s.var, %impl.elem0.loc16_3
// CHECK:STDOUT:   %specific_fn.loc16_3: <specific function> = specific_function %impl.elem0.loc16_3, @Op.3(constants.%struct_type.a.b.501) [concrete = constants.%Op.specific_fn.1e0]
// CHECK:STDOUT:   %bound_method.loc16_3.2: <bound method> = bound_method %s.var, %specific_fn.loc16_3
// CHECK:STDOUT:   %.loc16_3.2: ref %i32 = struct_access %s.var, element0
// CHECK:STDOUT:   %.loc16_3.3: %i32 = bind_value %.loc16_3.2
// CHECK:STDOUT:   %.loc16_3.4: ref %i32 = struct_access %s.var, element1
// CHECK:STDOUT:   %.loc16_3.5: %i32 = bind_value %.loc16_3.4
// CHECK:STDOUT:   %struct: %struct_type.a.b.501 = struct_value (%.loc16_3.3, %.loc16_3.5)
// CHECK:STDOUT:   %.loc16_3.6: %struct_type.a.b.501 = converted %s.var, %struct
// CHECK:STDOUT:   %no_op.loc16: init %empty_tuple.type = call %bound_method.loc16_3.2(%.loc16_3.6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
