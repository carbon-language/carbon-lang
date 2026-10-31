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

  var unused p: {.a: i32, .b: i32}* = &s;
  var unused q: i32* = &s.a;
  var unused r: i32* = &s.b;

  var t: (i32, i32) = (1, 2);
  var unused t0: i32* = &t.0;
  var unused t1: i32* = &t.1;
}

// CHECK:STDOUT: --- address_of_lvalue.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.250: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %ptr.aa2: type = ptr_type %struct_type.a.b.250 [concrete]
// CHECK:STDOUT:   %pattern_type.290: type = pattern_type %struct_type.a.b.250 [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.290 = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.290 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.4aa: %struct_type.a.b.cfd = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.a55: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.2e0: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %struct.940: %struct_type.a.b.250 = struct_value (%int_1.68a, %int_2.2e0) [concrete]
// CHECK:STDOUT:   %pattern_type.b62: type = pattern_type %ptr.aa2 [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.b62 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:   %p.var_patt: %pattern_type.b62 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.20f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.f12: %ptr.as.Copy.impl.Op.type.20f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ad7: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%struct_type.a.b.250) [concrete]
// CHECK:STDOUT:   %Copy.facet.6ac: %Copy.type = facet_value %ptr.aa2, (%Copy.impl_witness.ad7) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.627: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.6ac) [concrete]
// CHECK:STDOUT:   %.dab30: type = fn_type_with_self_type %Copy.WithSelf.Op.type.627, %Copy.facet.6ac [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.c1c: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%struct_type.a.b.250) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.09c: %ptr.as.Copy.impl.Op.type.c1c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.ddf: <specific function> = specific_function %ptr.as.Copy.impl.Op.09c, @ptr.as.Copy.impl.Op(%struct_type.a.b.250) [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %q.patt: %pattern_type.7dc = ref_binding_pattern q [concrete]
// CHECK:STDOUT:   %q.var_patt: %pattern_type.7dc = var_pattern %q.patt [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.83a: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Copy.facet.652: %Copy.type = facet_value %ptr.a17, (%Copy.impl_witness.83a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.9a2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.652) [concrete]
// CHECK:STDOUT:   %.309: type = fn_type_with_self_type %Copy.WithSelf.Op.type.9a2, %Copy.facet.652 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.dfc: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.81f: %ptr.as.Copy.impl.Op.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.9a6: <specific function> = specific_function %ptr.as.Copy.impl.Op.81f, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %r.patt: %pattern_type.7dc = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.var_patt: %pattern_type.7dc = var_pattern %r.patt [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.be9: %tuple.type.693 = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.35f: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.354: type = pattern_type %tuple.type.35f [concrete]
// CHECK:STDOUT:   %t.patt.b37: %pattern_type.354 = ref_binding_pattern t [concrete]
// CHECK:STDOUT:   %t.var_patt: %pattern_type.354 = var_pattern %t.patt.b37 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %tuple.380: %tuple.type.35f = tuple_value (%int_1.68a, %int_2.2e0) [concrete]
// CHECK:STDOUT:   %t0.patt: %pattern_type.7dc = ref_binding_pattern t0 [concrete]
// CHECK:STDOUT:   %t0.var_patt: %pattern_type.7dc = var_pattern %t0.patt [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %t1.patt: %pattern_type.7dc = ref_binding_pattern t1 [concrete]
// CHECK:STDOUT:   %t1.var_patt: %pattern_type.7dc = var_pattern %t1.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d4d: %SubobjectDestroy.type = facet_value %ptr.a17, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b8f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d4d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.abb: %Destroy.type = facet_value %ptr.a17, (%DestroyImpl.impl_witness.b8f, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.690: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.abb) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc22_3.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc22_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.49a: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.077: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.49a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c2b: %Destroy.type = facet_value %i32, (%DestroyImpl.impl_witness.077, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.845: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.c2b) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ec1: %SubobjectDestroy.type = facet_value %tuple.type.35f, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.df1: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ec1) [concrete]
// CHECK:STDOUT:   %Destroy.facet.58e: %Destroy.type = facet_value %tuple.type.35f, (%DestroyImpl.impl_witness.df1, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.269: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.58e) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.8c0: %SubobjectDestroy.type = facet_value %ptr.aa2, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.fe7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.8c0) [concrete]
// CHECK:STDOUT:   %Destroy.facet.75c: %Destroy.type = facet_value %ptr.aa2, (%DestroyImpl.impl_witness.fe7, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3d5: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.75c) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc16 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.582: %SubobjectDestroy.type = facet_value %struct_type.a.b.250, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6b7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.582) [concrete]
// CHECK:STDOUT:   %Destroy.facet.6bf: %Destroy.type = facet_value %struct_type.a.b.250, (%DestroyImpl.impl_witness.6b7, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.20b: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.6bf) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e474: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e474), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.06a: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.20f) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.f12)]
// CHECK:STDOUT:   %Copy.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.06a), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
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
// CHECK:STDOUT:   %s.var: ref %struct_type.a.b.250 = var_storage %s.var_patt
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc16: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc16_46.1: %struct_type.a.b.cfd = struct_literal (%int_1.loc16, %int_2.loc16) [concrete = constants.%struct.4aa]
// CHECK:STDOUT:   %impl.elem0.loc16_46.1: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc16_46.1: <bound method> = bound_method %int_1.loc16, %impl.elem0.loc16_46.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc16_46.1: <specific function> = specific_function %impl.elem0.loc16_46.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_46.2: <bound method> = bound_method %int_1.loc16, %specific_fn.loc16_46.1 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_46.1: init %i32 = call %bound_method.loc16_46.2(%int_1.loc16) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc16_46.2: init %i32 = converted %int_1.loc16, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_46.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc16_46.3: ref %i32 = struct_access %s.var, element0
// CHECK:STDOUT:   %.loc16_46.4: init %i32 to %.loc16_46.3 = in_place_init %.loc16_46.2 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc16_46.2: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc16_46.3: <bound method> = bound_method %int_2.loc16, %impl.elem0.loc16_46.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc16_46.2: <specific function> = specific_function %impl.elem0.loc16_46.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_46.4: <bound method> = bound_method %int_2.loc16, %specific_fn.loc16_46.2 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_46.2: init %i32 = call %bound_method.loc16_46.4(%int_2.loc16) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc16_46.5: init %i32 = converted %int_2.loc16, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_46.2 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc16_46.6: ref %i32 = struct_access %s.var, element1
// CHECK:STDOUT:   %.loc16_46.7: init %i32 to %.loc16_46.6 = in_place_init %.loc16_46.5 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc16_46.8: init %struct_type.a.b.250 to %s.var = struct_init (%.loc16_46.4, %.loc16_46.7) [concrete = constants.%struct.940]
// CHECK:STDOUT:   %.loc16_3.1: init %struct_type.a.b.250 = converted %.loc16_46.1, %.loc16_46.8 [concrete = constants.%struct.940]
// CHECK:STDOUT:   assign %s.var, %.loc16_3.1
// CHECK:STDOUT:   %.loc16_27: type = splice_block %struct_type.a.b.loc16 [concrete = constants.%struct_type.a.b.250] {
// CHECK:STDOUT:     %i32.loc16_15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc16_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.loc16: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.250]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %struct_type.a.b.250 = wrapper_binding s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.290 = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.290 = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.aa2 = var_storage %p.var_patt
// CHECK:STDOUT:   %s.ref.loc18: ref %struct_type.a.b.250 = name_ref s, %s
// CHECK:STDOUT:   %addr.loc18: %ptr.aa2 = addr_of %s.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18: %.dab30 = impl_witness_access constants.%Copy.impl_witness.ad7, element0 [concrete = constants.%ptr.as.Copy.impl.Op.09c]
// CHECK:STDOUT:   %bound_method.loc18_39.1: <bound method> = bound_method %addr.loc18, %impl.elem0.loc18
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @ptr.as.Copy.impl.Op(constants.%struct_type.a.b.250) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.ddf]
// CHECK:STDOUT:   %bound_method.loc18_39.2: <bound method> = bound_method %addr.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc18: init %ptr.aa2 = call %bound_method.loc18_39.2(%addr.loc18)
// CHECK:STDOUT:   assign %p.var, %ptr.as.Copy.impl.Op.call.loc18
// CHECK:STDOUT:   %.loc18_35: type = splice_block %ptr.loc18 [concrete = constants.%ptr.aa2] {
// CHECK:STDOUT:     %i32.loc18_22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc18_31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.loc18: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.250]
// CHECK:STDOUT:     %ptr.loc18: type = ptr_type %struct_type.a.b.loc18 [concrete = constants.%ptr.aa2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.aa2 = wrapper_binding p, %p.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.b62 = ref_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.b62 = var_pattern %p.patt [concrete = constants.%p.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %q.var: ref %ptr.a17 = var_storage %q.var_patt
// CHECK:STDOUT:   %s.ref.loc19: ref %struct_type.a.b.250 = name_ref s, %s
// CHECK:STDOUT:   %.loc19_26: ref %i32 = struct_access %s.ref.loc19, element0
// CHECK:STDOUT:   %addr.loc19: %ptr.a17 = addr_of %.loc19_26
// CHECK:STDOUT:   %impl.elem0.loc19: %.309 = impl_witness_access constants.%Copy.impl_witness.83a, element0 [concrete = constants.%ptr.as.Copy.impl.Op.81f]
// CHECK:STDOUT:   %bound_method.loc19_24.1: <bound method> = bound_method %addr.loc19, %impl.elem0.loc19
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.9a6]
// CHECK:STDOUT:   %bound_method.loc19_24.2: <bound method> = bound_method %addr.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc19: init %ptr.a17 = call %bound_method.loc19_24.2(%addr.loc19)
// CHECK:STDOUT:   assign %q.var, %ptr.as.Copy.impl.Op.call.loc19
// CHECK:STDOUT:   %.loc19_20: type = splice_block %ptr.loc19 [concrete = constants.%ptr.a17] {
// CHECK:STDOUT:     %i32.loc19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc19: type = ptr_type %i32.loc19 [concrete = constants.%ptr.a17]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %q: ref %ptr.a17 = wrapper_binding q, %q.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %q.patt: %pattern_type.7dc = ref_binding_pattern q [concrete = constants.%q.patt]
// CHECK:STDOUT:     %q.var_patt: %pattern_type.7dc = var_pattern %q.patt [concrete = constants.%q.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %r.var: ref %ptr.a17 = var_storage %r.var_patt
// CHECK:STDOUT:   %s.ref.loc20: ref %struct_type.a.b.250 = name_ref s, %s
// CHECK:STDOUT:   %.loc20_26: ref %i32 = struct_access %s.ref.loc20, element1
// CHECK:STDOUT:   %addr.loc20: %ptr.a17 = addr_of %.loc20_26
// CHECK:STDOUT:   %impl.elem0.loc20: %.309 = impl_witness_access constants.%Copy.impl_witness.83a, element0 [concrete = constants.%ptr.as.Copy.impl.Op.81f]
// CHECK:STDOUT:   %bound_method.loc20_24.1: <bound method> = bound_method %addr.loc20, %impl.elem0.loc20
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.9a6]
// CHECK:STDOUT:   %bound_method.loc20_24.2: <bound method> = bound_method %addr.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc20: init %ptr.a17 = call %bound_method.loc20_24.2(%addr.loc20)
// CHECK:STDOUT:   assign %r.var, %ptr.as.Copy.impl.Op.call.loc20
// CHECK:STDOUT:   %.loc20_20: type = splice_block %ptr.loc20 [concrete = constants.%ptr.a17] {
// CHECK:STDOUT:     %i32.loc20: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc20: type = ptr_type %i32.loc20 [concrete = constants.%ptr.a17]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %r: ref %ptr.a17 = wrapper_binding r, %r.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r.patt: %pattern_type.7dc = ref_binding_pattern r [concrete = constants.%r.patt]
// CHECK:STDOUT:     %r.var_patt: %pattern_type.7dc = var_pattern %r.patt [concrete = constants.%r.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %tuple.type.35f = var_storage %t.var_patt
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc22: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc22_28.1: %tuple.type.f94 = tuple_literal (%int_1.loc22, %int_2.loc22) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %impl.elem0.loc22_28.1: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc22_28.1: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_28.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc22_28.1: <specific function> = specific_function %impl.elem0.loc22_28.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_28.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22_28.1 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_28.1: init %i32 = call %bound_method.loc22_28.2(%int_1.loc22) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc22_28.2: init %i32 = converted %int_1.loc22, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_28.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %tuple.elem0.loc22: ref %i32 = tuple_access %t.var, element0
// CHECK:STDOUT:   %.loc22_28.3: init %i32 to %tuple.elem0.loc22 = in_place_init %.loc22_28.2 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc22_28.2: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc22_28.3: <bound method> = bound_method %int_2.loc22, %impl.elem0.loc22_28.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc22_28.2: <specific function> = specific_function %impl.elem0.loc22_28.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_28.4: <bound method> = bound_method %int_2.loc22, %specific_fn.loc22_28.2 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_28.2: init %i32 = call %bound_method.loc22_28.4(%int_2.loc22) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc22_28.4: init %i32 = converted %int_2.loc22, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_28.2 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %tuple.elem1.loc22: ref %i32 = tuple_access %t.var, element1
// CHECK:STDOUT:   %.loc22_28.5: init %i32 to %tuple.elem1.loc22 = in_place_init %.loc22_28.4 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc22_28.6: init %tuple.type.35f to %t.var = tuple_init (%.loc22_28.3, %.loc22_28.5) [concrete = constants.%tuple.380]
// CHECK:STDOUT:   %.loc22_3.1: init %tuple.type.35f = converted %.loc22_28.1, %.loc22_28.6 [concrete = constants.%tuple.380]
// CHECK:STDOUT:   assign %t.var, %.loc22_3.1
// CHECK:STDOUT:   %.loc22_19.1: type = splice_block %.loc22_19.3 [concrete = constants.%tuple.type.35f] {
// CHECK:STDOUT:     %i32.loc22_11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc22_16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc22_19.2: %tuple.type.693 = tuple_literal (%i32.loc22_11, %i32.loc22_16) [concrete = constants.%tuple.be9]
// CHECK:STDOUT:     %.loc22_19.3: type = converted %.loc22_19.2, constants.%tuple.type.35f [concrete = constants.%tuple.type.35f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %tuple.type.35f = wrapper_binding t, %t.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type.354 = ref_binding_pattern t [concrete = constants.%t.patt.b37]
// CHECK:STDOUT:     %t.var_patt: %pattern_type.354 = var_pattern %t.patt [concrete = constants.%t.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t0.var: ref %ptr.a17 = var_storage %t0.var_patt
// CHECK:STDOUT:   %t.ref.loc23: ref %tuple.type.35f = name_ref t, %t
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0.loc23: ref %i32 = tuple_access %t.ref.loc23, element0
// CHECK:STDOUT:   %addr.loc23: %ptr.a17 = addr_of %tuple.elem0.loc23
// CHECK:STDOUT:   %impl.elem0.loc23: %.309 = impl_witness_access constants.%Copy.impl_witness.83a, element0 [concrete = constants.%ptr.as.Copy.impl.Op.81f]
// CHECK:STDOUT:   %bound_method.loc23_25.1: <bound method> = bound_method %addr.loc23, %impl.elem0.loc23
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.9a6]
// CHECK:STDOUT:   %bound_method.loc23_25.2: <bound method> = bound_method %addr.loc23, %specific_fn.loc23
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc23: init %ptr.a17 = call %bound_method.loc23_25.2(%addr.loc23)
// CHECK:STDOUT:   assign %t0.var, %ptr.as.Copy.impl.Op.call.loc23
// CHECK:STDOUT:   %.loc23_21: type = splice_block %ptr.loc23 [concrete = constants.%ptr.a17] {
// CHECK:STDOUT:     %i32.loc23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc23: type = ptr_type %i32.loc23 [concrete = constants.%ptr.a17]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t0: ref %ptr.a17 = wrapper_binding t0, %t0.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t0.patt: %pattern_type.7dc = ref_binding_pattern t0 [concrete = constants.%t0.patt]
// CHECK:STDOUT:     %t0.var_patt: %pattern_type.7dc = var_pattern %t0.patt [concrete = constants.%t0.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t1.var: ref %ptr.a17 = var_storage %t1.var_patt
// CHECK:STDOUT:   %t.ref.loc24: ref %tuple.type.35f = name_ref t, %t
// CHECK:STDOUT:   %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc24: ref %i32 = tuple_access %t.ref.loc24, element1
// CHECK:STDOUT:   %addr.loc24: %ptr.a17 = addr_of %tuple.elem1.loc24
// CHECK:STDOUT:   %impl.elem0.loc24: %.309 = impl_witness_access constants.%Copy.impl_witness.83a, element0 [concrete = constants.%ptr.as.Copy.impl.Op.81f]
// CHECK:STDOUT:   %bound_method.loc24_25.1: <bound method> = bound_method %addr.loc24, %impl.elem0.loc24
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.9a6]
// CHECK:STDOUT:   %bound_method.loc24_25.2: <bound method> = bound_method %addr.loc24, %specific_fn.loc24
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc24: init %ptr.a17 = call %bound_method.loc24_25.2(%addr.loc24)
// CHECK:STDOUT:   assign %t1.var, %ptr.as.Copy.impl.Op.call.loc24
// CHECK:STDOUT:   %.loc24_21: type = splice_block %ptr.loc24 [concrete = constants.%ptr.a17] {
// CHECK:STDOUT:     %i32.loc24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc24: type = ptr_type %i32.loc24 [concrete = constants.%ptr.a17]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t1: ref %ptr.a17 = wrapper_binding t1, %t1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t1.patt: %pattern_type.7dc = ref_binding_pattern t1 [concrete = constants.%t1.patt]
// CHECK:STDOUT:     %t1.var_patt: %pattern_type.7dc = var_pattern %t1.patt [concrete = constants.%t1.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc24_3.1: %Destroy.type = facet_value constants.%ptr.a17, (constants.%DestroyImpl.impl_witness.b8f, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %.loc24_3.1: %Destroy.type = converted constants.%ptr.a17, %Destroy.facet.loc24_3.1 [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %Destroy.facet.loc24_3.2: %Destroy.type = facet_value constants.%ptr.a17, (constants.%DestroyImpl.impl_witness.b8f, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %.loc24_3.2: %Destroy.type = converted constants.%ptr.a17, %Destroy.facet.loc24_3.2 [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.abb) [concrete = constants.%SelfDestruct.specific_fn.690]
// CHECK:STDOUT:   %SelfDestruct.call.loc24: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%t1.var)
// CHECK:STDOUT:   %Destroy.facet.loc23_3.1: %Destroy.type = facet_value constants.%ptr.a17, (constants.%DestroyImpl.impl_witness.b8f, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %.loc23_3.1: %Destroy.type = converted constants.%ptr.a17, %Destroy.facet.loc23_3.1 [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %Destroy.facet.loc23_3.2: %Destroy.type = facet_value constants.%ptr.a17, (constants.%DestroyImpl.impl_witness.b8f, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %.loc23_3.2: %Destroy.type = converted constants.%ptr.a17, %Destroy.facet.loc23_3.2 [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.abb) [concrete = constants.%SelfDestruct.specific_fn.690]
// CHECK:STDOUT:   %SelfDestruct.call.loc23: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%t0.var)
// CHECK:STDOUT:   %Destroy.facet.loc22_3.1: %Destroy.type = facet_value constants.%tuple.type.35f, (constants.%DestroyImpl.impl_witness.df1, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.58e]
// CHECK:STDOUT:   %.loc22_3.2: %Destroy.type = converted constants.%tuple.type.35f, %Destroy.facet.loc22_3.1 [concrete = constants.%Destroy.facet.58e]
// CHECK:STDOUT:   %Destroy.facet.loc22_3.2: %Destroy.type = facet_value constants.%tuple.type.35f, (constants.%DestroyImpl.impl_witness.df1, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.58e]
// CHECK:STDOUT:   %.loc22_3.3: %Destroy.type = converted constants.%tuple.type.35f, %Destroy.facet.loc22_3.2 [concrete = constants.%Destroy.facet.58e]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.58e) [concrete = constants.%SelfDestruct.specific_fn.269]
// CHECK:STDOUT:   %SelfDestruct.call.loc22: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%t.var)
// CHECK:STDOUT:   %Destroy.facet.loc20_3.1: %Destroy.type = facet_value constants.%ptr.a17, (constants.%DestroyImpl.impl_witness.b8f, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %.loc20_3.1: %Destroy.type = converted constants.%ptr.a17, %Destroy.facet.loc20_3.1 [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %Destroy.facet.loc20_3.2: %Destroy.type = facet_value constants.%ptr.a17, (constants.%DestroyImpl.impl_witness.b8f, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %.loc20_3.2: %Destroy.type = converted constants.%ptr.a17, %Destroy.facet.loc20_3.2 [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.4: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.abb) [concrete = constants.%SelfDestruct.specific_fn.690]
// CHECK:STDOUT:   %SelfDestruct.call.loc20: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%r.var)
// CHECK:STDOUT:   %Destroy.facet.loc19_3.1: %Destroy.type = facet_value constants.%ptr.a17, (constants.%DestroyImpl.impl_witness.b8f, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %.loc19_3.1: %Destroy.type = converted constants.%ptr.a17, %Destroy.facet.loc19_3.1 [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %Destroy.facet.loc19_3.2: %Destroy.type = facet_value constants.%ptr.a17, (constants.%DestroyImpl.impl_witness.b8f, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %.loc19_3.2: %Destroy.type = converted constants.%ptr.a17, %Destroy.facet.loc19_3.2 [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.abb) [concrete = constants.%SelfDestruct.specific_fn.690]
// CHECK:STDOUT:   %SelfDestruct.call.loc19: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%q.var)
// CHECK:STDOUT:   %Destroy.facet.loc18_3.1: %Destroy.type = facet_value constants.%ptr.aa2, (constants.%DestroyImpl.impl_witness.fe7, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.75c]
// CHECK:STDOUT:   %.loc18_3.1: %Destroy.type = converted constants.%ptr.aa2, %Destroy.facet.loc18_3.1 [concrete = constants.%Destroy.facet.75c]
// CHECK:STDOUT:   %Destroy.facet.loc18_3.2: %Destroy.type = facet_value constants.%ptr.aa2, (constants.%DestroyImpl.impl_witness.fe7, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.75c]
// CHECK:STDOUT:   %.loc18_3.2: %Destroy.type = converted constants.%ptr.aa2, %Destroy.facet.loc18_3.2 [concrete = constants.%Destroy.facet.75c]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.75c) [concrete = constants.%SelfDestruct.specific_fn.3d5]
// CHECK:STDOUT:   %SelfDestruct.call.loc18: init %empty_tuple.type = call %SelfDestruct.specific_fn.6(%p.var)
// CHECK:STDOUT:   %Destroy.facet.loc16_3.1: %Destroy.type = facet_value constants.%struct_type.a.b.250, (constants.%DestroyImpl.impl_witness.6b7, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.6bf]
// CHECK:STDOUT:   %.loc16_3.2: %Destroy.type = converted constants.%struct_type.a.b.250, %Destroy.facet.loc16_3.1 [concrete = constants.%Destroy.facet.6bf]
// CHECK:STDOUT:   %Destroy.facet.loc16_3.2: %Destroy.type = facet_value constants.%struct_type.a.b.250, (constants.%DestroyImpl.impl_witness.6b7, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.6bf]
// CHECK:STDOUT:   %.loc16_3.3: %Destroy.type = converted constants.%struct_type.a.b.250, %Destroy.facet.loc16_3.2 [concrete = constants.%Destroy.facet.6bf]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.7: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.6bf) [concrete = constants.%SelfDestruct.specific_fn.20b]
// CHECK:STDOUT:   %SelfDestruct.call.loc16: init %empty_tuple.type = call %SelfDestruct.specific_fn.7(%s.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24(%self.param: ref %ptr.a17) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc22_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc22_3.2(%self.param: ref %tuple.type.35f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc22_3.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc22_3.1: %Destroy.type = converted constants.%i32, %Destroy.facet.loc22_3.1 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc22_3.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc22_3.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc22_3.2 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.c2b) [concrete = constants.%SelfDestruct.specific_fn.845]
// CHECK:STDOUT:   %SelfDestruct.call.loc22_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc22_3.3: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc22_3.3: %Destroy.type = converted constants.%i32, %Destroy.facet.loc22_3.3 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc22_3.4: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc22_3.4: %Destroy.type = converted constants.%i32, %Destroy.facet.loc22_3.4 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.c2b) [concrete = constants.%SelfDestruct.specific_fn.845]
// CHECK:STDOUT:   %SelfDestruct.call.loc22_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18(%self.param: ref %ptr.aa2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16(%self.param: ref %struct_type.a.b.250) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_3.1: ref %i32 = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc16_3.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc16_3.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc16_3.1 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc16_3.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc16_3.3: %Destroy.type = converted constants.%i32, %Destroy.facet.loc16_3.2 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.c2b) [concrete = constants.%SelfDestruct.specific_fn.845]
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc16_3.1)
// CHECK:STDOUT:   %.loc16_3.4: ref %i32 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc16_3.3: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc16_3.5: %Destroy.type = converted constants.%i32, %Destroy.facet.loc16_3.3 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc16_3.4: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc16_3.6: %Destroy.type = converted constants.%i32, %Destroy.facet.loc16_3.4 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.c2b) [concrete = constants.%SelfDestruct.specific_fn.845]
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc16_3.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
