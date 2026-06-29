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
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.b4c: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %ptr.939: type = ptr_type %struct_type.a.b.b4c [concrete]
// CHECK:STDOUT:   %pattern_type.20e3: type = pattern_type %struct_type.a.b.b4c [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.20e3 = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.20e3 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.4aa: %struct_type.a.b.cfd = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %struct.631: %struct_type.a.b.b4c = struct_value (%int_1.0c6, %int_2.295) [concrete]
// CHECK:STDOUT:   %pattern_type.88d: type = pattern_type %ptr.939 [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.88d = ref_binding_pattern p [concrete]
// CHECK:STDOUT:   %p.var_patt: %pattern_type.88d = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.209: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%struct_type.a.b.b4c) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.dbe: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%struct_type.a.b.b4c) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.287: %ptr.as.Copy.impl.Op.type.dbe = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.2c4: %Copy.type = facet_value %ptr.939, (%Copy.impl_witness.209) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.4f7: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.2c4) [concrete]
// CHECK:STDOUT:   %.bc9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.4f7, %Copy.facet.2c4 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.867: <specific function> = specific_function %ptr.as.Copy.impl.Op.287, @ptr.as.Copy.impl.Op(%struct_type.a.b.b4c) [concrete]
// CHECK:STDOUT:   %ptr.d08: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.f00: type = pattern_type %ptr.d08 [concrete]
// CHECK:STDOUT:   %q.patt: %pattern_type.f00 = ref_binding_pattern q [concrete]
// CHECK:STDOUT:   %q.var_patt: %pattern_type.f00 = var_pattern %q.patt [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.ce1: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.c1c: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.d1c: %ptr.as.Copy.impl.Op.type.c1c = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.b49: %Copy.type = facet_value %ptr.d08, (%Copy.impl_witness.ce1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f9d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.b49) [concrete]
// CHECK:STDOUT:   %.ac9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f9d, %Copy.facet.b49 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.b3e: <specific function> = specific_function %ptr.as.Copy.impl.Op.d1c, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %r.patt: %pattern_type.f00 = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.var_patt: %pattern_type.f00 = var_pattern %r.patt [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.b59: %tuple.type.24b = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.e55: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.394: type = pattern_type %tuple.type.e55 [concrete]
// CHECK:STDOUT:   %t.patt: %pattern_type.394 = ref_binding_pattern t [concrete]
// CHECK:STDOUT:   %t.var_patt: %pattern_type.394 = var_pattern %t.patt [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %tuple.102: %tuple.type.e55 = tuple_value (%int_1.0c6, %int_2.295) [concrete]
// CHECK:STDOUT:   %t0.patt: %pattern_type.f00 = ref_binding_pattern t0 [concrete]
// CHECK:STDOUT:   %t0.var_patt: %pattern_type.f00 = var_pattern %t0.patt [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %t1.patt: %pattern_type.f00 = ref_binding_pattern t1 [concrete]
// CHECK:STDOUT:   %t1.var_patt: %pattern_type.f00 = var_pattern %t1.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.loc24 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc22_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.5: type = fn_type @Destroy.Op.loc18 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.5: %Destroy.Op.type.1d8f74.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.6: type = fn_type @Destroy.Op.loc16 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.6: %Destroy.Op.type.1d8f74.6 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4f1 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %s.patt: %pattern_type.20e3 = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.20e3 = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %struct_type.a.b.b4c = var %s.var_patt
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc16: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc16_46.1: %struct_type.a.b.cfd = struct_literal (%int_1.loc16, %int_2.loc16) [concrete = constants.%struct.4aa]
// CHECK:STDOUT:   %impl.elem0.loc16_46.1: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc16_46.1: <bound method> = bound_method %int_1.loc16, %impl.elem0.loc16_46.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc16_46.1: <specific function> = specific_function %impl.elem0.loc16_46.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_46.2: <bound method> = bound_method %int_1.loc16, %specific_fn.loc16_46.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_46.1: init %i32 = call %bound_method.loc16_46.2(%int_1.loc16) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc16_46.2: init %i32 = converted %int_1.loc16, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_46.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc16_46.3: ref %i32 = struct_access %s.var, element0
// CHECK:STDOUT:   %.loc16_46.4: init %i32 to %.loc16_46.3 = in_place_init %.loc16_46.2 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc16_46.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc16_46.3: <bound method> = bound_method %int_2.loc16, %impl.elem0.loc16_46.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc16_46.2: <specific function> = specific_function %impl.elem0.loc16_46.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_46.4: <bound method> = bound_method %int_2.loc16, %specific_fn.loc16_46.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_46.2: init %i32 = call %bound_method.loc16_46.4(%int_2.loc16) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc16_46.5: init %i32 = converted %int_2.loc16, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_46.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc16_46.6: ref %i32 = struct_access %s.var, element1
// CHECK:STDOUT:   %.loc16_46.7: init %i32 to %.loc16_46.6 = in_place_init %.loc16_46.5 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc16_46.8: init %struct_type.a.b.b4c to %s.var = struct_init (%.loc16_46.4, %.loc16_46.7) [concrete = constants.%struct.631]
// CHECK:STDOUT:   %.loc16_3: init %struct_type.a.b.b4c = converted %.loc16_46.1, %.loc16_46.8 [concrete = constants.%struct.631]
// CHECK:STDOUT:   assign %s.var, %.loc16_3
// CHECK:STDOUT:   %.loc16_27: type = splice_block %struct_type.a.b.loc16 [concrete = constants.%struct_type.a.b.b4c] {
// CHECK:STDOUT:     %i32.loc16_15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc16_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.loc16: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.b4c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %struct_type.a.b.b4c = wrapper_binding s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.88d = ref_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.88d = var_pattern %p.patt [concrete = constants.%p.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.939 = var %p.var_patt
// CHECK:STDOUT:   %s.ref.loc18: ref %struct_type.a.b.b4c = name_ref s, %s
// CHECK:STDOUT:   %addr.loc18: %ptr.939 = addr_of %s.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18: %.bc9 = impl_witness_access constants.%Copy.impl_witness.209, element0 [concrete = constants.%ptr.as.Copy.impl.Op.287]
// CHECK:STDOUT:   %bound_method.loc18_39.1: <bound method> = bound_method %addr.loc18, %impl.elem0.loc18
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @ptr.as.Copy.impl.Op(constants.%struct_type.a.b.b4c) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.867]
// CHECK:STDOUT:   %bound_method.loc18_39.2: <bound method> = bound_method %addr.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc18: init %ptr.939 = call %bound_method.loc18_39.2(%addr.loc18)
// CHECK:STDOUT:   assign %p.var, %ptr.as.Copy.impl.Op.call.loc18
// CHECK:STDOUT:   %.loc18: type = splice_block %ptr.loc18 [concrete = constants.%ptr.939] {
// CHECK:STDOUT:     %i32.loc18_22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc18_31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.loc18: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.b4c]
// CHECK:STDOUT:     %ptr.loc18: type = ptr_type %struct_type.a.b.loc18 [concrete = constants.%ptr.939]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.939 = wrapper_binding p, %p.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %q.patt: %pattern_type.f00 = ref_binding_pattern q [concrete = constants.%q.patt]
// CHECK:STDOUT:     %q.var_patt: %pattern_type.f00 = var_pattern %q.patt [concrete = constants.%q.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %q.var: ref %ptr.d08 = var %q.var_patt
// CHECK:STDOUT:   %s.ref.loc19: ref %struct_type.a.b.b4c = name_ref s, %s
// CHECK:STDOUT:   %.loc19_26: ref %i32 = struct_access %s.ref.loc19, element0
// CHECK:STDOUT:   %addr.loc19: %ptr.d08 = addr_of %.loc19_26
// CHECK:STDOUT:   %impl.elem0.loc19: %.ac9 = impl_witness_access constants.%Copy.impl_witness.ce1, element0 [concrete = constants.%ptr.as.Copy.impl.Op.d1c]
// CHECK:STDOUT:   %bound_method.loc19_24.1: <bound method> = bound_method %addr.loc19, %impl.elem0.loc19
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.b3e]
// CHECK:STDOUT:   %bound_method.loc19_24.2: <bound method> = bound_method %addr.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc19: init %ptr.d08 = call %bound_method.loc19_24.2(%addr.loc19)
// CHECK:STDOUT:   assign %q.var, %ptr.as.Copy.impl.Op.call.loc19
// CHECK:STDOUT:   %.loc19_20: type = splice_block %ptr.loc19 [concrete = constants.%ptr.d08] {
// CHECK:STDOUT:     %i32.loc19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc19: type = ptr_type %i32.loc19 [concrete = constants.%ptr.d08]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %q: ref %ptr.d08 = wrapper_binding q, %q.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r.patt: %pattern_type.f00 = ref_binding_pattern r [concrete = constants.%r.patt]
// CHECK:STDOUT:     %r.var_patt: %pattern_type.f00 = var_pattern %r.patt [concrete = constants.%r.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %r.var: ref %ptr.d08 = var %r.var_patt
// CHECK:STDOUT:   %s.ref.loc20: ref %struct_type.a.b.b4c = name_ref s, %s
// CHECK:STDOUT:   %.loc20_26: ref %i32 = struct_access %s.ref.loc20, element1
// CHECK:STDOUT:   %addr.loc20: %ptr.d08 = addr_of %.loc20_26
// CHECK:STDOUT:   %impl.elem0.loc20: %.ac9 = impl_witness_access constants.%Copy.impl_witness.ce1, element0 [concrete = constants.%ptr.as.Copy.impl.Op.d1c]
// CHECK:STDOUT:   %bound_method.loc20_24.1: <bound method> = bound_method %addr.loc20, %impl.elem0.loc20
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.b3e]
// CHECK:STDOUT:   %bound_method.loc20_24.2: <bound method> = bound_method %addr.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc20: init %ptr.d08 = call %bound_method.loc20_24.2(%addr.loc20)
// CHECK:STDOUT:   assign %r.var, %ptr.as.Copy.impl.Op.call.loc20
// CHECK:STDOUT:   %.loc20_20: type = splice_block %ptr.loc20 [concrete = constants.%ptr.d08] {
// CHECK:STDOUT:     %i32.loc20: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc20: type = ptr_type %i32.loc20 [concrete = constants.%ptr.d08]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %r: ref %ptr.d08 = wrapper_binding r, %r.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type.394 = ref_binding_pattern t [concrete = constants.%t.patt]
// CHECK:STDOUT:     %t.var_patt: %pattern_type.394 = var_pattern %t.patt [concrete = constants.%t.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %tuple.type.e55 = var %t.var_patt
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc22: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc22_28.1: %tuple.type.f94 = tuple_literal (%int_1.loc22, %int_2.loc22) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %impl.elem0.loc22_28.1: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc22_28.1: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_28.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc22_28.1: <specific function> = specific_function %impl.elem0.loc22_28.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_28.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22_28.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_28.1: init %i32 = call %bound_method.loc22_28.2(%int_1.loc22) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc22_28.2: init %i32 = converted %int_1.loc22, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_28.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %tuple.elem0.loc22: ref %i32 = tuple_access %t.var, element0
// CHECK:STDOUT:   %.loc22_28.3: init %i32 to %tuple.elem0.loc22 = in_place_init %.loc22_28.2 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc22_28.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc22_28.3: <bound method> = bound_method %int_2.loc22, %impl.elem0.loc22_28.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc22_28.2: <specific function> = specific_function %impl.elem0.loc22_28.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_28.4: <bound method> = bound_method %int_2.loc22, %specific_fn.loc22_28.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_28.2: init %i32 = call %bound_method.loc22_28.4(%int_2.loc22) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc22_28.4: init %i32 = converted %int_2.loc22, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_28.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %tuple.elem1.loc22: ref %i32 = tuple_access %t.var, element1
// CHECK:STDOUT:   %.loc22_28.5: init %i32 to %tuple.elem1.loc22 = in_place_init %.loc22_28.4 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc22_28.6: init %tuple.type.e55 to %t.var = tuple_init (%.loc22_28.3, %.loc22_28.5) [concrete = constants.%tuple.102]
// CHECK:STDOUT:   %.loc22_3: init %tuple.type.e55 = converted %.loc22_28.1, %.loc22_28.6 [concrete = constants.%tuple.102]
// CHECK:STDOUT:   assign %t.var, %.loc22_3
// CHECK:STDOUT:   %.loc22_19.1: type = splice_block %.loc22_19.3 [concrete = constants.%tuple.type.e55] {
// CHECK:STDOUT:     %i32.loc22_11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc22_16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc22_19.2: %tuple.type.24b = tuple_literal (%i32.loc22_11, %i32.loc22_16) [concrete = constants.%tuple.b59]
// CHECK:STDOUT:     %.loc22_19.3: type = converted %.loc22_19.2, constants.%tuple.type.e55 [concrete = constants.%tuple.type.e55]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %tuple.type.e55 = wrapper_binding t, %t.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t0.patt: %pattern_type.f00 = ref_binding_pattern t0 [concrete = constants.%t0.patt]
// CHECK:STDOUT:     %t0.var_patt: %pattern_type.f00 = var_pattern %t0.patt [concrete = constants.%t0.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t0.var: ref %ptr.d08 = var %t0.var_patt
// CHECK:STDOUT:   %t.ref.loc23: ref %tuple.type.e55 = name_ref t, %t
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0.loc23: ref %i32 = tuple_access %t.ref.loc23, element0
// CHECK:STDOUT:   %addr.loc23: %ptr.d08 = addr_of %tuple.elem0.loc23
// CHECK:STDOUT:   %impl.elem0.loc23: %.ac9 = impl_witness_access constants.%Copy.impl_witness.ce1, element0 [concrete = constants.%ptr.as.Copy.impl.Op.d1c]
// CHECK:STDOUT:   %bound_method.loc23_25.1: <bound method> = bound_method %addr.loc23, %impl.elem0.loc23
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.b3e]
// CHECK:STDOUT:   %bound_method.loc23_25.2: <bound method> = bound_method %addr.loc23, %specific_fn.loc23
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc23: init %ptr.d08 = call %bound_method.loc23_25.2(%addr.loc23)
// CHECK:STDOUT:   assign %t0.var, %ptr.as.Copy.impl.Op.call.loc23
// CHECK:STDOUT:   %.loc23: type = splice_block %ptr.loc23 [concrete = constants.%ptr.d08] {
// CHECK:STDOUT:     %i32.loc23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc23: type = ptr_type %i32.loc23 [concrete = constants.%ptr.d08]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t0: ref %ptr.d08 = wrapper_binding t0, %t0.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t1.patt: %pattern_type.f00 = ref_binding_pattern t1 [concrete = constants.%t1.patt]
// CHECK:STDOUT:     %t1.var_patt: %pattern_type.f00 = var_pattern %t1.patt [concrete = constants.%t1.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t1.var: ref %ptr.d08 = var %t1.var_patt
// CHECK:STDOUT:   %t.ref.loc24: ref %tuple.type.e55 = name_ref t, %t
// CHECK:STDOUT:   %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc24: ref %i32 = tuple_access %t.ref.loc24, element1
// CHECK:STDOUT:   %addr.loc24: %ptr.d08 = addr_of %tuple.elem1.loc24
// CHECK:STDOUT:   %impl.elem0.loc24: %.ac9 = impl_witness_access constants.%Copy.impl_witness.ce1, element0 [concrete = constants.%ptr.as.Copy.impl.Op.d1c]
// CHECK:STDOUT:   %bound_method.loc24_25.1: <bound method> = bound_method %addr.loc24, %impl.elem0.loc24
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.b3e]
// CHECK:STDOUT:   %bound_method.loc24_25.2: <bound method> = bound_method %addr.loc24, %specific_fn.loc24
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc24: init %ptr.d08 = call %bound_method.loc24_25.2(%addr.loc24)
// CHECK:STDOUT:   assign %t1.var, %ptr.as.Copy.impl.Op.call.loc24
// CHECK:STDOUT:   %.loc24: type = splice_block %ptr.loc24 [concrete = constants.%ptr.d08] {
// CHECK:STDOUT:     %i32.loc24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc24: type = ptr_type %i32.loc24 [concrete = constants.%ptr.d08]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t1: ref %ptr.d08 = wrapper_binding t1, %t1.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc24: <bound method> = bound_method %t1.var, constants.%Destroy.Op.1a2547.1
// CHECK:STDOUT:   %Destroy.Op.call.loc24: init %empty_tuple.type = call %Destroy.Op.bound.loc24(%t1.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc23: <bound method> = bound_method %t0.var, constants.%Destroy.Op.1a2547.1
// CHECK:STDOUT:   %Destroy.Op.call.loc23: init %empty_tuple.type = call %Destroy.Op.bound.loc23(%t0.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc22: <bound method> = bound_method %t.var, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call.loc22: init %empty_tuple.type = call %Destroy.Op.bound.loc22(%t.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc20: <bound method> = bound_method %r.var, constants.%Destroy.Op.1a2547.1
// CHECK:STDOUT:   %Destroy.Op.call.loc20: init %empty_tuple.type = call %Destroy.Op.bound.loc20(%r.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc19: <bound method> = bound_method %q.var, constants.%Destroy.Op.1a2547.1
// CHECK:STDOUT:   %Destroy.Op.call.loc19: init %empty_tuple.type = call %Destroy.Op.bound.loc19(%q.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc18: <bound method> = bound_method %p.var, constants.%Destroy.Op.1a2547.5
// CHECK:STDOUT:   %Destroy.Op.call.loc18: init %empty_tuple.type = call %Destroy.Op.bound.loc18(%p.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc16: <bound method> = bound_method %s.var, constants.%Destroy.Op.1a2547.6
// CHECK:STDOUT:   %Destroy.Op.call.loc16: init %empty_tuple.type = call %Destroy.Op.bound.loc16(%s.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24(%self.param: ref %ptr.d08) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc22_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc22_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc22_3.3(%self.param: ref %tuple.type.e55) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18(%self.param: ref %ptr.939) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc16(%self.param: ref %struct_type.a.b.b4c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
