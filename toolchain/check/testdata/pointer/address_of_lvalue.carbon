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
// CHECK:STDOUT:   %struct_type.a.b.602: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %ptr.148: type = ptr_type %struct_type.a.b.602 [concrete]
// CHECK:STDOUT:   %pattern_type.32c: type = pattern_type %struct_type.a.b.602 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.4aa: %struct_type.a.b.cfd = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.33d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d5e: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.22e: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.cc3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.4ff: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %struct.a0d: %struct_type.a.b.602 = struct_value (%int_1.d5e, %int_2.4ff) [concrete]
// CHECK:STDOUT:   %pattern_type.d1b: type = pattern_type %ptr.148 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.b03: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.aed: %ptr.as.Copy.impl.Op.type.b03 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1d8: <witness> = impl_witness imports.%Copy.impl_witness_table.760, @ptr.as.Copy.impl(%struct_type.a.b.602) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.f07: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%struct_type.a.b.602) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.2d9: %ptr.as.Copy.impl.Op.type.f07 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.803: %Copy.type = facet_value %ptr.148, (%Copy.impl_witness.1d8) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a1f: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.803) [concrete]
// CHECK:STDOUT:   %.8d4: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a1f, %Copy.facet.803 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.f36: <specific function> = specific_function %ptr.as.Copy.impl.Op.2d9, @ptr.as.Copy.impl.Op(%struct_type.a.b.602) [concrete]
// CHECK:STDOUT:   %ptr.143: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.663: type = pattern_type %ptr.143 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.45b: <witness> = impl_witness imports.%Copy.impl_witness_table.760, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.eac: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.6e2: %ptr.as.Copy.impl.Op.type.eac = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.425: %Copy.type = facet_value %ptr.143, (%Copy.impl_witness.45b) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.72c: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.425) [concrete]
// CHECK:STDOUT:   %.948: type = fn_type_with_self_type %Copy.WithSelf.Op.type.72c, %Copy.facet.425 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.fe2: <specific function> = specific_function %ptr.as.Copy.impl.Op.6e2, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.8e3: %tuple.type.24b = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.87c: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.624: type = pattern_type %tuple.type.87c [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %tuple.03f: %tuple.type.87c = tuple_value (%int_1.d5e, %int_2.4ff) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.1: type = fn_type @Destroy.Op.loc24 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.1: %Destroy.Op.type.af7ec0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.4: type = fn_type @Destroy.Op.loc22_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.4: %Destroy.Op.type.af7ec0.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.5: type = fn_type @Destroy.Op.loc18 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.5: %Destroy.Op.type.af7ec0.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.6: type = fn_type @Destroy.Op.loc16 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.6: %Destroy.Op.type.af7ec0.6 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.13c: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.b03) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.aed)]
// CHECK:STDOUT:   %Copy.impl_witness_table.760 = impl_witness_table (%Core.import_ref.13c), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %s.patt: %pattern_type.32c = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.32c = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %struct_type.a.b.602 = var %s.var_patt
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc16: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc16_46.1: %struct_type.a.b.cfd = struct_literal (%int_1.loc16, %int_2.loc16) [concrete = constants.%struct.4aa]
// CHECK:STDOUT:   %impl.elem0.loc16_46.1: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc16_46.1: <bound method> = bound_method %int_1.loc16, %impl.elem0.loc16_46.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d]
// CHECK:STDOUT:   %specific_fn.loc16_46.1: <specific function> = specific_function %impl.elem0.loc16_46.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_46.2: <bound method> = bound_method %int_1.loc16, %specific_fn.loc16_46.1 [concrete = constants.%bound_method.33d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_46.1: init %i32 = call %bound_method.loc16_46.2(%int_1.loc16) [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc16_46.2: init %i32 = converted %int_1.loc16, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_46.1 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc16_46.3: ref %i32 = struct_access %s.var, element0
// CHECK:STDOUT:   %.loc16_46.4: init %i32 to %.loc16_46.3 = in_place_init %.loc16_46.2 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %impl.elem0.loc16_46.2: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc16_46.3: <bound method> = bound_method %int_2.loc16, %impl.elem0.loc16_46.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.22e]
// CHECK:STDOUT:   %specific_fn.loc16_46.2: <specific function> = specific_function %impl.elem0.loc16_46.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_46.4: <bound method> = bound_method %int_2.loc16, %specific_fn.loc16_46.2 [concrete = constants.%bound_method.cc3]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_46.2: init %i32 = call %bound_method.loc16_46.4(%int_2.loc16) [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc16_46.5: init %i32 = converted %int_2.loc16, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_46.2 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc16_46.6: ref %i32 = struct_access %s.var, element1
// CHECK:STDOUT:   %.loc16_46.7: init %i32 to %.loc16_46.6 = in_place_init %.loc16_46.5 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc16_46.8: init %struct_type.a.b.602 to %s.var = struct_init (%.loc16_46.4, %.loc16_46.7) [concrete = constants.%struct.a0d]
// CHECK:STDOUT:   %.loc16_3: init %struct_type.a.b.602 = converted %.loc16_46.1, %.loc16_46.8 [concrete = constants.%struct.a0d]
// CHECK:STDOUT:   assign %s.var, %.loc16_3
// CHECK:STDOUT:   %.loc16_27: type = splice_block %struct_type.a.b.loc16 [concrete = constants.%struct_type.a.b.602] {
// CHECK:STDOUT:     %i32.loc16_15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc16_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.loc16: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.602]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %struct_type.a.b.602 = ref_binding s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.d1b = ref_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.d1b = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.148 = var %p.var_patt
// CHECK:STDOUT:   %s.ref.loc18: ref %struct_type.a.b.602 = name_ref s, %s
// CHECK:STDOUT:   %addr.loc18: %ptr.148 = addr_of %s.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18: %.8d4 = impl_witness_access constants.%Copy.impl_witness.1d8, element0 [concrete = constants.%ptr.as.Copy.impl.Op.2d9]
// CHECK:STDOUT:   %bound_method.loc18_39.1: <bound method> = bound_method %addr.loc18, %impl.elem0.loc18
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @ptr.as.Copy.impl.Op(constants.%struct_type.a.b.602) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.f36]
// CHECK:STDOUT:   %bound_method.loc18_39.2: <bound method> = bound_method %addr.loc18, %specific_fn.loc18
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc18: init %ptr.148 = call %bound_method.loc18_39.2(%addr.loc18)
// CHECK:STDOUT:   assign %p.var, %ptr.as.Copy.impl.Op.call.loc18
// CHECK:STDOUT:   %.loc18: type = splice_block %ptr.loc18 [concrete = constants.%ptr.148] {
// CHECK:STDOUT:     %i32.loc18_22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc18_31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.loc18: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.602]
// CHECK:STDOUT:     %ptr.loc18: type = ptr_type %struct_type.a.b.loc18 [concrete = constants.%ptr.148]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.148 = ref_binding p, %p.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %q.patt: %pattern_type.663 = ref_binding_pattern q [concrete]
// CHECK:STDOUT:     %q.var_patt: %pattern_type.663 = var_pattern %q.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %q.var: ref %ptr.143 = var %q.var_patt
// CHECK:STDOUT:   %s.ref.loc19: ref %struct_type.a.b.602 = name_ref s, %s
// CHECK:STDOUT:   %.loc19_26: ref %i32 = struct_access %s.ref.loc19, element0
// CHECK:STDOUT:   %addr.loc19: %ptr.143 = addr_of %.loc19_26
// CHECK:STDOUT:   %impl.elem0.loc19: %.948 = impl_witness_access constants.%Copy.impl_witness.45b, element0 [concrete = constants.%ptr.as.Copy.impl.Op.6e2]
// CHECK:STDOUT:   %bound_method.loc19_24.1: <bound method> = bound_method %addr.loc19, %impl.elem0.loc19
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.fe2]
// CHECK:STDOUT:   %bound_method.loc19_24.2: <bound method> = bound_method %addr.loc19, %specific_fn.loc19
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc19: init %ptr.143 = call %bound_method.loc19_24.2(%addr.loc19)
// CHECK:STDOUT:   assign %q.var, %ptr.as.Copy.impl.Op.call.loc19
// CHECK:STDOUT:   %.loc19_20: type = splice_block %ptr.loc19 [concrete = constants.%ptr.143] {
// CHECK:STDOUT:     %i32.loc19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc19: type = ptr_type %i32.loc19 [concrete = constants.%ptr.143]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %q: ref %ptr.143 = ref_binding q, %q.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r.patt: %pattern_type.663 = ref_binding_pattern r [concrete]
// CHECK:STDOUT:     %r.var_patt: %pattern_type.663 = var_pattern %r.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %r.var: ref %ptr.143 = var %r.var_patt
// CHECK:STDOUT:   %s.ref.loc20: ref %struct_type.a.b.602 = name_ref s, %s
// CHECK:STDOUT:   %.loc20_26: ref %i32 = struct_access %s.ref.loc20, element1
// CHECK:STDOUT:   %addr.loc20: %ptr.143 = addr_of %.loc20_26
// CHECK:STDOUT:   %impl.elem0.loc20: %.948 = impl_witness_access constants.%Copy.impl_witness.45b, element0 [concrete = constants.%ptr.as.Copy.impl.Op.6e2]
// CHECK:STDOUT:   %bound_method.loc20_24.1: <bound method> = bound_method %addr.loc20, %impl.elem0.loc20
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.fe2]
// CHECK:STDOUT:   %bound_method.loc20_24.2: <bound method> = bound_method %addr.loc20, %specific_fn.loc20
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc20: init %ptr.143 = call %bound_method.loc20_24.2(%addr.loc20)
// CHECK:STDOUT:   assign %r.var, %ptr.as.Copy.impl.Op.call.loc20
// CHECK:STDOUT:   %.loc20_20: type = splice_block %ptr.loc20 [concrete = constants.%ptr.143] {
// CHECK:STDOUT:     %i32.loc20: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc20: type = ptr_type %i32.loc20 [concrete = constants.%ptr.143]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %r: ref %ptr.143 = ref_binding r, %r.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type.624 = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.var_patt: %pattern_type.624 = var_pattern %t.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %tuple.type.87c = var %t.var_patt
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc22: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc22_28.1: %tuple.type.f94 = tuple_literal (%int_1.loc22, %int_2.loc22) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %impl.elem0.loc22_28.1: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc22_28.1: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_28.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d]
// CHECK:STDOUT:   %specific_fn.loc22_28.1: <specific function> = specific_function %impl.elem0.loc22_28.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_28.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22_28.1 [concrete = constants.%bound_method.33d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_28.1: init %i32 = call %bound_method.loc22_28.2(%int_1.loc22) [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc22_28.2: init %i32 = converted %int_1.loc22, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_28.1 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %tuple.elem0.loc22: ref %i32 = tuple_access %t.var, element0
// CHECK:STDOUT:   %.loc22_28.3: init %i32 to %tuple.elem0.loc22 = in_place_init %.loc22_28.2 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %impl.elem0.loc22_28.2: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc22_28.3: <bound method> = bound_method %int_2.loc22, %impl.elem0.loc22_28.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.22e]
// CHECK:STDOUT:   %specific_fn.loc22_28.2: <specific function> = specific_function %impl.elem0.loc22_28.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_28.4: <bound method> = bound_method %int_2.loc22, %specific_fn.loc22_28.2 [concrete = constants.%bound_method.cc3]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_28.2: init %i32 = call %bound_method.loc22_28.4(%int_2.loc22) [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc22_28.4: init %i32 = converted %int_2.loc22, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_28.2 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %tuple.elem1.loc22: ref %i32 = tuple_access %t.var, element1
// CHECK:STDOUT:   %.loc22_28.5: init %i32 to %tuple.elem1.loc22 = in_place_init %.loc22_28.4 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc22_28.6: init %tuple.type.87c to %t.var = tuple_init (%.loc22_28.3, %.loc22_28.5) [concrete = constants.%tuple.03f]
// CHECK:STDOUT:   %.loc22_3: init %tuple.type.87c = converted %.loc22_28.1, %.loc22_28.6 [concrete = constants.%tuple.03f]
// CHECK:STDOUT:   assign %t.var, %.loc22_3
// CHECK:STDOUT:   %.loc22_19.1: type = splice_block %.loc22_19.3 [concrete = constants.%tuple.type.87c] {
// CHECK:STDOUT:     %i32.loc22_11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc22_16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc22_19.2: %tuple.type.24b = tuple_literal (%i32.loc22_11, %i32.loc22_16) [concrete = constants.%tuple.8e3]
// CHECK:STDOUT:     %.loc22_19.3: type = converted %.loc22_19.2, constants.%tuple.type.87c [concrete = constants.%tuple.type.87c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %tuple.type.87c = ref_binding t, %t.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t0.patt: %pattern_type.663 = ref_binding_pattern t0 [concrete]
// CHECK:STDOUT:     %t0.var_patt: %pattern_type.663 = var_pattern %t0.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t0.var: ref %ptr.143 = var %t0.var_patt
// CHECK:STDOUT:   %t.ref.loc23: ref %tuple.type.87c = name_ref t, %t
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0.loc23: ref %i32 = tuple_access %t.ref.loc23, element0
// CHECK:STDOUT:   %addr.loc23: %ptr.143 = addr_of %tuple.elem0.loc23
// CHECK:STDOUT:   %impl.elem0.loc23: %.948 = impl_witness_access constants.%Copy.impl_witness.45b, element0 [concrete = constants.%ptr.as.Copy.impl.Op.6e2]
// CHECK:STDOUT:   %bound_method.loc23_25.1: <bound method> = bound_method %addr.loc23, %impl.elem0.loc23
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.fe2]
// CHECK:STDOUT:   %bound_method.loc23_25.2: <bound method> = bound_method %addr.loc23, %specific_fn.loc23
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc23: init %ptr.143 = call %bound_method.loc23_25.2(%addr.loc23)
// CHECK:STDOUT:   assign %t0.var, %ptr.as.Copy.impl.Op.call.loc23
// CHECK:STDOUT:   %.loc23: type = splice_block %ptr.loc23 [concrete = constants.%ptr.143] {
// CHECK:STDOUT:     %i32.loc23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc23: type = ptr_type %i32.loc23 [concrete = constants.%ptr.143]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t0: ref %ptr.143 = ref_binding t0, %t0.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t1.patt: %pattern_type.663 = ref_binding_pattern t1 [concrete]
// CHECK:STDOUT:     %t1.var_patt: %pattern_type.663 = var_pattern %t1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t1.var: ref %ptr.143 = var %t1.var_patt
// CHECK:STDOUT:   %t.ref.loc24: ref %tuple.type.87c = name_ref t, %t
// CHECK:STDOUT:   %int_1.loc24: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc24: ref %i32 = tuple_access %t.ref.loc24, element1
// CHECK:STDOUT:   %addr.loc24: %ptr.143 = addr_of %tuple.elem1.loc24
// CHECK:STDOUT:   %impl.elem0.loc24: %.948 = impl_witness_access constants.%Copy.impl_witness.45b, element0 [concrete = constants.%ptr.as.Copy.impl.Op.6e2]
// CHECK:STDOUT:   %bound_method.loc24_25.1: <bound method> = bound_method %addr.loc24, %impl.elem0.loc24
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.fe2]
// CHECK:STDOUT:   %bound_method.loc24_25.2: <bound method> = bound_method %addr.loc24, %specific_fn.loc24
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc24: init %ptr.143 = call %bound_method.loc24_25.2(%addr.loc24)
// CHECK:STDOUT:   assign %t1.var, %ptr.as.Copy.impl.Op.call.loc24
// CHECK:STDOUT:   %.loc24: type = splice_block %ptr.loc24 [concrete = constants.%ptr.143] {
// CHECK:STDOUT:     %i32.loc24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc24: type = ptr_type %i32.loc24 [concrete = constants.%ptr.143]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t1: ref %ptr.143 = ref_binding t1, %t1.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc24: <bound method> = bound_method %t1.var, constants.%Destroy.Op.1dc86d.1
// CHECK:STDOUT:   %Destroy.Op.call.loc24: init %empty_tuple.type = call %Destroy.Op.bound.loc24(%t1.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc23: <bound method> = bound_method %t0.var, constants.%Destroy.Op.1dc86d.1
// CHECK:STDOUT:   %Destroy.Op.call.loc23: init %empty_tuple.type = call %Destroy.Op.bound.loc23(%t0.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc22: <bound method> = bound_method %t.var, constants.%Destroy.Op.1dc86d.4
// CHECK:STDOUT:   %Destroy.Op.call.loc22: init %empty_tuple.type = call %Destroy.Op.bound.loc22(%t.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc20: <bound method> = bound_method %r.var, constants.%Destroy.Op.1dc86d.1
// CHECK:STDOUT:   %Destroy.Op.call.loc20: init %empty_tuple.type = call %Destroy.Op.bound.loc20(%r.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc19: <bound method> = bound_method %q.var, constants.%Destroy.Op.1dc86d.1
// CHECK:STDOUT:   %Destroy.Op.call.loc19: init %empty_tuple.type = call %Destroy.Op.bound.loc19(%q.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc18: <bound method> = bound_method %p.var, constants.%Destroy.Op.1dc86d.5
// CHECK:STDOUT:   %Destroy.Op.call.loc18: init %empty_tuple.type = call %Destroy.Op.bound.loc18(%p.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc16: <bound method> = bound_method %s.var, constants.%Destroy.Op.1dc86d.6
// CHECK:STDOUT:   %Destroy.Op.call.loc16: init %empty_tuple.type = call %Destroy.Op.bound.loc16(%s.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24(%self.param: ref %ptr.143) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc22_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc22_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc22_3.3(%self.param: ref %tuple.type.87c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18(%self.param: ref %ptr.148) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc16(%self.param: ref %struct_type.a.b.602) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
