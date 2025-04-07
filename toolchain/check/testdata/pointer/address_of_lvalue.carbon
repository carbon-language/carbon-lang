// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.501: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %ptr.3ee: type = ptr_type %struct_type.a.b.501 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.501 = struct_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %s.patt: %struct_type.a.b.501 = binding_pattern s
// CHECK:STDOUT:     %.loc12_3.1: %struct_type.a.b.501 = var_pattern %s.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %struct_type.a.b.501 = var s
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc12: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc12_46.1: %struct_type.a.b.cfd = struct_literal (%int_1.loc12, %int_2.loc12)
// CHECK:STDOUT:   %impl.elem0.loc12_46.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc12_46.1: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12_46.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc12_46.1: <specific function> = specific_function %impl.elem0.loc12_46.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_46.2: <bound method> = bound_method %int_1.loc12, %specific_fn.loc12_46.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc12_46.1: init %i32 = call %bound_method.loc12_46.2(%int_1.loc12) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_46.2: init %i32 = converted %int_1.loc12, %int.convert_checked.loc12_46.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_46.3: ref %i32 = struct_access %s.var, element0
// CHECK:STDOUT:   %.loc12_46.4: init %i32 = initialize_from %.loc12_46.2 to %.loc12_46.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc12_46.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc12_46.3: <bound method> = bound_method %int_2.loc12, %impl.elem0.loc12_46.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc12_46.2: <specific function> = specific_function %impl.elem0.loc12_46.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_46.4: <bound method> = bound_method %int_2.loc12, %specific_fn.loc12_46.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc12_46.2: init %i32 = call %bound_method.loc12_46.4(%int_2.loc12) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc12_46.5: init %i32 = converted %int_2.loc12, %int.convert_checked.loc12_46.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc12_46.6: ref %i32 = struct_access %s.var, element1
// CHECK:STDOUT:   %.loc12_46.7: init %i32 = initialize_from %.loc12_46.5 to %.loc12_46.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc12_46.8: init %struct_type.a.b.501 = struct_init (%.loc12_46.4, %.loc12_46.7) to %s.var [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc12_3.2: init %struct_type.a.b.501 = converted %.loc12_46.1, %.loc12_46.8 [concrete = constants.%struct]
// CHECK:STDOUT:   assign %s.var, %.loc12_3.2
// CHECK:STDOUT:   %.loc12_27: type = splice_block %struct_type.a.b.loc12 [concrete = constants.%struct_type.a.b.501] {
// CHECK:STDOUT:     %int_32.loc12_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc12_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.loc12: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.501]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %struct_type.a.b.501 = bind_name s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %ptr.3ee = binding_pattern p
// CHECK:STDOUT:     %.loc14_3: %ptr.3ee = var_pattern %p.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.3ee = var p
// CHECK:STDOUT:   %s.ref.loc14: ref %struct_type.a.b.501 = name_ref s, %s
// CHECK:STDOUT:   %addr.loc14: %ptr.3ee = addr_of %s.ref.loc14
// CHECK:STDOUT:   assign %p.var, %addr.loc14
// CHECK:STDOUT:   %.loc14_28: type = splice_block %ptr.loc14 [concrete = constants.%ptr.3ee] {
// CHECK:STDOUT:     %int_32.loc14_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc14_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.loc14: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.501]
// CHECK:STDOUT:     %ptr.loc14: type = ptr_type %struct_type.a.b.loc14 [concrete = constants.%ptr.3ee]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.3ee = bind_name p, %p.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %q.patt: %ptr.235 = binding_pattern q
// CHECK:STDOUT:     %.loc15_3: %ptr.235 = var_pattern %q.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %q.var: ref %ptr.235 = var q
// CHECK:STDOUT:   %s.ref.loc15: ref %struct_type.a.b.501 = name_ref s, %s
// CHECK:STDOUT:   %.loc15_19: ref %i32 = struct_access %s.ref.loc15, element0
// CHECK:STDOUT:   %addr.loc15: %ptr.235 = addr_of %.loc15_19
// CHECK:STDOUT:   assign %q.var, %addr.loc15
// CHECK:STDOUT:   %.loc15_13: type = splice_block %ptr.loc15 [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc15: type = ptr_type %i32.loc15 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %q: ref %ptr.235 = bind_name q, %q.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r.patt: %ptr.235 = binding_pattern r
// CHECK:STDOUT:     %.loc16_3: %ptr.235 = var_pattern %r.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %r.var: ref %ptr.235 = var r
// CHECK:STDOUT:   %s.ref.loc16: ref %struct_type.a.b.501 = name_ref s, %s
// CHECK:STDOUT:   %.loc16_19: ref %i32 = struct_access %s.ref.loc16, element1
// CHECK:STDOUT:   %addr.loc16: %ptr.235 = addr_of %.loc16_19
// CHECK:STDOUT:   assign %r.var, %addr.loc16
// CHECK:STDOUT:   %.loc16_13: type = splice_block %ptr.loc16 [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc16: type = ptr_type %i32.loc16 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %r: ref %ptr.235 = bind_name r, %r.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %tuple.type.d07 = binding_pattern t
// CHECK:STDOUT:     %.loc18_3.1: %tuple.type.d07 = var_pattern %t.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %tuple.type.d07 = var t
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc18: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc18_28.1: %tuple.type.f94 = tuple_literal (%int_1.loc18, %int_2.loc18)
// CHECK:STDOUT:   %impl.elem0.loc18_28.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc18_28.1: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_28.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc18_28.1: <specific function> = specific_function %impl.elem0.loc18_28.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_28.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18_28.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc18_28.1: init %i32 = call %bound_method.loc18_28.2(%int_1.loc18) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_28.2: init %i32 = converted %int_1.loc18, %int.convert_checked.loc18_28.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem0.loc18: ref %i32 = tuple_access %t.var, element0
// CHECK:STDOUT:   %.loc18_28.3: init %i32 = initialize_from %.loc18_28.2 to %tuple.elem0.loc18 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc18_28.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc18_28.3: <bound method> = bound_method %int_2.loc18, %impl.elem0.loc18_28.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc18_28.2: <specific function> = specific_function %impl.elem0.loc18_28.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_28.4: <bound method> = bound_method %int_2.loc18, %specific_fn.loc18_28.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc18_28.2: init %i32 = call %bound_method.loc18_28.4(%int_2.loc18) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc18_28.4: init %i32 = converted %int_2.loc18, %int.convert_checked.loc18_28.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.elem1.loc18: ref %i32 = tuple_access %t.var, element1
// CHECK:STDOUT:   %.loc18_28.5: init %i32 = initialize_from %.loc18_28.4 to %tuple.elem1.loc18 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc18_28.6: init %tuple.type.d07 = tuple_init (%.loc18_28.3, %.loc18_28.5) to %t.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc18_3.2: init %tuple.type.d07 = converted %.loc18_28.1, %.loc18_28.6 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %t.var, %.loc18_3.2
// CHECK:STDOUT:   %.loc18_19.1: type = splice_block %.loc18_19.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:     %int_32.loc18_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc18_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18_19.2: %tuple.type.24b = tuple_literal (%i32.loc18_11, %i32.loc18_16)
// CHECK:STDOUT:     %.loc18_19.3: type = converted %.loc18_19.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %tuple.type.d07 = bind_name t, %t.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t0.patt: %ptr.235 = binding_pattern t0
// CHECK:STDOUT:     %.loc19_3: %ptr.235 = var_pattern %t0.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t0.var: ref %ptr.235 = var t0
// CHECK:STDOUT:   %t.ref.loc19: ref %tuple.type.d07 = name_ref t, %t
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0.loc19: ref %i32 = tuple_access %t.ref.loc19, element0
// CHECK:STDOUT:   %addr.loc19: %ptr.235 = addr_of %tuple.elem0.loc19
// CHECK:STDOUT:   assign %t0.var, %addr.loc19
// CHECK:STDOUT:   %.loc19_14: type = splice_block %ptr.loc19 [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc19: type = ptr_type %i32.loc19 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t0: ref %ptr.235 = bind_name t0, %t0.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t1.patt: %ptr.235 = binding_pattern t1
// CHECK:STDOUT:     %.loc20_3: %ptr.235 = var_pattern %t1.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t1.var: ref %ptr.235 = var t1
// CHECK:STDOUT:   %t.ref.loc20: ref %tuple.type.d07 = name_ref t, %t
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc20: ref %i32 = tuple_access %t.ref.loc20, element1
// CHECK:STDOUT:   %addr.loc20: %ptr.235 = addr_of %tuple.elem1.loc20
// CHECK:STDOUT:   assign %t1.var, %addr.loc20
// CHECK:STDOUT:   %.loc20_14: type = splice_block %ptr.loc20 [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc20: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc20: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc20: type = ptr_type %i32.loc20 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t1: ref %ptr.235 = bind_name t1, %t1.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
