// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/in_place_tuple_init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/in_place_tuple_init.carbon

// --- in_place_tuple_init.carbon
library "[[@TEST_NAME]]";

fn F() -> (i32, i32);

fn G() -> (i32, i32) {
  //@dump-sem-ir-begin
  var v: (i32, i32) = F();
  v = F();
  return F();
  //@dump-sem-ir-end
}

fn H() -> i32 {
  //@dump-sem-ir-begin
  return G().0;
  //@dump-sem-ir-end
}

// --- nested_tuple_in_place.carbon
library "[[@TEST_NAME]]";

fn F() -> (i32, i32, i32);

fn G() {
  //@dump-sem-ir-begin
  var unused v: ((i32, i32, i32), (i32, i32, i32)) = (F(), F());
  //@dump-sem-ir-end
}

fn H() {
  //@dump-sem-ir-begin
  var unused v: (i32, (i32, i32, i32), i32) = (1, F(), 2);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- in_place_tuple_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.b59: %tuple.type.24b = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.e55: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.394: type = pattern_type %tuple.type.e55 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %v.patt: %pattern_type.394 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt: %pattern_type.394 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc7_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %tuple.type.e55 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.394 = ref_binding_pattern v [concrete = constants.%v.patt]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.394 = var_pattern %v.patt [concrete = constants.%v.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %tuple.type.e55 = var %v.var_patt
// CHECK:STDOUT:   %F.ref.loc7: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc7_3: ref %tuple.type.e55 = splice_block %v.var {}
// CHECK:STDOUT:   %F.call.loc7: init %tuple.type.e55 to %.loc7_3 = call %F.ref.loc7()
// CHECK:STDOUT:   assign %v.var, %F.call.loc7
// CHECK:STDOUT:   %.loc7_19.1: type = splice_block %.loc7_19.3 [concrete = constants.%tuple.type.e55] {
// CHECK:STDOUT:     %i32.loc7_11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc7_16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7_19.2: %tuple.type.24b = tuple_literal (%i32.loc7_11, %i32.loc7_16) [concrete = constants.%tuple.b59]
// CHECK:STDOUT:     %.loc7_19.3: type = converted %.loc7_19.2, constants.%tuple.type.e55 [concrete = constants.%tuple.type.e55]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %tuple.type.e55 = wrapper_binding v, %v.var
// CHECK:STDOUT:   %v.ref: ref %tuple.type.e55 = name_ref v, %v
// CHECK:STDOUT:   %F.ref.loc8: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc8: ref %tuple.type.e55 = splice_block %v.ref {}
// CHECK:STDOUT:   %F.call.loc8: init %tuple.type.e55 to %.loc8 = call %F.ref.loc8()
// CHECK:STDOUT:   assign %v.ref, %F.call.loc8
// CHECK:STDOUT:   %F.ref.loc9: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %F.call.loc9: init %tuple.type.e55 to %.loc5_20.1 = call %F.ref.loc9()
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %v.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%v.var)
// CHECK:STDOUT:   return %F.call.loc9 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.3(%self.param: ref %tuple.type.e55) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc15_12.1: ref %tuple.type.e55 = temporary_storage
// CHECK:STDOUT:   %G.call: init %tuple.type.e55 to %.loc15_12.1 = call %G.ref()
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc15_12.2: ref %tuple.type.e55 = temporary %.loc15_12.1, %G.call
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %.loc15_12.2, element0
// CHECK:STDOUT:   %.loc15_13: %i32 = acquire_value %tuple.elem0
// CHECK:STDOUT:   %impl.elem0: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc15_13.1: <bound method> = bound_method %.loc15_13, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_13.2: <bound method> = bound_method %.loc15_13, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc15_13.2(%.loc15_13)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc15_12.2, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc15_12.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_tuple_in_place.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.04d: %tuple.type.ff9 = tuple_value (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.555: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.f9f: type = tuple_type (%tuple.type.ff9, %tuple.type.ff9) [concrete]
// CHECK:STDOUT:   %tuple.6a5: %tuple.type.f9f = tuple_value (%tuple.04d, %tuple.04d) [concrete]
// CHECK:STDOUT:   %tuple.type.015: type = tuple_type (%tuple.type.555, %tuple.type.555) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.e12: type = pattern_type %tuple.type.015 [concrete]
// CHECK:STDOUT:   %v.patt.dde: %pattern_type.e12 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt.678: %pattern_type.e12 = var_pattern %v.patt.dde [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc7_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.3a3: type = tuple_type (type, %tuple.type.ff9, type) [concrete]
// CHECK:STDOUT:   %tuple.82e: %tuple.type.3a3 = tuple_value (%i32, %tuple.04d, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.b7e: type = tuple_type (%i32, %tuple.type.555, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.f8e: type = pattern_type %tuple.type.b7e [concrete]
// CHECK:STDOUT:   %v.patt.7e2: %pattern_type.f8e = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt.409: %pattern_type.f8e = var_pattern %v.patt.7e2 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.c47: type = tuple_type (Core.IntLiteral, %tuple.type.555, Core.IntLiteral) [concrete]
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
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.5: type = fn_type @Destroy.Op.loc13 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.5: %Destroy.Op.type.1d8f74.5 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.e12 = ref_binding_pattern v [concrete = constants.%v.patt.dde]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.e12 = var_pattern %v.patt [concrete = constants.%v.var_patt.678]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %tuple.type.015 = var %v.var_patt
// CHECK:STDOUT:   %F.ref.loc7_55: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %tuple.elem0: ref %tuple.type.555 = tuple_access %v.var, element0
// CHECK:STDOUT:   %F.call.loc7_57: init %tuple.type.555 to %tuple.elem0 = call %F.ref.loc7_55()
// CHECK:STDOUT:   %F.ref.loc7_60: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %tuple.elem1: ref %tuple.type.555 = tuple_access %v.var, element1
// CHECK:STDOUT:   %F.call.loc7_62: init %tuple.type.555 to %tuple.elem1 = call %F.ref.loc7_60()
// CHECK:STDOUT:   %.loc7_63.1: %tuple.type.015 = tuple_literal (%F.call.loc7_57, %F.call.loc7_62)
// CHECK:STDOUT:   %.loc7_63.2: init %tuple.type.015 to %v.var = tuple_init (%F.call.loc7_57, %F.call.loc7_62)
// CHECK:STDOUT:   %.loc7_3: init %tuple.type.015 = converted %.loc7_63.1, %.loc7_63.2
// CHECK:STDOUT:   assign %v.var, %.loc7_3
// CHECK:STDOUT:   %.loc7_50.1: type = splice_block %.loc7_50.5 [concrete = constants.%tuple.type.015] {
// CHECK:STDOUT:     %i32.loc7_19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc7_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc7_29: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7_32: %tuple.type.ff9 = tuple_literal (%i32.loc7_19, %i32.loc7_24, %i32.loc7_29) [concrete = constants.%tuple.04d]
// CHECK:STDOUT:     %i32.loc7_36: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc7_41: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc7_46: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7_49: %tuple.type.ff9 = tuple_literal (%i32.loc7_36, %i32.loc7_41, %i32.loc7_46) [concrete = constants.%tuple.04d]
// CHECK:STDOUT:     %.loc7_50.2: %tuple.type.f9f = tuple_literal (%.loc7_32, %.loc7_49) [concrete = constants.%tuple.6a5]
// CHECK:STDOUT:     %.loc7_50.3: type = converted constants.%tuple.04d, constants.%tuple.type.555 [concrete = constants.%tuple.type.555]
// CHECK:STDOUT:     %.loc7_50.4: type = converted constants.%tuple.04d, constants.%tuple.type.555 [concrete = constants.%tuple.type.555]
// CHECK:STDOUT:     %.loc7_50.5: type = converted %.loc7_50.2, constants.%tuple.type.015 [concrete = constants.%tuple.type.015]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %tuple.type.015 = wrapper_binding v, %v.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %v.var, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%v.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.3(%self.param: ref %tuple.type.555) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.4(%self.param: ref %tuple.type.015) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.f8e = ref_binding_pattern v [concrete = constants.%v.patt.7e2]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.f8e = var_pattern %v.patt [concrete = constants.%v.var_patt.409]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %tuple.type.b7e = var %v.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %tuple.elem1: ref %tuple.type.555 = tuple_access %v.var, element1
// CHECK:STDOUT:   %F.call: init %tuple.type.555 to %tuple.elem1 = call %F.ref()
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc13_57.1: %tuple.type.c47 = tuple_literal (%int_1, %F.call, %int_2)
// CHECK:STDOUT:   %impl.elem0.loc13_57.1: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc13_57.1: <bound method> = bound_method %int_1, %impl.elem0.loc13_57.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc13_57.1: <specific function> = specific_function %impl.elem0.loc13_57.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_57.2: <bound method> = bound_method %int_1, %specific_fn.loc13_57.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_57.1: init %i32 = call %bound_method.loc13_57.2(%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc13_57.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_57.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %v.var, element0
// CHECK:STDOUT:   %.loc13_57.3: init %i32 to %tuple.elem0 = in_place_init %.loc13_57.2 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc13_57.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc13_57.3: <bound method> = bound_method %int_2, %impl.elem0.loc13_57.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc13_57.2: <specific function> = specific_function %impl.elem0.loc13_57.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_57.4: <bound method> = bound_method %int_2, %specific_fn.loc13_57.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_57.2: init %i32 = call %bound_method.loc13_57.4(%int_2) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc13_57.4: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_57.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %tuple.elem2: ref %i32 = tuple_access %v.var, element2
// CHECK:STDOUT:   %.loc13_57.5: init %i32 to %tuple.elem2 = in_place_init %.loc13_57.4 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc13_57.6: init %tuple.type.b7e to %v.var = tuple_init (%.loc13_57.3, %F.call, %.loc13_57.5)
// CHECK:STDOUT:   %.loc13_3: init %tuple.type.b7e = converted %.loc13_57.1, %.loc13_57.6
// CHECK:STDOUT:   assign %v.var, %.loc13_3
// CHECK:STDOUT:   %.loc13_43.1: type = splice_block %.loc13_43.4 [concrete = constants.%tuple.type.b7e] {
// CHECK:STDOUT:     %i32.loc13_18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc13_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc13_29: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc13_34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc13_37: %tuple.type.ff9 = tuple_literal (%i32.loc13_24, %i32.loc13_29, %i32.loc13_34) [concrete = constants.%tuple.04d]
// CHECK:STDOUT:     %i32.loc13_40: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc13_43.2: %tuple.type.3a3 = tuple_literal (%i32.loc13_18, %.loc13_37, %i32.loc13_40) [concrete = constants.%tuple.82e]
// CHECK:STDOUT:     %.loc13_43.3: type = converted constants.%tuple.04d, constants.%tuple.type.555 [concrete = constants.%tuple.type.555]
// CHECK:STDOUT:     %.loc13_43.4: type = converted %.loc13_43.2, constants.%tuple.type.b7e [concrete = constants.%tuple.type.b7e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %tuple.type.b7e = wrapper_binding v, %v.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %v.var, constants.%Destroy.Op.1a2547.5
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%v.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13(%self.param: ref %tuple.type.b7e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
