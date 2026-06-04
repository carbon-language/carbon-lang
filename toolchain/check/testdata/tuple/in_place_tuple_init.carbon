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
// CHECK:STDOUT:   %tuple.79d: %tuple.type.24b = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.e5a: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.7d6: type = pattern_type %tuple.type.e5a [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.3: type = fn_type @Destroy.Op.loc7_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.3: %Destroy.Op.type.790b3b.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.2f2: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.a64: %Int.as.Copy.impl.Op.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1a4: <witness> = impl_witness imports.%Copy.impl_witness_table.0ac, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.422: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.ce7: %Int.as.Copy.impl.Op.type.422 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.1a4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7e2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.604: type = fn_type_with_self_type %Copy.WithSelf.Op.type.7e2, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.ce7, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.55f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.2f2) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.a64)]
// CHECK:STDOUT:   %Copy.impl_witness_table.0ac = impl_witness_table (%Core.import_ref.55f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %tuple.type.e5a {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.7d6 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.7d6 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %tuple.type.e5a = var %v.var_patt
// CHECK:STDOUT:   %F.ref.loc7: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc7_3: ref %tuple.type.e5a = splice_block %v.var {}
// CHECK:STDOUT:   %F.call.loc7: init %tuple.type.e5a to %.loc7_3 = call %F.ref.loc7()
// CHECK:STDOUT:   assign %v.var, %F.call.loc7
// CHECK:STDOUT:   %.loc7_19.1: type = splice_block %.loc7_19.3 [concrete = constants.%tuple.type.e5a] {
// CHECK:STDOUT:     %i32.loc7_11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc7_16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7_19.2: %tuple.type.24b = tuple_literal (%i32.loc7_11, %i32.loc7_16) [concrete = constants.%tuple.79d]
// CHECK:STDOUT:     %.loc7_19.3: type = converted %.loc7_19.2, constants.%tuple.type.e5a [concrete = constants.%tuple.type.e5a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %tuple.type.e5a = ref_binding v, %v.var
// CHECK:STDOUT:   %v.ref: ref %tuple.type.e5a = name_ref v, %v
// CHECK:STDOUT:   %F.ref.loc8: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc8: ref %tuple.type.e5a = splice_block %v.ref {}
// CHECK:STDOUT:   %F.call.loc8: init %tuple.type.e5a to %.loc8 = call %F.ref.loc8()
// CHECK:STDOUT:   assign %v.ref, %F.call.loc8
// CHECK:STDOUT:   %F.ref.loc9: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %F.call.loc9: init %tuple.type.e5a to %.loc5_20.1 = call %F.ref.loc9()
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %v.var, constants.%Destroy.Op.2dd605.3
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
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.3(%self.param: ref %tuple.type.e5a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc15_12.1: ref %tuple.type.e5a = temporary_storage
// CHECK:STDOUT:   %G.call: init %tuple.type.e5a to %.loc15_12.1 = call %G.ref()
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc15_12.2: ref %tuple.type.e5a = temporary %.loc15_12.1, %G.call
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %.loc15_12.2, element0
// CHECK:STDOUT:   %.loc15_13: %i32 = acquire_value %tuple.elem0
// CHECK:STDOUT:   %impl.elem0: %.604 = impl_witness_access constants.%Copy.impl_witness.1a4, element0 [concrete = constants.%Int.as.Copy.impl.Op.ce7]
// CHECK:STDOUT:   %bound_method.loc15_13.1: <bound method> = bound_method %.loc15_13, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_13.2: <bound method> = bound_method %.loc15_13, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc15_13.2(%.loc15_13)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc15_12.2, constants.%Destroy.Op.2dd605.3
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
// CHECK:STDOUT:   %tuple.27c: %tuple.type.ff9 = tuple_value (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.05f: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.f9f: type = tuple_type (%tuple.type.ff9, %tuple.type.ff9) [concrete]
// CHECK:STDOUT:   %tuple.d3b: %tuple.type.f9f = tuple_value (%tuple.27c, %tuple.27c) [concrete]
// CHECK:STDOUT:   %tuple.type.657: type = tuple_type (%tuple.type.05f, %tuple.type.05f) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.f75: type = pattern_type %tuple.type.657 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.4: type = fn_type @Destroy.Op.loc7_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.4: %Destroy.Op.type.790b3b.4 = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.3a3: type = tuple_type (type, %tuple.type.ff9, type) [concrete]
// CHECK:STDOUT:   %tuple.1c0: %tuple.type.3a3 = tuple_value (%i32, %tuple.27c, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.8e6: type = tuple_type (%i32, %tuple.type.05f, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.f5e: type = pattern_type %tuple.type.8e6 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.274: type = tuple_type (Core.IntLiteral, %tuple.type.05f, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.360: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3d6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0bd, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.360 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3d6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.474: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da5, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c1a: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.e5b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d9c: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.bf7: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %bound_method.7d5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.228: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.5: type = fn_type @Destroy.Op.loc13 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.5: %Destroy.Op.type.790b3b.5 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.b42: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0bd = impl_witness_table (%Core.import_ref.b42), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.f75 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.f75 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %tuple.type.657 = var %v.var_patt
// CHECK:STDOUT:   %F.ref.loc7_55: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %tuple.elem0: ref %tuple.type.05f = tuple_access %v.var, element0
// CHECK:STDOUT:   %F.call.loc7_57: init %tuple.type.05f to %tuple.elem0 = call %F.ref.loc7_55()
// CHECK:STDOUT:   %F.ref.loc7_60: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %tuple.elem1: ref %tuple.type.05f = tuple_access %v.var, element1
// CHECK:STDOUT:   %F.call.loc7_62: init %tuple.type.05f to %tuple.elem1 = call %F.ref.loc7_60()
// CHECK:STDOUT:   %.loc7_63.1: %tuple.type.657 = tuple_literal (%F.call.loc7_57, %F.call.loc7_62)
// CHECK:STDOUT:   %.loc7_63.2: init %tuple.type.657 to %v.var = tuple_init (%F.call.loc7_57, %F.call.loc7_62)
// CHECK:STDOUT:   %.loc7_3: init %tuple.type.657 = converted %.loc7_63.1, %.loc7_63.2
// CHECK:STDOUT:   assign %v.var, %.loc7_3
// CHECK:STDOUT:   %.loc7_50.1: type = splice_block %.loc7_50.5 [concrete = constants.%tuple.type.657] {
// CHECK:STDOUT:     %i32.loc7_19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc7_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc7_29: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7_32: %tuple.type.ff9 = tuple_literal (%i32.loc7_19, %i32.loc7_24, %i32.loc7_29) [concrete = constants.%tuple.27c]
// CHECK:STDOUT:     %i32.loc7_36: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc7_41: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc7_46: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7_49: %tuple.type.ff9 = tuple_literal (%i32.loc7_36, %i32.loc7_41, %i32.loc7_46) [concrete = constants.%tuple.27c]
// CHECK:STDOUT:     %.loc7_50.2: %tuple.type.f9f = tuple_literal (%.loc7_32, %.loc7_49) [concrete = constants.%tuple.d3b]
// CHECK:STDOUT:     %.loc7_50.3: type = converted constants.%tuple.27c, constants.%tuple.type.05f [concrete = constants.%tuple.type.05f]
// CHECK:STDOUT:     %.loc7_50.4: type = converted constants.%tuple.27c, constants.%tuple.type.05f [concrete = constants.%tuple.type.05f]
// CHECK:STDOUT:     %.loc7_50.5: type = converted %.loc7_50.2, constants.%tuple.type.657 [concrete = constants.%tuple.type.657]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %tuple.type.657 = ref_binding v, %v.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %v.var, constants.%Destroy.Op.2dd605.4
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
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.3(%self.param: ref %tuple.type.05f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.4(%self.param: ref %tuple.type.657) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.f5e = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.f5e = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %tuple.type.8e6 = var %v.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %tuple.elem1: ref %tuple.type.05f = tuple_access %v.var, element1
// CHECK:STDOUT:   %F.call: init %tuple.type.05f to %tuple.elem1 = call %F.ref()
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc13_57.1: %tuple.type.274 = tuple_literal (%int_1, %F.call, %int_2)
// CHECK:STDOUT:   %impl.elem0.loc13_57.1: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc13_57.1: <bound method> = bound_method %int_1, %impl.elem0.loc13_57.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c1a]
// CHECK:STDOUT:   %specific_fn.loc13_57.1: <specific function> = specific_function %impl.elem0.loc13_57.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_57.2: <bound method> = bound_method %int_1, %specific_fn.loc13_57.1 [concrete = constants.%bound_method.e5b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_57.1: init %i32 = call %bound_method.loc13_57.2(%int_1) [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %.loc13_57.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_57.1 [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %v.var, element0
// CHECK:STDOUT:   %.loc13_57.3: init %i32 to %tuple.elem0 = in_place_init %.loc13_57.2 [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %impl.elem0.loc13_57.2: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc13_57.3: <bound method> = bound_method %int_2, %impl.elem0.loc13_57.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.bf7]
// CHECK:STDOUT:   %specific_fn.loc13_57.2: <specific function> = specific_function %impl.elem0.loc13_57.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_57.4: <bound method> = bound_method %int_2, %specific_fn.loc13_57.2 [concrete = constants.%bound_method.7d5]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_57.2: init %i32 = call %bound_method.loc13_57.4(%int_2) [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %.loc13_57.4: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_57.2 [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %tuple.elem2: ref %i32 = tuple_access %v.var, element2
// CHECK:STDOUT:   %.loc13_57.5: init %i32 to %tuple.elem2 = in_place_init %.loc13_57.4 [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %.loc13_57.6: init %tuple.type.8e6 to %v.var = tuple_init (%.loc13_57.3, %F.call, %.loc13_57.5)
// CHECK:STDOUT:   %.loc13_3: init %tuple.type.8e6 = converted %.loc13_57.1, %.loc13_57.6
// CHECK:STDOUT:   assign %v.var, %.loc13_3
// CHECK:STDOUT:   %.loc13_43.1: type = splice_block %.loc13_43.4 [concrete = constants.%tuple.type.8e6] {
// CHECK:STDOUT:     %i32.loc13_18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc13_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc13_29: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc13_34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc13_37: %tuple.type.ff9 = tuple_literal (%i32.loc13_24, %i32.loc13_29, %i32.loc13_34) [concrete = constants.%tuple.27c]
// CHECK:STDOUT:     %i32.loc13_40: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc13_43.2: %tuple.type.3a3 = tuple_literal (%i32.loc13_18, %.loc13_37, %i32.loc13_40) [concrete = constants.%tuple.1c0]
// CHECK:STDOUT:     %.loc13_43.3: type = converted constants.%tuple.27c, constants.%tuple.type.05f [concrete = constants.%tuple.type.05f]
// CHECK:STDOUT:     %.loc13_43.4: type = converted %.loc13_43.2, constants.%tuple.type.8e6 [concrete = constants.%tuple.type.8e6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %tuple.type.8e6 = ref_binding v, %v.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %v.var, constants.%Destroy.Op.2dd605.5
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%v.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13(%self.param: ref %tuple.type.8e6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
