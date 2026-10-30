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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.be9: %tuple.type.693 = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.35f: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.354: type = pattern_type %tuple.type.35f [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %v.patt: %pattern_type.354 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt: %pattern_type.354 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_3.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.49a: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.077: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.49a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c2b: %Destroy.type = facet_value %i32, (%DestroyImpl.impl_witness.077, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ec1: %SubobjectDestroy.type = facet_value %tuple.type.35f, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.df1: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ec1) [concrete]
// CHECK:STDOUT:   %Destroy.facet.58e: %Destroy.type = facet_value %tuple.type.35f, (%DestroyImpl.impl_witness.df1, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %tuple.type.35f {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref %tuple.type.35f = var_storage %v.var_patt
// CHECK:STDOUT:   %F.ref.loc7: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc7_3.1: ref %tuple.type.35f = splice_block %v.var {}
// CHECK:STDOUT:   %F.call.loc7: init %tuple.type.35f to %.loc7_3.1 = call %F.ref.loc7()
// CHECK:STDOUT:   assign %v.var, %F.call.loc7
// CHECK:STDOUT:   %.loc7_19.1: type = splice_block %.loc7_19.3 [concrete = constants.%tuple.type.35f] {
// CHECK:STDOUT:     %i32.loc7_11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc7_16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7_19.2: %tuple.type.693 = tuple_literal (%i32.loc7_11, %i32.loc7_16) [concrete = constants.%tuple.be9]
// CHECK:STDOUT:     %.loc7_19.3: type = converted %.loc7_19.2, constants.%tuple.type.35f [concrete = constants.%tuple.type.35f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %tuple.type.35f = wrapper_binding v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.354 = ref_binding_pattern v [concrete = constants.%v.patt]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.354 = var_pattern %v.patt [concrete = constants.%v.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.ref: ref %tuple.type.35f = name_ref v, %v
// CHECK:STDOUT:   %F.ref.loc8: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc8: ref %tuple.type.35f = splice_block %v.ref {}
// CHECK:STDOUT:   %F.call.loc8: init %tuple.type.35f to %.loc8 = call %F.ref.loc8()
// CHECK:STDOUT:   assign %v.ref, %F.call.loc8
// CHECK:STDOUT:   %F.ref.loc9: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %F.call.loc9: init %tuple.type.35f to %.loc5_20.1 = call %F.ref.loc9()
// CHECK:STDOUT:   %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%tuple.type.35f, (constants.%DestroyImpl.impl_witness.df1, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.58e]
// CHECK:STDOUT:   %.loc7_3.2: %Destroy.type = converted constants.%tuple.type.35f, %Destroy.facet.loc7_3.1 [concrete = constants.%Destroy.facet.58e]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%tuple.type.35f, (constants.%DestroyImpl.impl_witness.df1, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.58e]
// CHECK:STDOUT:   %.loc7_3.3: %Destroy.type = converted constants.%tuple.type.35f, %Destroy.facet.loc7_3.2 [concrete = constants.%Destroy.facet.58e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%v.var)
// CHECK:STDOUT:   return %F.call.loc9 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_3.2(%self.param: ref %tuple.type.35f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc7_3.1: %Destroy.type = converted constants.%i32, %Destroy.facet.loc7_3.1 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc7_3.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc7_3.2 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc7_3.3: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc7_3.3: %Destroy.type = converted constants.%i32, %Destroy.facet.loc7_3.3 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.4: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc7_3.4: %Destroy.type = converted constants.%i32, %Destroy.facet.loc7_3.4 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc15_12.1: ref %tuple.type.35f = temporary_storage
// CHECK:STDOUT:   %G.call: init %tuple.type.35f to %.loc15_12.1 = call %G.ref()
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc15_12.2: ref %tuple.type.35f = temporary %.loc15_12.1, %G.call
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %.loc15_12.2, element0
// CHECK:STDOUT:   %.loc15_13: %i32 = acquire_value %tuple.elem0
// CHECK:STDOUT:   %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc15_13.1: <bound method> = bound_method %.loc15_13, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_13.2: <bound method> = bound_method %.loc15_13, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc15_13.2(%.loc15_13)
// CHECK:STDOUT:   %Destroy.facet.loc15_12.1: %Destroy.type = facet_value constants.%tuple.type.35f, (constants.%DestroyImpl.impl_witness.df1, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.58e]
// CHECK:STDOUT:   %.loc15_12.3: %Destroy.type = converted constants.%tuple.type.35f, %Destroy.facet.loc15_12.1 [concrete = constants.%Destroy.facet.58e]
// CHECK:STDOUT:   %Destroy.facet.loc15_12.2: %Destroy.type = facet_value constants.%tuple.type.35f, (constants.%DestroyImpl.impl_witness.df1, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.58e]
// CHECK:STDOUT:   %.loc15_12.4: %Destroy.type = converted constants.%tuple.type.35f, %Destroy.facet.loc15_12.2 [concrete = constants.%Destroy.facet.58e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc15_12.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_tuple_in_place.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.531: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.0a8: %tuple.type.531 = tuple_value (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.abc: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.d58: type = tuple_type (%tuple.type.531, %tuple.type.531) [concrete]
// CHECK:STDOUT:   %tuple.f78: %tuple.type.d58 = tuple_value (%tuple.0a8, %tuple.0a8) [concrete]
// CHECK:STDOUT:   %tuple.type.860: type = tuple_type (%tuple.type.abc, %tuple.type.abc) [concrete]
// CHECK:STDOUT:   %pattern_type.d09: type = pattern_type %tuple.type.860 [concrete]
// CHECK:STDOUT:   %v.patt.cb2: %pattern_type.d09 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt.9f0: %pattern_type.d09 = var_pattern %v.patt.cb2 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_3.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.49a: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.077: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.49a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c2b: %Destroy.type = facet_value %i32, (%DestroyImpl.impl_witness.077, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_3.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.2e8: %SubobjectDestroy.type = facet_value %tuple.type.abc, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.62e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.2e8) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c1b: %Destroy.type = facet_value %tuple.type.abc, (%DestroyImpl.impl_witness.62e, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.45e: %SubobjectDestroy.type = facet_value %tuple.type.860, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b57: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.45e) [concrete]
// CHECK:STDOUT:   %Destroy.facet.e01: %Destroy.type = facet_value %tuple.type.860, (%DestroyImpl.impl_witness.b57, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %tuple.type.f8d: type = tuple_type (type, %tuple.type.531, type) [concrete]
// CHECK:STDOUT:   %tuple.2b8: %tuple.type.f8d = tuple_value (%i32, %tuple.0a8, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.3b7: type = tuple_type (%i32, %tuple.type.abc, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.427: type = pattern_type %tuple.type.3b7 [concrete]
// CHECK:STDOUT:   %v.patt.f61: %pattern_type.427 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt.4b5: %pattern_type.427 = var_pattern %v.patt.f61 [concrete]
// CHECK:STDOUT:   %tuple.type.376: type = tuple_type (Core.IntLiteral, %tuple.type.abc, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.a55: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.2e0: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d44: %SubobjectDestroy.type = facet_value %tuple.type.3b7, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.cb5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d44) [concrete]
// CHECK:STDOUT:   %Destroy.facet.fbe: %Destroy.type = facet_value %tuple.type.3b7, (%DestroyImpl.impl_witness.cb5, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref %tuple.type.860 = var_storage %v.var_patt
// CHECK:STDOUT:   %F.ref.loc7_55: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %tuple.elem0: ref %tuple.type.abc = tuple_access %v.var, element0
// CHECK:STDOUT:   %F.call.loc7_57: init %tuple.type.abc to %tuple.elem0 = call %F.ref.loc7_55()
// CHECK:STDOUT:   %F.ref.loc7_60: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %tuple.elem1: ref %tuple.type.abc = tuple_access %v.var, element1
// CHECK:STDOUT:   %F.call.loc7_62: init %tuple.type.abc to %tuple.elem1 = call %F.ref.loc7_60()
// CHECK:STDOUT:   %.loc7_63.1: %tuple.type.860 = tuple_literal (%F.call.loc7_57, %F.call.loc7_62)
// CHECK:STDOUT:   %.loc7_63.2: init %tuple.type.860 to %v.var = tuple_init (%F.call.loc7_57, %F.call.loc7_62)
// CHECK:STDOUT:   %.loc7_3.1: init %tuple.type.860 = converted %.loc7_63.1, %.loc7_63.2
// CHECK:STDOUT:   assign %v.var, %.loc7_3.1
// CHECK:STDOUT:   %.loc7_50.1: type = splice_block %.loc7_50.5 [concrete = constants.%tuple.type.860] {
// CHECK:STDOUT:     %i32.loc7_19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc7_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc7_29: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7_32: %tuple.type.531 = tuple_literal (%i32.loc7_19, %i32.loc7_24, %i32.loc7_29) [concrete = constants.%tuple.0a8]
// CHECK:STDOUT:     %i32.loc7_36: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc7_41: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc7_46: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7_49: %tuple.type.531 = tuple_literal (%i32.loc7_36, %i32.loc7_41, %i32.loc7_46) [concrete = constants.%tuple.0a8]
// CHECK:STDOUT:     %.loc7_50.2: %tuple.type.d58 = tuple_literal (%.loc7_32, %.loc7_49) [concrete = constants.%tuple.f78]
// CHECK:STDOUT:     %.loc7_50.3: type = converted constants.%tuple.0a8, constants.%tuple.type.abc [concrete = constants.%tuple.type.abc]
// CHECK:STDOUT:     %.loc7_50.4: type = converted constants.%tuple.0a8, constants.%tuple.type.abc [concrete = constants.%tuple.type.abc]
// CHECK:STDOUT:     %.loc7_50.5: type = converted %.loc7_50.2, constants.%tuple.type.860 [concrete = constants.%tuple.type.860]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %tuple.type.860 = wrapper_binding v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.d09 = ref_binding_pattern v [concrete = constants.%v.patt.cb2]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.d09 = var_pattern %v.patt [concrete = constants.%v.var_patt.9f0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%tuple.type.860, (constants.%DestroyImpl.impl_witness.b57, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.e01]
// CHECK:STDOUT:   %.loc7_3.2: %Destroy.type = converted constants.%tuple.type.860, %Destroy.facet.loc7_3.1 [concrete = constants.%Destroy.facet.e01]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%tuple.type.860, (constants.%DestroyImpl.impl_witness.b57, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.e01]
// CHECK:STDOUT:   %.loc7_3.3: %Destroy.type = converted constants.%tuple.type.860, %Destroy.facet.loc7_3.2 [concrete = constants.%Destroy.facet.e01]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%v.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_3.2(%self.param: ref %tuple.type.abc) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %tuple.elem2: ref %i32 = tuple_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc7_3.1: %Destroy.type = converted constants.%i32, %Destroy.facet.loc7_3.1 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc7_3.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc7_3.2 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem2)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc7_3.3: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc7_3.3: %Destroy.type = converted constants.%i32, %Destroy.facet.loc7_3.3 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.4: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc7_3.4: %Destroy.type = converted constants.%i32, %Destroy.facet.loc7_3.4 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc7_3.5: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc7_3.5: %Destroy.type = converted constants.%i32, %Destroy.facet.loc7_3.5 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.6: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc7_3.6: %Destroy.type = converted constants.%i32, %Destroy.facet.loc7_3.6 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7_3.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_3.3(%self.param: ref %tuple.type.860) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1: ref %tuple.type.abc = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%tuple.type.abc, (constants.%DestroyImpl.impl_witness.62e, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.c1b]
// CHECK:STDOUT:   %.loc7_3.1: %Destroy.type = converted constants.%tuple.type.abc, %Destroy.facet.loc7_3.1 [concrete = constants.%Destroy.facet.c1b]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%tuple.type.abc, (constants.%DestroyImpl.impl_witness.62e, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.c1b]
// CHECK:STDOUT:   %.loc7_3.2: %Destroy.type = converted constants.%tuple.type.abc, %Destroy.facet.loc7_3.2 [concrete = constants.%Destroy.facet.c1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %tuple.type.abc = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc7_3.3: %Destroy.type = facet_value constants.%tuple.type.abc, (constants.%DestroyImpl.impl_witness.62e, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.c1b]
// CHECK:STDOUT:   %.loc7_3.3: %Destroy.type = converted constants.%tuple.type.abc, %Destroy.facet.loc7_3.3 [concrete = constants.%Destroy.facet.c1b]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.4: %Destroy.type = facet_value constants.%tuple.type.abc, (constants.%DestroyImpl.impl_witness.62e, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.c1b]
// CHECK:STDOUT:   %.loc7_3.4: %Destroy.type = converted constants.%tuple.type.abc, %Destroy.facet.loc7_3.4 [concrete = constants.%Destroy.facet.c1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref %tuple.type.3b7 = var_storage %v.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %tuple.elem1: ref %tuple.type.abc = tuple_access %v.var, element1
// CHECK:STDOUT:   %F.call: init %tuple.type.abc to %tuple.elem1 = call %F.ref()
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc13_57.1: %tuple.type.376 = tuple_literal (%int_1, %F.call, %int_2)
// CHECK:STDOUT:   %impl.elem0.loc13_57.1: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc13_57.1: <bound method> = bound_method %int_1, %impl.elem0.loc13_57.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc13_57.1: <specific function> = specific_function %impl.elem0.loc13_57.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_57.2: <bound method> = bound_method %int_1, %specific_fn.loc13_57.1 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_57.1: init %i32 = call %bound_method.loc13_57.2(%int_1) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc13_57.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_57.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %v.var, element0
// CHECK:STDOUT:   %.loc13_57.3: init %i32 to %tuple.elem0 = in_place_init %.loc13_57.2 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc13_57.2: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc13_57.3: <bound method> = bound_method %int_2, %impl.elem0.loc13_57.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc13_57.2: <specific function> = specific_function %impl.elem0.loc13_57.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_57.4: <bound method> = bound_method %int_2, %specific_fn.loc13_57.2 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_57.2: init %i32 = call %bound_method.loc13_57.4(%int_2) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc13_57.4: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13_57.2 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %tuple.elem2: ref %i32 = tuple_access %v.var, element2
// CHECK:STDOUT:   %.loc13_57.5: init %i32 to %tuple.elem2 = in_place_init %.loc13_57.4 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc13_57.6: init %tuple.type.3b7 to %v.var = tuple_init (%.loc13_57.3, %F.call, %.loc13_57.5)
// CHECK:STDOUT:   %.loc13_3.1: init %tuple.type.3b7 = converted %.loc13_57.1, %.loc13_57.6
// CHECK:STDOUT:   assign %v.var, %.loc13_3.1
// CHECK:STDOUT:   %.loc13_43.1: type = splice_block %.loc13_43.4 [concrete = constants.%tuple.type.3b7] {
// CHECK:STDOUT:     %i32.loc13_18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc13_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc13_29: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc13_34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc13_37: %tuple.type.531 = tuple_literal (%i32.loc13_24, %i32.loc13_29, %i32.loc13_34) [concrete = constants.%tuple.0a8]
// CHECK:STDOUT:     %i32.loc13_40: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc13_43.2: %tuple.type.f8d = tuple_literal (%i32.loc13_18, %.loc13_37, %i32.loc13_40) [concrete = constants.%tuple.2b8]
// CHECK:STDOUT:     %.loc13_43.3: type = converted constants.%tuple.0a8, constants.%tuple.type.abc [concrete = constants.%tuple.type.abc]
// CHECK:STDOUT:     %.loc13_43.4: type = converted %.loc13_43.2, constants.%tuple.type.3b7 [concrete = constants.%tuple.type.3b7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %tuple.type.3b7 = wrapper_binding v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.427 = ref_binding_pattern v [concrete = constants.%v.patt.f61]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.427 = var_pattern %v.patt [concrete = constants.%v.var_patt.4b5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%tuple.type.3b7, (constants.%DestroyImpl.impl_witness.cb5, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.fbe]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%tuple.type.3b7, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.fbe]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%tuple.type.3b7, (constants.%DestroyImpl.impl_witness.cb5, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.fbe]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%tuple.type.3b7, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.fbe]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%v.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13(%self.param: ref %tuple.type.3b7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %tuple.elem2: ref %i32 = tuple_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc13_3.1: %Destroy.type = converted constants.%i32, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem2)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1: ref %tuple.type.abc = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc13_3.3: %Destroy.type = facet_value constants.%tuple.type.abc, (constants.%DestroyImpl.impl_witness.62e, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.c1b]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%tuple.type.abc, %Destroy.facet.loc13_3.3 [concrete = constants.%Destroy.facet.c1b]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.4: %Destroy.type = facet_value constants.%tuple.type.abc, (constants.%DestroyImpl.impl_witness.62e, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.c1b]
// CHECK:STDOUT:   %.loc13_3.4: %Destroy.type = converted constants.%tuple.type.abc, %Destroy.facet.loc13_3.4 [concrete = constants.%Destroy.facet.c1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_3.5: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc13_3.5: %Destroy.type = converted constants.%i32, %Destroy.facet.loc13_3.5 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.6: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc13_3.6: %Destroy.type = converted constants.%i32, %Destroy.facet.loc13_3.6 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13_3.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
