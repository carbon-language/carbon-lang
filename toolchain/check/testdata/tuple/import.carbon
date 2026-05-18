// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/import.carbon

// --- implicit.carbon

package Implicit;

var a_ref: (i32,) = (0,);
var b_ref: (((i32,), i32), (i32, i32)) = (((0,), 1), (2, 3));

class C(X:! (i32, i32)) {}

fn F() -> C((1, 2));

// --- implicit.impl.carbon

impl package Implicit;

//@dump-sem-ir-begin
var a: (i32,) = a_ref;
var b: (((i32,), i32), (i32, i32)) = b_ref;
var c: C((1, 2)) = F();
//@dump-sem-ir-end

// --- fail_bad_type.impl.carbon

impl package Implicit;

// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE+8]]:14: error: cannot initialize tuple of 2 elements from tuple with 3 elements [TupleInitElementCountMismatch]
// CHECK:STDERR: var c_bad: C((1, 2, 3)) = F();
// CHECK:STDERR:              ^~~~~~~~~
// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: implicit.carbon:7:9: note: initializing generic parameter `X` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(X:! (i32, i32)) {}
// CHECK:STDERR:         ^~~~~~~~~~~~~~
// CHECK:STDERR:
var c_bad: C((1, 2, 3)) = F();

// --- fail_bad_value.impl.carbon

impl package Implicit;

// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `C((1, 2))` to `C((3, 4))` [ConversionFailure]
// CHECK:STDERR: var c_bad: C((3, 4)) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+4]]:1: note: type `C((1, 2))` does not implement interface `Core.ImplicitAs(C((3, 4)))` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: var c_bad: C((3, 4)) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var c_bad: C((3, 4)) = F();

// --- symbolic_decl.carbon

library "[[@TEST_NAME]]";

interface I {
  let value:! (i32,);
}

class C {}

impl C as I where .value = (1,) {}

// --- import_symbolic_decl.carbon

library "[[@TEST_NAME]]";
import library "symbolic_decl";

fn F() -> (i32,) {
  //@dump-sem-ir-begin
  return (C as I).value;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.abb: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad = struct_value () [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.c38: %tuple.type.85c = tuple_value (%i32) [concrete]
// CHECK:STDOUT:   %tuple.type.4f8: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.064: type = pattern_type %tuple.type.4f8 [concrete]
// CHECK:STDOUT:   %tuple.elem0.0a1: ref %i32 = tuple_access imports.%a_ref.var, element0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.521: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.783: %Int.as.Copy.impl.Op.type.521 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f1f: <witness> = impl_witness imports.%Copy.impl_witness_table.592, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.f98: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6d3: %Int.as.Copy.impl.Op.type.f98 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f1f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.83a: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2ec: type = fn_type_with_self_type %Copy.WithSelf.Op.type.83a, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.6d3, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.8c7: type = tuple_type (%tuple.type.85c, type) [concrete]
// CHECK:STDOUT:   %tuple.71c: %tuple.type.8c7 = tuple_value (%tuple.c38, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.519: %tuple.type.24b = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.58c: type = tuple_type (%tuple.type.8c7, %tuple.type.24b) [concrete]
// CHECK:STDOUT:   %tuple.c71: %tuple.type.58c = tuple_value (%tuple.71c, %tuple.519) [concrete]
// CHECK:STDOUT:   %tuple.type.187: type = tuple_type (%tuple.type.4f8, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.e53: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.399: type = tuple_type (%tuple.type.187, %tuple.type.e53) [concrete]
// CHECK:STDOUT:   %pattern_type.23d: type = pattern_type %tuple.type.399 [concrete]
// CHECK:STDOUT:   %tuple.elem0.f3d: ref %tuple.type.187 = tuple_access imports.%b_ref.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem0.0e2: ref %tuple.type.187 = tuple_access file.%b.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem0.6f2: ref %tuple.type.4f8 = tuple_access %tuple.elem0.f3d, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem0.fa3: ref %tuple.type.4f8 = tuple_access %tuple.elem0.0e2, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem0.f1e: ref %i32 = tuple_access %tuple.elem0.6f2, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1.c52: ref %i32 = tuple_access %tuple.elem0.f3d, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem1.590: ref %i32 = tuple_access %tuple.elem0.0e2, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem1.0d8: ref %tuple.type.e53 = tuple_access imports.%b_ref.var, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem1.68d: ref %tuple.type.e53 = tuple_access file.%b.var, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem0.e39: ref %i32 = tuple_access %tuple.elem1.0d8, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem0.7b2: ref %i32 = tuple_access %tuple.elem1.68d, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1.e26: ref %i32 = tuple_access %tuple.elem1.0d8, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem1.736: ref %i32 = tuple_access %tuple.elem1.68d, element1 [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ef: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.203: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.554, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ca: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ca = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.0ef = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.203) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.9f3: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.6cf: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.9f3, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.1c4: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d72: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.099: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.c74: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.cb7: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.21c: %tuple.type.e53 = tuple_value (%int_1.d72, %int_2.cb7) [concrete]
// CHECK:STDOUT:   %C.683: type = class_type @C, @C(%tuple.21c) [concrete]
// CHECK:STDOUT:   %pattern_type.271: type = pattern_type %C.683 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Implicit.a_ref: ref %tuple.type.4f8 = import_ref Implicit//default, a_ref, loaded [concrete = %a_ref.var]
// CHECK:STDOUT:   %Implicit.b_ref: ref %tuple.type.399 = import_ref Implicit//default, b_ref, loaded [concrete = %b_ref.var]
// CHECK:STDOUT:   %Implicit.C: %C.type = import_ref Implicit//default, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Implicit.F: %F.type = import_ref Implicit//default, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Implicit.import_ref.a02: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad) = import_ref Implicit//default, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.abb)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.554 = impl_witness_table (%Implicit.import_ref.a02), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %a_ref.patt: %pattern_type.064 = ref_binding_pattern a_ref [concrete]
// CHECK:STDOUT:   %a_ref.var_patt: %pattern_type.064 = var_pattern %a_ref.patt [concrete]
// CHECK:STDOUT:   %a_ref.var: ref %tuple.type.4f8 = var %a_ref.var_patt [concrete]
// CHECK:STDOUT:   %Core.import_ref.0bb: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.521) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.783)]
// CHECK:STDOUT:   %Copy.impl_witness_table.592 = impl_witness_table (%Core.import_ref.0bb), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %b_ref.patt: %pattern_type.23d = ref_binding_pattern b_ref [concrete]
// CHECK:STDOUT:   %b_ref.var_patt: %pattern_type.23d = var_pattern %b_ref.patt [concrete]
// CHECK:STDOUT:   %b_ref.var: ref %tuple.type.399 = var %b_ref.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.064 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.064 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %tuple.type.4f8 = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc5_13.1: type = splice_block %.loc5_13.3 [concrete = constants.%tuple.type.4f8] {
// CHECK:STDOUT:     %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_13.2: %tuple.type.85c = tuple_literal (%i32.loc5) [concrete = constants.%tuple.c38]
// CHECK:STDOUT:     %.loc5_13.3: type = converted %.loc5_13.2, constants.%tuple.type.4f8 [concrete = constants.%tuple.type.4f8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %tuple.type.4f8 = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.23d = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.23d = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %tuple.type.399 = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc6_34.1: type = splice_block %.loc6_34.6 [concrete = constants.%tuple.type.399] {
// CHECK:STDOUT:     %i32.loc6_11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_15: %tuple.type.85c = tuple_literal (%i32.loc6_11) [concrete = constants.%tuple.c38]
// CHECK:STDOUT:     %i32.loc6_18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_21: %tuple.type.8c7 = tuple_literal (%.loc6_15, %i32.loc6_18) [concrete = constants.%tuple.71c]
// CHECK:STDOUT:     %i32.loc6_25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc6_30: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_33: %tuple.type.24b = tuple_literal (%i32.loc6_25, %i32.loc6_30) [concrete = constants.%tuple.519]
// CHECK:STDOUT:     %.loc6_34.2: %tuple.type.58c = tuple_literal (%.loc6_21, %.loc6_33) [concrete = constants.%tuple.c71]
// CHECK:STDOUT:     %.loc6_34.3: type = converted constants.%tuple.c38, constants.%tuple.type.4f8 [concrete = constants.%tuple.type.4f8]
// CHECK:STDOUT:     %.loc6_34.4: type = converted constants.%tuple.71c, constants.%tuple.type.187 [concrete = constants.%tuple.type.187]
// CHECK:STDOUT:     %.loc6_34.5: type = converted constants.%tuple.519, constants.%tuple.type.e53 [concrete = constants.%tuple.type.e53]
// CHECK:STDOUT:     %.loc6_34.6: type = converted %.loc6_34.2, constants.%tuple.type.399 [concrete = constants.%tuple.type.399]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %tuple.type.399 = ref_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.271 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.271 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.683 = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc7_16.1: type = splice_block %C [concrete = constants.%C.683] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Implicit.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc7_15.1: %tuple.type.f94 = tuple_literal (%int_1, %int_2) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:     %impl.elem0.loc7_15.1: %.6cf = impl_witness_access constants.%ImplicitAs.impl_witness.203, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:     %bound_method.loc7_15.1: <bound method> = bound_method %int_1, %impl.elem0.loc7_15.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b7]
// CHECK:STDOUT:     %specific_fn.loc7_15.1: <specific function> = specific_function %impl.elem0.loc7_15.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc7_15.2: <bound method> = bound_method %int_1, %specific_fn.loc7_15.1 [concrete = constants.%bound_method.1c4]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_15.1: init %i32 = call %bound_method.loc7_15.2(%int_1) [concrete = constants.%int_1.d72]
// CHECK:STDOUT:     %.loc7_15.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_15.1 [concrete = constants.%int_1.d72]
// CHECK:STDOUT:     %.loc7_15.3: %i32 = converted %int_1, %.loc7_15.2 [concrete = constants.%int_1.d72]
// CHECK:STDOUT:     %impl.elem0.loc7_15.2: %.6cf = impl_witness_access constants.%ImplicitAs.impl_witness.203, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:     %bound_method.loc7_15.3: <bound method> = bound_method %int_2, %impl.elem0.loc7_15.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.099]
// CHECK:STDOUT:     %specific_fn.loc7_15.2: <specific function> = specific_function %impl.elem0.loc7_15.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc7_15.4: <bound method> = bound_method %int_2, %specific_fn.loc7_15.2 [concrete = constants.%bound_method.c74]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_15.2: init %i32 = call %bound_method.loc7_15.4(%int_2) [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:     %.loc7_15.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_15.2 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:     %.loc7_15.5: %i32 = converted %int_2, %.loc7_15.4 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:     %tuple: %tuple.type.e53 = tuple_value (%.loc7_15.3, %.loc7_15.5) [concrete = constants.%tuple.21c]
// CHECK:STDOUT:     %.loc7_16.2: %tuple.type.e53 = converted %.loc7_15.1, %tuple [concrete = constants.%tuple.21c]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%tuple.21c) [concrete = constants.%C.683]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.683 = ref_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %tuple.type.4f8 = name_ref a_ref, imports.%Implicit.a_ref [concrete = imports.%a_ref.var]
// CHECK:STDOUT:   %tuple.elem0.loc5: ref %i32 = tuple_access %a_ref.ref, element0 [concrete = constants.%tuple.elem0.0a1]
// CHECK:STDOUT:   %.loc5_17.1: %i32 = acquire_value %tuple.elem0.loc5
// CHECK:STDOUT:   %impl.elem0.loc5: %.2ec = impl_witness_access constants.%Copy.impl_witness.f1f, element0 [concrete = constants.%Int.as.Copy.impl.Op.6d3]
// CHECK:STDOUT:   %bound_method.loc5_17.1: <bound method> = bound_method %.loc5_17.1, %impl.elem0.loc5
// CHECK:STDOUT:   %specific_fn.loc5: <specific function> = specific_function %impl.elem0.loc5, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_17.2: <bound method> = bound_method %.loc5_17.1, %specific_fn.loc5
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc5: init %i32 = call %bound_method.loc5_17.2(%.loc5_17.1)
// CHECK:STDOUT:   %.loc5_17.2: init %tuple.type.4f8 to file.%a.var = tuple_init (%Int.as.Copy.impl.Op.call.loc5)
// CHECK:STDOUT:   %.loc5_1: init %tuple.type.4f8 = converted %a_ref.ref, %.loc5_17.2
// CHECK:STDOUT:   assign file.%a.var, %.loc5_1
// CHECK:STDOUT:   %b_ref.ref: ref %tuple.type.399 = name_ref b_ref, imports.%Implicit.b_ref [concrete = imports.%b_ref.var]
// CHECK:STDOUT:   %tuple.elem0.loc6_38.1: ref %tuple.type.187 = tuple_access %b_ref.ref, element0 [concrete = constants.%tuple.elem0.f3d]
// CHECK:STDOUT:   %tuple.elem0.loc6_38.2: ref %tuple.type.4f8 = tuple_access %tuple.elem0.loc6_38.1, element0 [concrete = constants.%tuple.elem0.6f2]
// CHECK:STDOUT:   %tuple.elem0.loc6_38.3: ref %i32 = tuple_access %tuple.elem0.loc6_38.2, element0 [concrete = constants.%tuple.elem0.f1e]
// CHECK:STDOUT:   %.loc6_38.1: %i32 = acquire_value %tuple.elem0.loc6_38.3
// CHECK:STDOUT:   %impl.elem0.loc6_38.1: %.2ec = impl_witness_access constants.%Copy.impl_witness.f1f, element0 [concrete = constants.%Int.as.Copy.impl.Op.6d3]
// CHECK:STDOUT:   %bound_method.loc6_38.1: <bound method> = bound_method %.loc6_38.1, %impl.elem0.loc6_38.1
// CHECK:STDOUT:   %specific_fn.loc6_38.1: <specific function> = specific_function %impl.elem0.loc6_38.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_38.2: <bound method> = bound_method %.loc6_38.1, %specific_fn.loc6_38.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_38.1: init %i32 = call %bound_method.loc6_38.2(%.loc6_38.1)
// CHECK:STDOUT:   %tuple.elem0.loc6_38.4: ref %tuple.type.187 = tuple_access file.%b.var, element0 [concrete = constants.%tuple.elem0.0e2]
// CHECK:STDOUT:   %tuple.elem0.loc6_38.5: ref %tuple.type.4f8 = tuple_access %tuple.elem0.loc6_38.4, element0 [concrete = constants.%tuple.elem0.fa3]
// CHECK:STDOUT:   %.loc6_38.2: init %tuple.type.4f8 to %tuple.elem0.loc6_38.5 = tuple_init (%Int.as.Copy.impl.Op.call.loc6_38.1)
// CHECK:STDOUT:   %.loc6_38.3: init %tuple.type.4f8 = converted %tuple.elem0.loc6_38.2, %.loc6_38.2
// CHECK:STDOUT:   %.loc6_38.4: init %tuple.type.4f8 to %tuple.elem0.loc6_38.5 = in_place_init %.loc6_38.3
// CHECK:STDOUT:   %tuple.elem1.loc6_38.1: ref %i32 = tuple_access %tuple.elem0.loc6_38.1, element1 [concrete = constants.%tuple.elem1.c52]
// CHECK:STDOUT:   %.loc6_38.5: %i32 = acquire_value %tuple.elem1.loc6_38.1
// CHECK:STDOUT:   %impl.elem0.loc6_38.2: %.2ec = impl_witness_access constants.%Copy.impl_witness.f1f, element0 [concrete = constants.%Int.as.Copy.impl.Op.6d3]
// CHECK:STDOUT:   %bound_method.loc6_38.3: <bound method> = bound_method %.loc6_38.5, %impl.elem0.loc6_38.2
// CHECK:STDOUT:   %specific_fn.loc6_38.2: <specific function> = specific_function %impl.elem0.loc6_38.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_38.4: <bound method> = bound_method %.loc6_38.5, %specific_fn.loc6_38.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_38.2: init %i32 = call %bound_method.loc6_38.4(%.loc6_38.5)
// CHECK:STDOUT:   %tuple.elem1.loc6_38.2: ref %i32 = tuple_access %tuple.elem0.loc6_38.4, element1 [concrete = constants.%tuple.elem1.590]
// CHECK:STDOUT:   %.loc6_38.6: init %i32 to %tuple.elem1.loc6_38.2 = in_place_init %Int.as.Copy.impl.Op.call.loc6_38.2
// CHECK:STDOUT:   %.loc6_38.7: init %tuple.type.187 to %tuple.elem0.loc6_38.4 = tuple_init (%.loc6_38.4, %.loc6_38.6)
// CHECK:STDOUT:   %.loc6_38.8: init %tuple.type.187 = converted %tuple.elem0.loc6_38.1, %.loc6_38.7
// CHECK:STDOUT:   %tuple.elem1.loc6_38.3: ref %tuple.type.e53 = tuple_access %b_ref.ref, element1 [concrete = constants.%tuple.elem1.0d8]
// CHECK:STDOUT:   %tuple.elem0.loc6_38.6: ref %i32 = tuple_access %tuple.elem1.loc6_38.3, element0 [concrete = constants.%tuple.elem0.e39]
// CHECK:STDOUT:   %.loc6_38.9: %i32 = acquire_value %tuple.elem0.loc6_38.6
// CHECK:STDOUT:   %impl.elem0.loc6_38.3: %.2ec = impl_witness_access constants.%Copy.impl_witness.f1f, element0 [concrete = constants.%Int.as.Copy.impl.Op.6d3]
// CHECK:STDOUT:   %bound_method.loc6_38.5: <bound method> = bound_method %.loc6_38.9, %impl.elem0.loc6_38.3
// CHECK:STDOUT:   %specific_fn.loc6_38.3: <specific function> = specific_function %impl.elem0.loc6_38.3, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_38.6: <bound method> = bound_method %.loc6_38.9, %specific_fn.loc6_38.3
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_38.3: init %i32 = call %bound_method.loc6_38.6(%.loc6_38.9)
// CHECK:STDOUT:   %tuple.elem1.loc6_38.4: ref %tuple.type.e53 = tuple_access file.%b.var, element1 [concrete = constants.%tuple.elem1.68d]
// CHECK:STDOUT:   %tuple.elem0.loc6_38.7: ref %i32 = tuple_access %tuple.elem1.loc6_38.4, element0 [concrete = constants.%tuple.elem0.7b2]
// CHECK:STDOUT:   %.loc6_38.10: init %i32 to %tuple.elem0.loc6_38.7 = in_place_init %Int.as.Copy.impl.Op.call.loc6_38.3
// CHECK:STDOUT:   %tuple.elem1.loc6_38.5: ref %i32 = tuple_access %tuple.elem1.loc6_38.3, element1 [concrete = constants.%tuple.elem1.e26]
// CHECK:STDOUT:   %.loc6_38.11: %i32 = acquire_value %tuple.elem1.loc6_38.5
// CHECK:STDOUT:   %impl.elem0.loc6_38.4: %.2ec = impl_witness_access constants.%Copy.impl_witness.f1f, element0 [concrete = constants.%Int.as.Copy.impl.Op.6d3]
// CHECK:STDOUT:   %bound_method.loc6_38.7: <bound method> = bound_method %.loc6_38.11, %impl.elem0.loc6_38.4
// CHECK:STDOUT:   %specific_fn.loc6_38.4: <specific function> = specific_function %impl.elem0.loc6_38.4, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_38.8: <bound method> = bound_method %.loc6_38.11, %specific_fn.loc6_38.4
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_38.4: init %i32 = call %bound_method.loc6_38.8(%.loc6_38.11)
// CHECK:STDOUT:   %tuple.elem1.loc6_38.6: ref %i32 = tuple_access %tuple.elem1.loc6_38.4, element1 [concrete = constants.%tuple.elem1.736]
// CHECK:STDOUT:   %.loc6_38.12: init %i32 to %tuple.elem1.loc6_38.6 = in_place_init %Int.as.Copy.impl.Op.call.loc6_38.4
// CHECK:STDOUT:   %.loc6_38.13: init %tuple.type.e53 to %tuple.elem1.loc6_38.4 = tuple_init (%.loc6_38.10, %.loc6_38.12)
// CHECK:STDOUT:   %.loc6_38.14: init %tuple.type.e53 = converted %tuple.elem1.loc6_38.3, %.loc6_38.13
// CHECK:STDOUT:   %.loc6_38.15: init %tuple.type.399 to file.%b.var = tuple_init (%.loc6_38.8, %.loc6_38.14)
// CHECK:STDOUT:   %.loc6_1: init %tuple.type.399 = converted %b_ref.ref, %.loc6_38.15
// CHECK:STDOUT:   assign file.%b.var, %.loc6_1
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Implicit.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc7: ref %C.683 = splice_block file.%c.var [concrete = file.%c.var] {}
// CHECK:STDOUT:   %F.call: init %C.683 to %.loc7 = call %F.ref()
// CHECK:STDOUT:   assign file.%c.var, %F.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_symbolic_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.4f8: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %int_1: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.d42: %tuple.type.4f8 = tuple_value (%int_1) [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.15f: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.cc8 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.521: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.783: %Int.as.Copy.impl.Op.type.521 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f1f: <witness> = impl_witness imports.%Copy.impl_witness_table.592, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.f98: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6d3: %Int.as.Copy.impl.Op.type.f98 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f1f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.83a: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2ec: type = fn_type_with_self_type %Copy.WithSelf.Op.type.83a, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1, %Int.as.Copy.impl.Op.6d3 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.6d3, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//symbolic_decl, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//symbolic_decl, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.1a9: %I.assoc_type = import_ref Main//symbolic_decl, loc5_12, loaded [concrete = constants.%assoc0.15f]
// CHECK:STDOUT:   %Main.import_ref.12c: %tuple.type.4f8 = import_ref Main//symbolic_decl, loc10_33, loaded [concrete = constants.%tuple.d42]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.12c), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.cc8: %tuple.type.4f8 = import_ref Main//symbolic_decl, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: %tuple.type.4f8 = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.0bb: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.521) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.783)]
// CHECK:STDOUT:   %Copy.impl_witness_table.592 = impl_witness_table (%Core.import_ref.0bb), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %tuple.type.4f8 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18.1: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.1a9 [concrete = constants.%assoc0.15f]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: %tuple.type.4f8 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%tuple.d42]
// CHECK:STDOUT:   %tuple.elem0: %i32 = tuple_access %impl.elem0.loc7_18.1, element0 [concrete = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.2ec = impl_witness_access constants.%Copy.impl_witness.f1f, element0 [concrete = constants.%Int.as.Copy.impl.Op.6d3]
// CHECK:STDOUT:   %bound_method.loc7_18.1: <bound method> = bound_method %tuple.elem0, %impl.elem0.loc7_18.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc7_18.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_18.2: <bound method> = bound_method %tuple.elem0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc7_18.2(%tuple.elem0) [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc7_18.2: init %tuple.type.4f8 = tuple_init (%Int.as.Copy.impl.Op.call) [concrete = constants.%tuple.d42]
// CHECK:STDOUT:   %.loc7_24: init %tuple.type.4f8 = converted %impl.elem0.loc7_18.1, %.loc7_18.2 [concrete = constants.%tuple.d42]
// CHECK:STDOUT:   return %.loc7_24
// CHECK:STDOUT: }
// CHECK:STDOUT:
