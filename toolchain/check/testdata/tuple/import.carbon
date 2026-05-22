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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b90: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.061: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b90 = struct_value () [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.b71: %tuple.type.85c = tuple_value (%i32) [concrete]
// CHECK:STDOUT:   %tuple.type.597: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.e93: type = pattern_type %tuple.type.597 [concrete]
// CHECK:STDOUT:   %tuple.elem0.149: ref %i32 = tuple_access imports.%a_ref.var, element0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.c59: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.fb0: %Int.as.Copy.impl.Op.type.c59 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.c36: <witness> = impl_witness imports.%Copy.impl_witness_table.03a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.e20: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.123: %Int.as.Copy.impl.Op.type.e20 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.c36) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.56d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.cf6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.56d, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.123, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.8c7: type = tuple_type (%tuple.type.85c, type) [concrete]
// CHECK:STDOUT:   %tuple.838: %tuple.type.8c7 = tuple_value (%tuple.b71, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.f66: %tuple.type.24b = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.58c: type = tuple_type (%tuple.type.8c7, %tuple.type.24b) [concrete]
// CHECK:STDOUT:   %tuple.530: %tuple.type.58c = tuple_value (%tuple.838, %tuple.f66) [concrete]
// CHECK:STDOUT:   %tuple.type.d0b: type = tuple_type (%tuple.type.597, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.e6c: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.7f6: type = tuple_type (%tuple.type.d0b, %tuple.type.e6c) [concrete]
// CHECK:STDOUT:   %pattern_type.697: type = pattern_type %tuple.type.7f6 [concrete]
// CHECK:STDOUT:   %tuple.elem0.804: ref %tuple.type.d0b = tuple_access imports.%b_ref.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem0.8f4: ref %tuple.type.d0b = tuple_access file.%b.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem0.f74: ref %tuple.type.597 = tuple_access %tuple.elem0.804, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem0.bb5: ref %tuple.type.597 = tuple_access %tuple.elem0.8f4, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem0.032: ref %i32 = tuple_access %tuple.elem0.f74, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1.e77: ref %i32 = tuple_access %tuple.elem0.804, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem1.9c7: ref %i32 = tuple_access %tuple.elem0.8f4, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem1.af9: ref %tuple.type.e6c = tuple_access imports.%b_ref.var, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem1.3c8: ref %tuple.type.e6c = tuple_access file.%b.var, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem0.1a8: ref %i32 = tuple_access %tuple.elem1.af9, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem0.b1a: ref %i32 = tuple_access %tuple.elem1.3c8, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1.528: ref %i32 = tuple_access %tuple.elem1.af9, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem1.b9f: ref %i32 = tuple_access %tuple.elem1.3c8, element1 [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.73f: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9a1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.097, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.972: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.536: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.972 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.73f = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.9a1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a60: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.fbd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a60, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.483: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.536 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.536, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.6cf: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.50a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9b3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.536 [concrete]
// CHECK:STDOUT:   %bound_method.aad: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.21f: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.a40: %tuple.type.e6c = tuple_value (%int_1.50a, %int_2.21f) [concrete]
// CHECK:STDOUT:   %C.5c2: type = class_type @C, @C(%tuple.a40) [concrete]
// CHECK:STDOUT:   %pattern_type.104: type = pattern_type %C.5c2 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Implicit.a_ref: ref %tuple.type.597 = import_ref Implicit//default, a_ref, loaded [concrete = %a_ref.var]
// CHECK:STDOUT:   %Implicit.b_ref: ref %tuple.type.7f6 = import_ref Implicit//default, b_ref, loaded [concrete = %b_ref.var]
// CHECK:STDOUT:   %Implicit.C: %C.type = import_ref Implicit//default, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Implicit.F: %F.type = import_ref Implicit//default, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Implicit.import_ref.85c: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b90) = import_ref Implicit//default, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.061)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.097 = impl_witness_table (%Implicit.import_ref.85c), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %a_ref.patt: %pattern_type.e93 = ref_binding_pattern a_ref [concrete]
// CHECK:STDOUT:   %a_ref.var_patt: %pattern_type.e93 = var_pattern %a_ref.patt [concrete]
// CHECK:STDOUT:   %a_ref.var: ref %tuple.type.597 = var %a_ref.var_patt [concrete]
// CHECK:STDOUT:   %Core.import_ref.8e5: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.c59) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.fb0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.03a = impl_witness_table (%Core.import_ref.8e5), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %b_ref.patt: %pattern_type.697 = ref_binding_pattern b_ref [concrete]
// CHECK:STDOUT:   %b_ref.var_patt: %pattern_type.697 = var_pattern %b_ref.patt [concrete]
// CHECK:STDOUT:   %b_ref.var: ref %tuple.type.7f6 = var %b_ref.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.e93 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.e93 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %tuple.type.597 = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc5_13.1: type = splice_block %.loc5_13.3 [concrete = constants.%tuple.type.597] {
// CHECK:STDOUT:     %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_13.2: %tuple.type.85c = tuple_literal (%i32.loc5) [concrete = constants.%tuple.b71]
// CHECK:STDOUT:     %.loc5_13.3: type = converted %.loc5_13.2, constants.%tuple.type.597 [concrete = constants.%tuple.type.597]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %tuple.type.597 = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.697 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.697 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %tuple.type.7f6 = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc6_34.1: type = splice_block %.loc6_34.6 [concrete = constants.%tuple.type.7f6] {
// CHECK:STDOUT:     %i32.loc6_11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_15: %tuple.type.85c = tuple_literal (%i32.loc6_11) [concrete = constants.%tuple.b71]
// CHECK:STDOUT:     %i32.loc6_18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_21: %tuple.type.8c7 = tuple_literal (%.loc6_15, %i32.loc6_18) [concrete = constants.%tuple.838]
// CHECK:STDOUT:     %i32.loc6_25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc6_30: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_33: %tuple.type.24b = tuple_literal (%i32.loc6_25, %i32.loc6_30) [concrete = constants.%tuple.f66]
// CHECK:STDOUT:     %.loc6_34.2: %tuple.type.58c = tuple_literal (%.loc6_21, %.loc6_33) [concrete = constants.%tuple.530]
// CHECK:STDOUT:     %.loc6_34.3: type = converted constants.%tuple.b71, constants.%tuple.type.597 [concrete = constants.%tuple.type.597]
// CHECK:STDOUT:     %.loc6_34.4: type = converted constants.%tuple.838, constants.%tuple.type.d0b [concrete = constants.%tuple.type.d0b]
// CHECK:STDOUT:     %.loc6_34.5: type = converted constants.%tuple.f66, constants.%tuple.type.e6c [concrete = constants.%tuple.type.e6c]
// CHECK:STDOUT:     %.loc6_34.6: type = converted %.loc6_34.2, constants.%tuple.type.7f6 [concrete = constants.%tuple.type.7f6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %tuple.type.7f6 = ref_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.104 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.104 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.5c2 = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc7_16.1: type = splice_block %C [concrete = constants.%C.5c2] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Implicit.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc7_15.1: %tuple.type.f94 = tuple_literal (%int_1, %int_2) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:     %impl.elem0.loc7_15.1: %.fbd = impl_witness_access constants.%ImplicitAs.impl_witness.9a1, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.536]
// CHECK:STDOUT:     %bound_method.loc7_15.1: <bound method> = bound_method %int_1, %impl.elem0.loc7_15.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.483]
// CHECK:STDOUT:     %specific_fn.loc7_15.1: <specific function> = specific_function %impl.elem0.loc7_15.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc7_15.2: <bound method> = bound_method %int_1, %specific_fn.loc7_15.1 [concrete = constants.%bound_method.6cf]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_15.1: init %i32 = call %bound_method.loc7_15.2(%int_1) [concrete = constants.%int_1.50a]
// CHECK:STDOUT:     %.loc7_15.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_15.1 [concrete = constants.%int_1.50a]
// CHECK:STDOUT:     %.loc7_15.3: %i32 = converted %int_1, %.loc7_15.2 [concrete = constants.%int_1.50a]
// CHECK:STDOUT:     %impl.elem0.loc7_15.2: %.fbd = impl_witness_access constants.%ImplicitAs.impl_witness.9a1, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.536]
// CHECK:STDOUT:     %bound_method.loc7_15.3: <bound method> = bound_method %int_2, %impl.elem0.loc7_15.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9b3]
// CHECK:STDOUT:     %specific_fn.loc7_15.2: <specific function> = specific_function %impl.elem0.loc7_15.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc7_15.4: <bound method> = bound_method %int_2, %specific_fn.loc7_15.2 [concrete = constants.%bound_method.aad]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_15.2: init %i32 = call %bound_method.loc7_15.4(%int_2) [concrete = constants.%int_2.21f]
// CHECK:STDOUT:     %.loc7_15.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_15.2 [concrete = constants.%int_2.21f]
// CHECK:STDOUT:     %.loc7_15.5: %i32 = converted %int_2, %.loc7_15.4 [concrete = constants.%int_2.21f]
// CHECK:STDOUT:     %tuple: %tuple.type.e6c = tuple_value (%.loc7_15.3, %.loc7_15.5) [concrete = constants.%tuple.a40]
// CHECK:STDOUT:     %.loc7_16.2: %tuple.type.e6c = converted %.loc7_15.1, %tuple [concrete = constants.%tuple.a40]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%tuple.a40) [concrete = constants.%C.5c2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.5c2 = ref_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %tuple.type.597 = name_ref a_ref, imports.%Implicit.a_ref [concrete = imports.%a_ref.var]
// CHECK:STDOUT:   %tuple.elem0.loc5: ref %i32 = tuple_access %a_ref.ref, element0 [concrete = constants.%tuple.elem0.149]
// CHECK:STDOUT:   %.loc5_17.1: %i32 = acquire_value %tuple.elem0.loc5
// CHECK:STDOUT:   %impl.elem0.loc5: %.cf6 = impl_witness_access constants.%Copy.impl_witness.c36, element0 [concrete = constants.%Int.as.Copy.impl.Op.123]
// CHECK:STDOUT:   %bound_method.loc5_17.1: <bound method> = bound_method %.loc5_17.1, %impl.elem0.loc5
// CHECK:STDOUT:   %specific_fn.loc5: <specific function> = specific_function %impl.elem0.loc5, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_17.2: <bound method> = bound_method %.loc5_17.1, %specific_fn.loc5
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc5: init %i32 = call %bound_method.loc5_17.2(%.loc5_17.1)
// CHECK:STDOUT:   %.loc5_17.2: init %tuple.type.597 to file.%a.var = tuple_init (%Int.as.Copy.impl.Op.call.loc5)
// CHECK:STDOUT:   %.loc5_1: init %tuple.type.597 = converted %a_ref.ref, %.loc5_17.2
// CHECK:STDOUT:   assign file.%a.var, %.loc5_1
// CHECK:STDOUT:   %b_ref.ref: ref %tuple.type.7f6 = name_ref b_ref, imports.%Implicit.b_ref [concrete = imports.%b_ref.var]
// CHECK:STDOUT:   %tuple.elem0.loc6_38.1: ref %tuple.type.d0b = tuple_access %b_ref.ref, element0 [concrete = constants.%tuple.elem0.804]
// CHECK:STDOUT:   %tuple.elem0.loc6_38.2: ref %tuple.type.597 = tuple_access %tuple.elem0.loc6_38.1, element0 [concrete = constants.%tuple.elem0.f74]
// CHECK:STDOUT:   %tuple.elem0.loc6_38.3: ref %i32 = tuple_access %tuple.elem0.loc6_38.2, element0 [concrete = constants.%tuple.elem0.032]
// CHECK:STDOUT:   %.loc6_38.1: %i32 = acquire_value %tuple.elem0.loc6_38.3
// CHECK:STDOUT:   %impl.elem0.loc6_38.1: %.cf6 = impl_witness_access constants.%Copy.impl_witness.c36, element0 [concrete = constants.%Int.as.Copy.impl.Op.123]
// CHECK:STDOUT:   %bound_method.loc6_38.1: <bound method> = bound_method %.loc6_38.1, %impl.elem0.loc6_38.1
// CHECK:STDOUT:   %specific_fn.loc6_38.1: <specific function> = specific_function %impl.elem0.loc6_38.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_38.2: <bound method> = bound_method %.loc6_38.1, %specific_fn.loc6_38.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_38.1: init %i32 = call %bound_method.loc6_38.2(%.loc6_38.1)
// CHECK:STDOUT:   %tuple.elem0.loc6_38.4: ref %tuple.type.d0b = tuple_access file.%b.var, element0 [concrete = constants.%tuple.elem0.8f4]
// CHECK:STDOUT:   %tuple.elem0.loc6_38.5: ref %tuple.type.597 = tuple_access %tuple.elem0.loc6_38.4, element0 [concrete = constants.%tuple.elem0.bb5]
// CHECK:STDOUT:   %.loc6_38.2: init %tuple.type.597 to %tuple.elem0.loc6_38.5 = tuple_init (%Int.as.Copy.impl.Op.call.loc6_38.1)
// CHECK:STDOUT:   %.loc6_38.3: init %tuple.type.597 = converted %tuple.elem0.loc6_38.2, %.loc6_38.2
// CHECK:STDOUT:   %.loc6_38.4: init %tuple.type.597 to %tuple.elem0.loc6_38.5 = in_place_init %.loc6_38.3
// CHECK:STDOUT:   %tuple.elem1.loc6_38.1: ref %i32 = tuple_access %tuple.elem0.loc6_38.1, element1 [concrete = constants.%tuple.elem1.e77]
// CHECK:STDOUT:   %.loc6_38.5: %i32 = acquire_value %tuple.elem1.loc6_38.1
// CHECK:STDOUT:   %impl.elem0.loc6_38.2: %.cf6 = impl_witness_access constants.%Copy.impl_witness.c36, element0 [concrete = constants.%Int.as.Copy.impl.Op.123]
// CHECK:STDOUT:   %bound_method.loc6_38.3: <bound method> = bound_method %.loc6_38.5, %impl.elem0.loc6_38.2
// CHECK:STDOUT:   %specific_fn.loc6_38.2: <specific function> = specific_function %impl.elem0.loc6_38.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_38.4: <bound method> = bound_method %.loc6_38.5, %specific_fn.loc6_38.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_38.2: init %i32 = call %bound_method.loc6_38.4(%.loc6_38.5)
// CHECK:STDOUT:   %tuple.elem1.loc6_38.2: ref %i32 = tuple_access %tuple.elem0.loc6_38.4, element1 [concrete = constants.%tuple.elem1.9c7]
// CHECK:STDOUT:   %.loc6_38.6: init %i32 to %tuple.elem1.loc6_38.2 = in_place_init %Int.as.Copy.impl.Op.call.loc6_38.2
// CHECK:STDOUT:   %.loc6_38.7: init %tuple.type.d0b to %tuple.elem0.loc6_38.4 = tuple_init (%.loc6_38.4, %.loc6_38.6)
// CHECK:STDOUT:   %.loc6_38.8: init %tuple.type.d0b = converted %tuple.elem0.loc6_38.1, %.loc6_38.7
// CHECK:STDOUT:   %tuple.elem1.loc6_38.3: ref %tuple.type.e6c = tuple_access %b_ref.ref, element1 [concrete = constants.%tuple.elem1.af9]
// CHECK:STDOUT:   %tuple.elem0.loc6_38.6: ref %i32 = tuple_access %tuple.elem1.loc6_38.3, element0 [concrete = constants.%tuple.elem0.1a8]
// CHECK:STDOUT:   %.loc6_38.9: %i32 = acquire_value %tuple.elem0.loc6_38.6
// CHECK:STDOUT:   %impl.elem0.loc6_38.3: %.cf6 = impl_witness_access constants.%Copy.impl_witness.c36, element0 [concrete = constants.%Int.as.Copy.impl.Op.123]
// CHECK:STDOUT:   %bound_method.loc6_38.5: <bound method> = bound_method %.loc6_38.9, %impl.elem0.loc6_38.3
// CHECK:STDOUT:   %specific_fn.loc6_38.3: <specific function> = specific_function %impl.elem0.loc6_38.3, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_38.6: <bound method> = bound_method %.loc6_38.9, %specific_fn.loc6_38.3
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_38.3: init %i32 = call %bound_method.loc6_38.6(%.loc6_38.9)
// CHECK:STDOUT:   %tuple.elem1.loc6_38.4: ref %tuple.type.e6c = tuple_access file.%b.var, element1 [concrete = constants.%tuple.elem1.3c8]
// CHECK:STDOUT:   %tuple.elem0.loc6_38.7: ref %i32 = tuple_access %tuple.elem1.loc6_38.4, element0 [concrete = constants.%tuple.elem0.b1a]
// CHECK:STDOUT:   %.loc6_38.10: init %i32 to %tuple.elem0.loc6_38.7 = in_place_init %Int.as.Copy.impl.Op.call.loc6_38.3
// CHECK:STDOUT:   %tuple.elem1.loc6_38.5: ref %i32 = tuple_access %tuple.elem1.loc6_38.3, element1 [concrete = constants.%tuple.elem1.528]
// CHECK:STDOUT:   %.loc6_38.11: %i32 = acquire_value %tuple.elem1.loc6_38.5
// CHECK:STDOUT:   %impl.elem0.loc6_38.4: %.cf6 = impl_witness_access constants.%Copy.impl_witness.c36, element0 [concrete = constants.%Int.as.Copy.impl.Op.123]
// CHECK:STDOUT:   %bound_method.loc6_38.7: <bound method> = bound_method %.loc6_38.11, %impl.elem0.loc6_38.4
// CHECK:STDOUT:   %specific_fn.loc6_38.4: <specific function> = specific_function %impl.elem0.loc6_38.4, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_38.8: <bound method> = bound_method %.loc6_38.11, %specific_fn.loc6_38.4
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6_38.4: init %i32 = call %bound_method.loc6_38.8(%.loc6_38.11)
// CHECK:STDOUT:   %tuple.elem1.loc6_38.6: ref %i32 = tuple_access %tuple.elem1.loc6_38.4, element1 [concrete = constants.%tuple.elem1.b9f]
// CHECK:STDOUT:   %.loc6_38.12: init %i32 to %tuple.elem1.loc6_38.6 = in_place_init %Int.as.Copy.impl.Op.call.loc6_38.4
// CHECK:STDOUT:   %.loc6_38.13: init %tuple.type.e6c to %tuple.elem1.loc6_38.4 = tuple_init (%.loc6_38.10, %.loc6_38.12)
// CHECK:STDOUT:   %.loc6_38.14: init %tuple.type.e6c = converted %tuple.elem1.loc6_38.3, %.loc6_38.13
// CHECK:STDOUT:   %.loc6_38.15: init %tuple.type.7f6 to file.%b.var = tuple_init (%.loc6_38.8, %.loc6_38.14)
// CHECK:STDOUT:   %.loc6_1: init %tuple.type.7f6 = converted %b_ref.ref, %.loc6_38.15
// CHECK:STDOUT:   assign file.%b.var, %.loc6_1
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Implicit.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc7: ref %C.5c2 = splice_block file.%c.var [concrete = file.%c.var] {}
// CHECK:STDOUT:   %F.call: init %C.5c2 to %.loc7 = call %F.ref()
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
// CHECK:STDOUT:   %tuple.type.c87: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %int_1: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.86f: %tuple.type.c87 = tuple_value (%int_1) [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.2ac: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.48f [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.32d: <witness> = impl_witness imports.%Copy.impl_witness_table.07a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.4a0: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c4a: %Int.as.Copy.impl.Op.type.4a0 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.32d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.afe: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a5f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.afe, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1, %Int.as.Copy.impl.Op.c4a [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c4a, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//symbolic_decl, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//symbolic_decl, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.435: %I.assoc_type = import_ref Main//symbolic_decl, loc5_12, loaded [concrete = constants.%assoc0.2ac]
// CHECK:STDOUT:   %Main.import_ref.d46: %tuple.type.c87 = import_ref Main//symbolic_decl, loc10_33, loaded [concrete = constants.%tuple.86f]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.d46), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.48f: %tuple.type.c87 = import_ref Main//symbolic_decl, loc5_12, loaded [concrete = %value]
// CHECK:STDOUT:   %value: %tuple.type.c87 = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.809: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c85)]
// CHECK:STDOUT:   %Copy.impl_witness_table.07a = impl_witness_table (%Core.import_ref.809), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %tuple.type.c87 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18.1: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %value.ref: %I.assoc_type = name_ref value, imports.%Main.import_ref.435 [concrete = constants.%assoc0.2ac]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: %tuple.type.c87 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%tuple.86f]
// CHECK:STDOUT:   %tuple.elem0: %i32 = tuple_access %impl.elem0.loc7_18.1, element0 [concrete = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc7_18.1: <bound method> = bound_method %tuple.elem0, %impl.elem0.loc7_18.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc7_18.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_18.2: <bound method> = bound_method %tuple.elem0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc7_18.2(%tuple.elem0) [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc7_18.2: init %tuple.type.c87 = tuple_init (%Int.as.Copy.impl.Op.call) [concrete = constants.%tuple.86f]
// CHECK:STDOUT:   %.loc7_24: init %tuple.type.c87 = converted %impl.elem0.loc7_18.1, %.loc7_18.2 [concrete = constants.%tuple.86f]
// CHECK:STDOUT:   return %.loc7_24
// CHECK:STDOUT: }
// CHECK:STDOUT:
