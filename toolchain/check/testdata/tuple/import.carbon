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
// CHECK:STDERR:         ^
// CHECK:STDERR:
var c_bad: C((1, 2, 3)) = F();

// --- fail_bad_value.impl.carbon

impl package Implicit;

// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `C((1, 2))` to `C((3, 4))` [ConversionFailure]
// CHECK:STDERR: var c_bad: C((3, 4)) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+4]]:1: note: type `C((1, 2))` does not implement interface `Core.ImplicitAs(C((3, 4)))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var c_bad: C((3, 4)) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var c_bad: C((3, 4)) = F();

// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9a6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.458: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.c2c: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.type.dd4: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.2e8: type = pattern_type %tuple.type.dd4 [concrete]
// CHECK:STDOUT:   %tuple.elem0.994: ref %i32 = tuple_access imports.%a_ref.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.type.8c7: type = tuple_type (%tuple.type.85c, type) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.58c: type = tuple_type (%tuple.type.8c7, %tuple.type.24b) [concrete]
// CHECK:STDOUT:   %tuple.type.154: type = tuple_type (%tuple.type.dd4, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.cfa: type = tuple_type (%tuple.type.154, %tuple.type.c2c) [concrete]
// CHECK:STDOUT:   %pattern_type.ab9: type = pattern_type %tuple.type.cfa [concrete]
// CHECK:STDOUT:   %tuple.elem0.693: ref %tuple.type.154 = tuple_access imports.%b_ref.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem0.76a: ref %tuple.type.154 = tuple_access file.%b.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem0.4c9: ref %tuple.type.dd4 = tuple_access %tuple.elem0.693, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem0.6fd: ref %tuple.type.dd4 = tuple_access %tuple.elem0.76a, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem0.390: ref %i32 = tuple_access %tuple.elem0.4c9, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1.723: ref %i32 = tuple_access %tuple.elem0.693, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem1.ae3: ref %i32 = tuple_access %tuple.elem0.76a, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem1.92b: ref %tuple.type.c2c = tuple_access imports.%b_ref.var, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem1.4f3: ref %tuple.type.c2c = tuple_access file.%b.var, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem0.2ff: ref %i32 = tuple_access %tuple.elem1.92b, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem0.790: ref %i32 = tuple_access %tuple.elem1.4f3, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1.282: ref %i32 = tuple_access %tuple.elem1.92b, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem1.822: ref %i32 = tuple_access %tuple.elem1.4f3, element1 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b9e: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.ea0: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1ad, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e14: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.4cb: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e14 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.b9e = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.940: type = fn_type_with_self_type %ImplicitAs.Convert.type.ea0, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dc5: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.4cb [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.4cb, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.c37: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.47b: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.30f: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.4cb [concrete]
// CHECK:STDOUT:   %bound_method.62b: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.d0d: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.c2c = tuple_value (%int_1.47b, %int_2.d0d) [concrete]
// CHECK:STDOUT:   %C.cf0: type = class_type @C, @C(%tuple) [concrete]
// CHECK:STDOUT:   %pattern_type.7e2: type = pattern_type %C.cf0 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Implicit.a_ref: ref %tuple.type.dd4 = import_ref Implicit//default, a_ref, loaded [concrete = %a_ref.var]
// CHECK:STDOUT:   %Implicit.b_ref: ref %tuple.type.cfa = import_ref Implicit//default, b_ref, loaded [concrete = %b_ref.var]
// CHECK:STDOUT:   %Implicit.C: %C.type = import_ref Implicit//default, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Implicit.F: %F.type = import_ref Implicit//default, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Implicit.import_ref.773: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9a6) = import_ref Implicit//default, inst152 [indirect], loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.458)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1ad = impl_witness_table (%Implicit.import_ref.773), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %a_ref.patt: %pattern_type.2e8 = binding_pattern a_ref [concrete]
// CHECK:STDOUT:   %a_ref.var_patt: %pattern_type.2e8 = var_pattern %a_ref.patt [concrete]
// CHECK:STDOUT:   %a_ref.var: ref %tuple.type.dd4 = var %a_ref.var_patt [concrete]
// CHECK:STDOUT:   %b_ref.patt: %pattern_type.ab9 = binding_pattern b_ref [concrete]
// CHECK:STDOUT:   %b_ref.var_patt: %pattern_type.ab9 = var_pattern %b_ref.patt [concrete]
// CHECK:STDOUT:   %b_ref.var: ref %tuple.type.cfa = var %b_ref.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.2e8 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.2e8 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %tuple.type.dd4 = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc5_13.1: type = splice_block %.loc5_13.3 [concrete = constants.%tuple.type.dd4] {
// CHECK:STDOUT:     %int_32.loc5: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_13.2: %tuple.type.85c = tuple_literal (%i32.loc5)
// CHECK:STDOUT:     %.loc5_13.3: type = converted %.loc5_13.2, constants.%tuple.type.dd4 [concrete = constants.%tuple.type.dd4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %tuple.type.dd4 = bind_name a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.ab9 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.ab9 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %tuple.type.cfa = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc6_34.1: type = splice_block %.loc6_34.6 [concrete = constants.%tuple.type.cfa] {
// CHECK:STDOUT:     %int_32.loc6_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_15: %tuple.type.85c = tuple_literal (%i32.loc6_11)
// CHECK:STDOUT:     %int_32.loc6_18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_21: %tuple.type.8c7 = tuple_literal (%.loc6_15, %i32.loc6_18)
// CHECK:STDOUT:     %int_32.loc6_25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc6_30: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_30: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_33: %tuple.type.24b = tuple_literal (%i32.loc6_25, %i32.loc6_30)
// CHECK:STDOUT:     %.loc6_34.2: %tuple.type.58c = tuple_literal (%.loc6_21, %.loc6_33)
// CHECK:STDOUT:     %.loc6_34.3: type = converted %.loc6_15, constants.%tuple.type.dd4 [concrete = constants.%tuple.type.dd4]
// CHECK:STDOUT:     %.loc6_34.4: type = converted %.loc6_21, constants.%tuple.type.154 [concrete = constants.%tuple.type.154]
// CHECK:STDOUT:     %.loc6_34.5: type = converted %.loc6_33, constants.%tuple.type.c2c [concrete = constants.%tuple.type.c2c]
// CHECK:STDOUT:     %.loc6_34.6: type = converted %.loc6_34.2, constants.%tuple.type.cfa [concrete = constants.%tuple.type.cfa]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %tuple.type.cfa = bind_name b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.7e2 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.7e2 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.cf0 = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc7_16.1: type = splice_block %C [concrete = constants.%C.cf0] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Implicit.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc7_15.1: %tuple.type.f94 = tuple_literal (%int_1, %int_2)
// CHECK:STDOUT:     %impl.elem0.loc7_15.1: %.940 = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.4cb]
// CHECK:STDOUT:     %bound_method.loc7_15.1: <bound method> = bound_method %int_1, %impl.elem0.loc7_15.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dc5]
// CHECK:STDOUT:     %specific_fn.loc7_15.1: <specific function> = specific_function %impl.elem0.loc7_15.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc7_15.2: <bound method> = bound_method %int_1, %specific_fn.loc7_15.1 [concrete = constants.%bound_method.c37]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_15.1: init %i32 = call %bound_method.loc7_15.2(%int_1) [concrete = constants.%int_1.47b]
// CHECK:STDOUT:     %.loc7_15.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_15.1 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:     %.loc7_15.3: %i32 = converted %int_1, %.loc7_15.2 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:     %impl.elem0.loc7_15.2: %.940 = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.4cb]
// CHECK:STDOUT:     %bound_method.loc7_15.3: <bound method> = bound_method %int_2, %impl.elem0.loc7_15.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.30f]
// CHECK:STDOUT:     %specific_fn.loc7_15.2: <specific function> = specific_function %impl.elem0.loc7_15.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc7_15.4: <bound method> = bound_method %int_2, %specific_fn.loc7_15.2 [concrete = constants.%bound_method.62b]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_15.2: init %i32 = call %bound_method.loc7_15.4(%int_2) [concrete = constants.%int_2.d0d]
// CHECK:STDOUT:     %.loc7_15.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_15.2 [concrete = constants.%int_2.d0d]
// CHECK:STDOUT:     %.loc7_15.5: %i32 = converted %int_2, %.loc7_15.4 [concrete = constants.%int_2.d0d]
// CHECK:STDOUT:     %tuple: %tuple.type.c2c = tuple_value (%.loc7_15.3, %.loc7_15.5) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc7_16.2: %tuple.type.c2c = converted %.loc7_15.1, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%tuple) [concrete = constants.%C.cf0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.cf0 = bind_name c, %c.var [concrete = %c.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %tuple.type.dd4 = name_ref a_ref, imports.%Implicit.a_ref [concrete = imports.%a_ref.var]
// CHECK:STDOUT:   %tuple.elem0.loc5: ref %i32 = tuple_access %a_ref.ref, element0 [concrete = constants.%tuple.elem0.994]
// CHECK:STDOUT:   %.loc5_17.1: %i32 = bind_value %tuple.elem0.loc5
// CHECK:STDOUT:   %.loc5_17.2: init %tuple.type.dd4 = tuple_init (%.loc5_17.1) to file.%a.var
// CHECK:STDOUT:   %.loc5_1: init %tuple.type.dd4 = converted %a_ref.ref, %.loc5_17.2
// CHECK:STDOUT:   assign file.%a.var, %.loc5_1
// CHECK:STDOUT:   %b_ref.ref: ref %tuple.type.cfa = name_ref b_ref, imports.%Implicit.b_ref [concrete = imports.%b_ref.var]
// CHECK:STDOUT:   %tuple.elem0.loc6_38.1: ref %tuple.type.154 = tuple_access %b_ref.ref, element0 [concrete = constants.%tuple.elem0.693]
// CHECK:STDOUT:   %tuple.elem0.loc6_38.2: ref %tuple.type.dd4 = tuple_access %tuple.elem0.loc6_38.1, element0 [concrete = constants.%tuple.elem0.4c9]
// CHECK:STDOUT:   %tuple.elem0.loc6_38.3: ref %i32 = tuple_access %tuple.elem0.loc6_38.2, element0 [concrete = constants.%tuple.elem0.390]
// CHECK:STDOUT:   %.loc6_38.1: %i32 = bind_value %tuple.elem0.loc6_38.3
// CHECK:STDOUT:   %tuple.elem0.loc6_38.4: ref %tuple.type.154 = tuple_access file.%b.var, element0 [concrete = constants.%tuple.elem0.76a]
// CHECK:STDOUT:   %tuple.elem0.loc6_38.5: ref %tuple.type.dd4 = tuple_access %tuple.elem0.loc6_38.4, element0 [concrete = constants.%tuple.elem0.6fd]
// CHECK:STDOUT:   %.loc6_38.2: init %tuple.type.dd4 = tuple_init (%.loc6_38.1) to %tuple.elem0.loc6_38.5
// CHECK:STDOUT:   %.loc6_38.3: init %tuple.type.dd4 = converted %tuple.elem0.loc6_38.2, %.loc6_38.2
// CHECK:STDOUT:   %.loc6_38.4: init %tuple.type.dd4 = initialize_from %.loc6_38.3 to %tuple.elem0.loc6_38.5
// CHECK:STDOUT:   %tuple.elem1.loc6_38.1: ref %i32 = tuple_access %tuple.elem0.loc6_38.1, element1 [concrete = constants.%tuple.elem1.723]
// CHECK:STDOUT:   %.loc6_38.5: %i32 = bind_value %tuple.elem1.loc6_38.1
// CHECK:STDOUT:   %tuple.elem1.loc6_38.2: ref %i32 = tuple_access %tuple.elem0.loc6_38.4, element1 [concrete = constants.%tuple.elem1.ae3]
// CHECK:STDOUT:   %.loc6_38.6: init %i32 = initialize_from %.loc6_38.5 to %tuple.elem1.loc6_38.2
// CHECK:STDOUT:   %.loc6_38.7: init %tuple.type.154 = tuple_init (%.loc6_38.4, %.loc6_38.6) to %tuple.elem0.loc6_38.4
// CHECK:STDOUT:   %.loc6_38.8: init %tuple.type.154 = converted %tuple.elem0.loc6_38.1, %.loc6_38.7
// CHECK:STDOUT:   %tuple.elem1.loc6_38.3: ref %tuple.type.c2c = tuple_access %b_ref.ref, element1 [concrete = constants.%tuple.elem1.92b]
// CHECK:STDOUT:   %tuple.elem0.loc6_38.6: ref %i32 = tuple_access %tuple.elem1.loc6_38.3, element0 [concrete = constants.%tuple.elem0.2ff]
// CHECK:STDOUT:   %.loc6_38.9: %i32 = bind_value %tuple.elem0.loc6_38.6
// CHECK:STDOUT:   %tuple.elem1.loc6_38.4: ref %tuple.type.c2c = tuple_access file.%b.var, element1 [concrete = constants.%tuple.elem1.4f3]
// CHECK:STDOUT:   %tuple.elem0.loc6_38.7: ref %i32 = tuple_access %tuple.elem1.loc6_38.4, element0 [concrete = constants.%tuple.elem0.790]
// CHECK:STDOUT:   %.loc6_38.10: init %i32 = initialize_from %.loc6_38.9 to %tuple.elem0.loc6_38.7
// CHECK:STDOUT:   %tuple.elem1.loc6_38.5: ref %i32 = tuple_access %tuple.elem1.loc6_38.3, element1 [concrete = constants.%tuple.elem1.282]
// CHECK:STDOUT:   %.loc6_38.11: %i32 = bind_value %tuple.elem1.loc6_38.5
// CHECK:STDOUT:   %tuple.elem1.loc6_38.6: ref %i32 = tuple_access %tuple.elem1.loc6_38.4, element1 [concrete = constants.%tuple.elem1.822]
// CHECK:STDOUT:   %.loc6_38.12: init %i32 = initialize_from %.loc6_38.11 to %tuple.elem1.loc6_38.6
// CHECK:STDOUT:   %.loc6_38.13: init %tuple.type.c2c = tuple_init (%.loc6_38.10, %.loc6_38.12) to %tuple.elem1.loc6_38.4
// CHECK:STDOUT:   %.loc6_38.14: init %tuple.type.c2c = converted %tuple.elem1.loc6_38.3, %.loc6_38.13
// CHECK:STDOUT:   %.loc6_38.15: init %tuple.type.cfa = tuple_init (%.loc6_38.8, %.loc6_38.14) to file.%b.var
// CHECK:STDOUT:   %.loc6_1: init %tuple.type.cfa = converted %b_ref.ref, %.loc6_38.15
// CHECK:STDOUT:   assign file.%b.var, %.loc6_1
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Implicit.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc7: ref %C.cf0 = splice_block file.%c.var [concrete = file.%c.var] {}
// CHECK:STDOUT:   %F.call: init %C.cf0 = call %F.ref() to %.loc7
// CHECK:STDOUT:   assign file.%c.var, %F.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
