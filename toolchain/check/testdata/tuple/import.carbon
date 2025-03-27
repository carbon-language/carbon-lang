// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

var a: (i32,) = a_ref;
var b: (((i32,), i32), (i32, i32)) = b_ref;
var c: C((1, 2)) = F();

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

// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+7]]:1: error: cannot implicitly convert value of type `C((1, 2))` to `C((3, 4))` [ConversionFailure]
// CHECK:STDERR: var c_bad: C((3, 4)) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+4]]:1: note: type `C((1, 2))` does not implement interface `Core.ImplicitAs(C((3, 4)))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var c_bad: C((3, 4)) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var c_bad: C((3, 4)) = F();

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.d04: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.592: %tuple.type.a1c = tuple_value (%int_0.6a9) [concrete]
// CHECK:STDOUT:   %tuple.type.8c7: type = tuple_type (%tuple.type.85c, type) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.58c: type = tuple_type (%tuple.type.8c7, %tuple.type.24b) [concrete]
// CHECK:STDOUT:   %tuple.type.fe6: type = tuple_type (%tuple.type.a1c, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.fc1: type = tuple_type (%tuple.type.fe6, %tuple.type.d07) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.type.186: type = tuple_type (%tuple.type.985, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.type.d47: type = tuple_type (%tuple.type.186, %tuple.type.f94) [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.236: %tuple.type.fe6 = tuple_value (%tuple.592, %int_1.5d2) [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.11a: %tuple.type.d07 = tuple_value (%int_2.ef8, %int_3.822) [concrete]
// CHECK:STDOUT:   %tuple.ba9: %tuple.type.fc1 = tuple_value (%tuple.236, %tuple.11a) [concrete]
// CHECK:STDOUT:   %X: %tuple.type.d07 = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %X.patt: %tuple.type.d07 = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.54f: type = class_type @C, @C(%X) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %tuple.21c: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %C.ee0: type = class_type @C, @C(%tuple.21c) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:     .b_ref = %b_ref
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_ref.patt: %tuple.type.a1c = binding_pattern a_ref
// CHECK:STDOUT:     %.loc4_1: %tuple.type.a1c = var_pattern %a_ref.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ref.var: ref %tuple.type.a1c = var a_ref
// CHECK:STDOUT:   %.loc4_17.1: type = splice_block %.loc4_17.3 [concrete = constants.%tuple.type.a1c] {
// CHECK:STDOUT:     %int_32.loc4: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.2: %tuple.type.85c = tuple_literal (%i32.loc4)
// CHECK:STDOUT:     %.loc4_17.3: type = converted %.loc4_17.2, constants.%tuple.type.a1c [concrete = constants.%tuple.type.a1c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ref: ref %tuple.type.a1c = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_ref.patt: %tuple.type.fc1 = binding_pattern b_ref
// CHECK:STDOUT:     %.loc5_1: %tuple.type.fc1 = var_pattern %b_ref.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_ref.var: ref %tuple.type.fc1 = var b_ref
// CHECK:STDOUT:   %.loc5_38.1: type = splice_block %.loc5_38.6 [concrete = constants.%tuple.type.fc1] {
// CHECK:STDOUT:     %int_32.loc5_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_19: %tuple.type.85c = tuple_literal (%i32.loc5_15)
// CHECK:STDOUT:     %int_32.loc5_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_25: %tuple.type.8c7 = tuple_literal (%.loc5_19, %i32.loc5_22)
// CHECK:STDOUT:     %int_32.loc5_29: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_29: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_34: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_34: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_37: %tuple.type.24b = tuple_literal (%i32.loc5_29, %i32.loc5_34)
// CHECK:STDOUT:     %.loc5_38.2: %tuple.type.58c = tuple_literal (%.loc5_25, %.loc5_37)
// CHECK:STDOUT:     %.loc5_38.3: type = converted %.loc5_19, constants.%tuple.type.a1c [concrete = constants.%tuple.type.a1c]
// CHECK:STDOUT:     %.loc5_38.4: type = converted %.loc5_25, constants.%tuple.type.fe6 [concrete = constants.%tuple.type.fe6]
// CHECK:STDOUT:     %.loc5_38.5: type = converted %.loc5_37, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:     %.loc5_38.6: type = converted %.loc5_38.2, constants.%tuple.type.fc1 [concrete = constants.%tuple.type.fc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_ref: ref %tuple.type.fc1 = bind_name b_ref, %b_ref.var
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %X.patt.loc7_9.1: %tuple.type.d07 = symbolic_binding_pattern X, 0 [symbolic = %X.patt.loc7_9.2 (constants.%X.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_22.1: type = splice_block %.loc7_22.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:       %int_32.loc7_14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc7_14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_32.loc7_19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc7_19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc7_22.2: %tuple.type.24b = tuple_literal (%i32.loc7_14, %i32.loc7_19)
// CHECK:STDOUT:       %.loc7_22.3: type = converted %.loc7_22.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %X.loc7_9.1: %tuple.type.d07 = bind_symbolic_name X, 0 [symbolic = %X.loc7_9.2 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %C.ee0 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C.ee0 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc9_18.1: %tuple.type.f94 = tuple_literal (%int_1, %int_2)
// CHECK:STDOUT:     %impl.elem0.loc9_18.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc9_18.1: <bound method> = bound_method %int_1, %impl.elem0.loc9_18.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:     %specific_fn.loc9_18.1: <specific function> = specific_function %impl.elem0.loc9_18.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_18.2: <bound method> = bound_method %int_1, %specific_fn.loc9_18.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:     %int.convert_checked.loc9_18.1: init %i32 = call %bound_method.loc9_18.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc9_18.2: %i32 = value_of_initializer %int.convert_checked.loc9_18.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc9_18.3: %i32 = converted %int_1, %.loc9_18.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %impl.elem0.loc9_18.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc9_18.3: <bound method> = bound_method %int_2, %impl.elem0.loc9_18.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:     %specific_fn.loc9_18.2: <specific function> = specific_function %impl.elem0.loc9_18.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_18.4: <bound method> = bound_method %int_2, %specific_fn.loc9_18.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:     %int.convert_checked.loc9_18.2: init %i32 = call %bound_method.loc9_18.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc9_18.4: %i32 = value_of_initializer %int.convert_checked.loc9_18.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc9_18.5: %i32 = converted %int_2, %.loc9_18.4 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %tuple: %tuple.type.d07 = tuple_value (%.loc9_18.3, %.loc9_18.5) [concrete = constants.%tuple.21c]
// CHECK:STDOUT:     %.loc9_19: %tuple.type.d07 = converted %.loc9_18.1, %tuple [concrete = constants.%tuple.21c]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%tuple.21c) [concrete = constants.%C.ee0]
// CHECK:STDOUT:     %return.param: ref %C.ee0 = out_param call_param0
// CHECK:STDOUT:     %return: ref %C.ee0 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%X.loc7_9.1: %tuple.type.d07) {
// CHECK:STDOUT:   %X.loc7_9.2: %tuple.type.d07 = bind_symbolic_name X, 0 [symbolic = %X.loc7_9.2 (constants.%X)]
// CHECK:STDOUT:   %X.patt.loc7_9.2: %tuple.type.d07 = symbolic_binding_pattern X, 0 [symbolic = %X.patt.loc7_9.2 (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.54f
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.ee0;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc4: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc4_24.1: %tuple.type.985 = tuple_literal (%int_0.loc4)
// CHECK:STDOUT:   %impl.elem0.loc4: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc4_24.1: <bound method> = bound_method %int_0.loc4, %impl.elem0.loc4 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc4: <specific function> = specific_function %impl.elem0.loc4, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_24.2: <bound method> = bound_method %int_0.loc4, %specific_fn.loc4 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc4: init %i32 = call %bound_method.loc4_24.2(%int_0.loc4) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc4_24.2: init %i32 = converted %int_0.loc4, %int.convert_checked.loc4 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc4_24.3: init %tuple.type.a1c = tuple_init (%.loc4_24.2) to file.%a_ref.var [concrete = constants.%tuple.592]
// CHECK:STDOUT:   %.loc4_1: init %tuple.type.a1c = converted %.loc4_24.1, %.loc4_24.3 [concrete = constants.%tuple.592]
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc4_1
// CHECK:STDOUT:   %int_0.loc5: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc5_47.1: %tuple.type.985 = tuple_literal (%int_0.loc5)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc5_51.1: %tuple.type.186 = tuple_literal (%.loc5_47.1, %int_1)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc5_59.1: %tuple.type.f94 = tuple_literal (%int_2, %int_3)
// CHECK:STDOUT:   %.loc5_60.1: %tuple.type.d47 = tuple_literal (%.loc5_51.1, %.loc5_59.1)
// CHECK:STDOUT:   %impl.elem0.loc5_47: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc5_47.1: <bound method> = bound_method %int_0.loc5, %impl.elem0.loc5_47 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc5_47: <specific function> = specific_function %impl.elem0.loc5_47, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_47.2: <bound method> = bound_method %int_0.loc5, %specific_fn.loc5_47 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc5_47: init %i32 = call %bound_method.loc5_47.2(%int_0.loc5) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc5_47.2: init %i32 = converted %int_0.loc5, %int.convert_checked.loc5_47 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %tuple.elem0.loc5_60: ref %tuple.type.fe6 = tuple_access file.%b_ref.var, element0
// CHECK:STDOUT:   %tuple.elem0.loc5_51: ref %tuple.type.a1c = tuple_access %tuple.elem0.loc5_60, element0
// CHECK:STDOUT:   %.loc5_47.3: init %tuple.type.a1c = tuple_init (%.loc5_47.2) to %tuple.elem0.loc5_51 [concrete = constants.%tuple.592]
// CHECK:STDOUT:   %.loc5_51.2: init %tuple.type.a1c = converted %.loc5_47.1, %.loc5_47.3 [concrete = constants.%tuple.592]
// CHECK:STDOUT:   %.loc5_51.3: init %tuple.type.a1c = initialize_from %.loc5_51.2 to %tuple.elem0.loc5_51 [concrete = constants.%tuple.592]
// CHECK:STDOUT:   %impl.elem0.loc5_51: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc5_51.1: <bound method> = bound_method %int_1, %impl.elem0.loc5_51 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc5_51: <specific function> = specific_function %impl.elem0.loc5_51, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_51.2: <bound method> = bound_method %int_1, %specific_fn.loc5_51 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc5_51: init %i32 = call %bound_method.loc5_51.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc5_51.4: init %i32 = converted %int_1, %int.convert_checked.loc5_51 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem1.loc5_51: ref %i32 = tuple_access %tuple.elem0.loc5_60, element1
// CHECK:STDOUT:   %.loc5_51.5: init %i32 = initialize_from %.loc5_51.4 to %tuple.elem1.loc5_51 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc5_51.6: init %tuple.type.fe6 = tuple_init (%.loc5_51.3, %.loc5_51.5) to %tuple.elem0.loc5_60 [concrete = constants.%tuple.236]
// CHECK:STDOUT:   %.loc5_60.2: init %tuple.type.fe6 = converted %.loc5_51.1, %.loc5_51.6 [concrete = constants.%tuple.236]
// CHECK:STDOUT:   %impl.elem0.loc5_59.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc5_59.1: <bound method> = bound_method %int_2, %impl.elem0.loc5_59.1 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc5_59.1: <specific function> = specific_function %impl.elem0.loc5_59.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_59.2: <bound method> = bound_method %int_2, %specific_fn.loc5_59.1 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc5_59.1: init %i32 = call %bound_method.loc5_59.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc5_59.2: init %i32 = converted %int_2, %int.convert_checked.loc5_59.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.elem1.loc5_60: ref %tuple.type.d07 = tuple_access file.%b_ref.var, element1
// CHECK:STDOUT:   %tuple.elem0.loc5_59: ref %i32 = tuple_access %tuple.elem1.loc5_60, element0
// CHECK:STDOUT:   %.loc5_59.3: init %i32 = initialize_from %.loc5_59.2 to %tuple.elem0.loc5_59 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc5_59.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc5_59.3: <bound method> = bound_method %int_3, %impl.elem0.loc5_59.2 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc5_59.2: <specific function> = specific_function %impl.elem0.loc5_59.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_59.4: <bound method> = bound_method %int_3, %specific_fn.loc5_59.2 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc5_59.2: init %i32 = call %bound_method.loc5_59.4(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc5_59.4: init %i32 = converted %int_3, %int.convert_checked.loc5_59.2 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %tuple.elem1.loc5_59: ref %i32 = tuple_access %tuple.elem1.loc5_60, element1
// CHECK:STDOUT:   %.loc5_59.5: init %i32 = initialize_from %.loc5_59.4 to %tuple.elem1.loc5_59 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc5_59.6: init %tuple.type.d07 = tuple_init (%.loc5_59.3, %.loc5_59.5) to %tuple.elem1.loc5_60 [concrete = constants.%tuple.11a]
// CHECK:STDOUT:   %.loc5_60.3: init %tuple.type.d07 = converted %.loc5_59.1, %.loc5_59.6 [concrete = constants.%tuple.11a]
// CHECK:STDOUT:   %.loc5_60.4: init %tuple.type.fc1 = tuple_init (%.loc5_60.2, %.loc5_60.3) to file.%b_ref.var [concrete = constants.%tuple.ba9]
// CHECK:STDOUT:   %.loc5_1: init %tuple.type.fc1 = converted %.loc5_60.1, %.loc5_60.4 [concrete = constants.%tuple.ba9]
// CHECK:STDOUT:   assign file.%b_ref.var, %.loc5_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X.loc7_9.2 => constants.%X
// CHECK:STDOUT:   %X.patt.loc7_9.2 => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.21c) {
// CHECK:STDOUT:   %X.loc7_9.2 => constants.%tuple.21c
// CHECK:STDOUT:   %X.patt.loc7_9.2 => constants.%tuple.21c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.type.dd4: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %tuple.type.8c7: type = tuple_type (%tuple.type.85c, type) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.58c: type = tuple_type (%tuple.type.8c7, %tuple.type.24b) [concrete]
// CHECK:STDOUT:   %tuple.type.154: type = tuple_type (%tuple.type.dd4, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.c2c: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.cfa: type = tuple_type (%tuple.type.154, %tuple.type.c2c) [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %X: %tuple.type.c2c = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %X.patt: %tuple.type.c2c = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b9e: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.ea0: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.847: <witness> = impl_witness (imports.%Implicit.import_ref.773), @impl.a8d(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.e14: type = fn_type @Convert.2, @impl.a8d(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.4cb: %Convert.type.e14 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.b9e = facet_value Core.IntLiteral, (%impl_witness.847) [concrete]
// CHECK:STDOUT:   %.419: type = fn_type_with_self_type %Convert.type.ea0, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.dc5: <bound method> = bound_method %int_1.5b8, %Convert.4cb [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.4cb, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.c37: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.47b: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.30f: <bound method> = bound_method %int_2.ecc, %Convert.4cb [concrete]
// CHECK:STDOUT:   %bound_method.62b: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.d0d: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.c2c = tuple_value (%int_1.47b, %int_2.d0d) [concrete]
// CHECK:STDOUT:   %C.cf0: type = class_type @C, @C(%tuple) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Implicit.a_ref: ref %tuple.type.dd4 = import_ref Implicit//default, a_ref, loaded
// CHECK:STDOUT:   %Implicit.b_ref: ref %tuple.type.cfa = import_ref Implicit//default, b_ref, loaded
// CHECK:STDOUT:   %Implicit.C: %C.type = import_ref Implicit//default, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Implicit.F: %F.type = import_ref Implicit//default, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import_ref.48d: %tuple.type.c2c = import_ref Implicit//default, loc7_9, loaded [symbolic = @C.%X (constants.%X)]
// CHECK:STDOUT:   %Implicit.import_ref.8f2: <witness> = import_ref Implicit//default, loc7_26, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Implicit.import_ref.964 = import_ref Implicit//default, inst1174 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .a_ref = imports.%Implicit.a_ref
// CHECK:STDOUT:     .b_ref = imports.%Implicit.b_ref
// CHECK:STDOUT:     .C = imports.%Implicit.C
// CHECK:STDOUT:     .F = imports.%Implicit.F
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %tuple.type.dd4 = binding_pattern a
// CHECK:STDOUT:     %.loc4_1: %tuple.type.dd4 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %tuple.type.dd4 = var a
// CHECK:STDOUT:   %.loc4_13.1: type = splice_block %.loc4_13.3 [concrete = constants.%tuple.type.dd4] {
// CHECK:STDOUT:     %int_32.loc4: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_13.2: %tuple.type.85c = tuple_literal (%i32.loc4)
// CHECK:STDOUT:     %.loc4_13.3: type = converted %.loc4_13.2, constants.%tuple.type.dd4 [concrete = constants.%tuple.type.dd4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %tuple.type.dd4 = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %tuple.type.cfa = binding_pattern b
// CHECK:STDOUT:     %.loc5_1: %tuple.type.cfa = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %tuple.type.cfa = var b
// CHECK:STDOUT:   %.loc5_34.1: type = splice_block %.loc5_34.6 [concrete = constants.%tuple.type.cfa] {
// CHECK:STDOUT:     %int_32.loc5_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_15: %tuple.type.85c = tuple_literal (%i32.loc5_11)
// CHECK:STDOUT:     %int_32.loc5_18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_21: %tuple.type.8c7 = tuple_literal (%.loc5_15, %i32.loc5_18)
// CHECK:STDOUT:     %int_32.loc5_25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_30: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_30: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_33: %tuple.type.24b = tuple_literal (%i32.loc5_25, %i32.loc5_30)
// CHECK:STDOUT:     %.loc5_34.2: %tuple.type.58c = tuple_literal (%.loc5_21, %.loc5_33)
// CHECK:STDOUT:     %.loc5_34.3: type = converted %.loc5_15, constants.%tuple.type.dd4 [concrete = constants.%tuple.type.dd4]
// CHECK:STDOUT:     %.loc5_34.4: type = converted %.loc5_21, constants.%tuple.type.154 [concrete = constants.%tuple.type.154]
// CHECK:STDOUT:     %.loc5_34.5: type = converted %.loc5_33, constants.%tuple.type.c2c [concrete = constants.%tuple.type.c2c]
// CHECK:STDOUT:     %.loc5_34.6: type = converted %.loc5_34.2, constants.%tuple.type.cfa [concrete = constants.%tuple.type.cfa]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %tuple.type.cfa = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C.cf0 = binding_pattern c
// CHECK:STDOUT:     %.loc6_1: %C.cf0 = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.cf0 = var c
// CHECK:STDOUT:   %.loc6_16.1: type = splice_block %C [concrete = constants.%C.cf0] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Implicit.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc6_15.1: %tuple.type.f94 = tuple_literal (%int_1, %int_2)
// CHECK:STDOUT:     %impl.elem0.loc6_15.1: %.419 = impl_witness_access constants.%impl_witness.847, element0 [concrete = constants.%Convert.4cb]
// CHECK:STDOUT:     %bound_method.loc6_15.1: <bound method> = bound_method %int_1, %impl.elem0.loc6_15.1 [concrete = constants.%Convert.bound.dc5]
// CHECK:STDOUT:     %specific_fn.loc6_15.1: <specific function> = specific_function %impl.elem0.loc6_15.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_15.2: <bound method> = bound_method %int_1, %specific_fn.loc6_15.1 [concrete = constants.%bound_method.c37]
// CHECK:STDOUT:     %int.convert_checked.loc6_15.1: init %i32 = call %bound_method.loc6_15.2(%int_1) [concrete = constants.%int_1.47b]
// CHECK:STDOUT:     %.loc6_15.2: %i32 = value_of_initializer %int.convert_checked.loc6_15.1 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:     %.loc6_15.3: %i32 = converted %int_1, %.loc6_15.2 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:     %impl.elem0.loc6_15.2: %.419 = impl_witness_access constants.%impl_witness.847, element0 [concrete = constants.%Convert.4cb]
// CHECK:STDOUT:     %bound_method.loc6_15.3: <bound method> = bound_method %int_2, %impl.elem0.loc6_15.2 [concrete = constants.%Convert.bound.30f]
// CHECK:STDOUT:     %specific_fn.loc6_15.2: <specific function> = specific_function %impl.elem0.loc6_15.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_15.4: <bound method> = bound_method %int_2, %specific_fn.loc6_15.2 [concrete = constants.%bound_method.62b]
// CHECK:STDOUT:     %int.convert_checked.loc6_15.2: init %i32 = call %bound_method.loc6_15.4(%int_2) [concrete = constants.%int_2.d0d]
// CHECK:STDOUT:     %.loc6_15.4: %i32 = value_of_initializer %int.convert_checked.loc6_15.2 [concrete = constants.%int_2.d0d]
// CHECK:STDOUT:     %.loc6_15.5: %i32 = converted %int_2, %.loc6_15.4 [concrete = constants.%int_2.d0d]
// CHECK:STDOUT:     %tuple: %tuple.type.c2c = tuple_value (%.loc6_15.3, %.loc6_15.5) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc6_16.2: %tuple.type.c2c = converted %.loc6_15.1, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%tuple) [concrete = constants.%C.cf0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.cf0 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Implicit.import_ref.48d: %tuple.type.c2c) [from "implicit.carbon"] {
// CHECK:STDOUT:   %X: %tuple.type.c2c = bind_symbolic_name X, 0 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %X.patt: %tuple.type.c2c = symbolic_binding_pattern X, 0 [symbolic = %X.patt (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Implicit.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Implicit.import_ref.964
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.cf0 [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %tuple.type.dd4 = name_ref a_ref, imports.%Implicit.a_ref
// CHECK:STDOUT:   %tuple.elem0.loc4: ref %i32 = tuple_access %a_ref.ref, element0
// CHECK:STDOUT:   %.loc4_17.1: %i32 = bind_value %tuple.elem0.loc4
// CHECK:STDOUT:   %.loc4_17.2: init %tuple.type.dd4 = tuple_init (%.loc4_17.1) to file.%a.var
// CHECK:STDOUT:   %.loc4_1: init %tuple.type.dd4 = converted %a_ref.ref, %.loc4_17.2
// CHECK:STDOUT:   assign file.%a.var, %.loc4_1
// CHECK:STDOUT:   %b_ref.ref: ref %tuple.type.cfa = name_ref b_ref, imports.%Implicit.b_ref
// CHECK:STDOUT:   %tuple.elem0.loc5_38.1: ref %tuple.type.154 = tuple_access %b_ref.ref, element0
// CHECK:STDOUT:   %tuple.elem0.loc5_38.2: ref %tuple.type.dd4 = tuple_access %tuple.elem0.loc5_38.1, element0
// CHECK:STDOUT:   %tuple.elem0.loc5_38.3: ref %i32 = tuple_access %tuple.elem0.loc5_38.2, element0
// CHECK:STDOUT:   %.loc5_38.1: %i32 = bind_value %tuple.elem0.loc5_38.3
// CHECK:STDOUT:   %tuple.elem0.loc5_38.4: ref %tuple.type.154 = tuple_access file.%b.var, element0
// CHECK:STDOUT:   %tuple.elem0.loc5_38.5: ref %tuple.type.dd4 = tuple_access %tuple.elem0.loc5_38.4, element0
// CHECK:STDOUT:   %.loc5_38.2: init %tuple.type.dd4 = tuple_init (%.loc5_38.1) to %tuple.elem0.loc5_38.5
// CHECK:STDOUT:   %.loc5_38.3: init %tuple.type.dd4 = converted %tuple.elem0.loc5_38.2, %.loc5_38.2
// CHECK:STDOUT:   %.loc5_38.4: init %tuple.type.dd4 = initialize_from %.loc5_38.3 to %tuple.elem0.loc5_38.5
// CHECK:STDOUT:   %tuple.elem1.loc5_38.1: ref %i32 = tuple_access %tuple.elem0.loc5_38.1, element1
// CHECK:STDOUT:   %.loc5_38.5: %i32 = bind_value %tuple.elem1.loc5_38.1
// CHECK:STDOUT:   %tuple.elem1.loc5_38.2: ref %i32 = tuple_access %tuple.elem0.loc5_38.4, element1
// CHECK:STDOUT:   %.loc5_38.6: init %i32 = initialize_from %.loc5_38.5 to %tuple.elem1.loc5_38.2
// CHECK:STDOUT:   %.loc5_38.7: init %tuple.type.154 = tuple_init (%.loc5_38.4, %.loc5_38.6) to %tuple.elem0.loc5_38.4
// CHECK:STDOUT:   %.loc5_38.8: init %tuple.type.154 = converted %tuple.elem0.loc5_38.1, %.loc5_38.7
// CHECK:STDOUT:   %tuple.elem1.loc5_38.3: ref %tuple.type.c2c = tuple_access %b_ref.ref, element1
// CHECK:STDOUT:   %tuple.elem0.loc5_38.6: ref %i32 = tuple_access %tuple.elem1.loc5_38.3, element0
// CHECK:STDOUT:   %.loc5_38.9: %i32 = bind_value %tuple.elem0.loc5_38.6
// CHECK:STDOUT:   %tuple.elem1.loc5_38.4: ref %tuple.type.c2c = tuple_access file.%b.var, element1
// CHECK:STDOUT:   %tuple.elem0.loc5_38.7: ref %i32 = tuple_access %tuple.elem1.loc5_38.4, element0
// CHECK:STDOUT:   %.loc5_38.10: init %i32 = initialize_from %.loc5_38.9 to %tuple.elem0.loc5_38.7
// CHECK:STDOUT:   %tuple.elem1.loc5_38.5: ref %i32 = tuple_access %tuple.elem1.loc5_38.3, element1
// CHECK:STDOUT:   %.loc5_38.11: %i32 = bind_value %tuple.elem1.loc5_38.5
// CHECK:STDOUT:   %tuple.elem1.loc5_38.6: ref %i32 = tuple_access %tuple.elem1.loc5_38.4, element1
// CHECK:STDOUT:   %.loc5_38.12: init %i32 = initialize_from %.loc5_38.11 to %tuple.elem1.loc5_38.6
// CHECK:STDOUT:   %.loc5_38.13: init %tuple.type.c2c = tuple_init (%.loc5_38.10, %.loc5_38.12) to %tuple.elem1.loc5_38.4
// CHECK:STDOUT:   %.loc5_38.14: init %tuple.type.c2c = converted %tuple.elem1.loc5_38.3, %.loc5_38.13
// CHECK:STDOUT:   %.loc5_38.15: init %tuple.type.cfa = tuple_init (%.loc5_38.8, %.loc5_38.14) to file.%b.var
// CHECK:STDOUT:   %.loc5_1: init %tuple.type.cfa = converted %b_ref.ref, %.loc5_38.15
// CHECK:STDOUT:   assign file.%b.var, %.loc5_1
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Implicit.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc6: ref %C.cf0 = splice_block file.%c.var {}
// CHECK:STDOUT:   %F.call: init %C.cf0 = call %F.ref() to %.loc6
// CHECK:STDOUT:   assign file.%c.var, %F.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %X.patt => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple) {
// CHECK:STDOUT:   %X => constants.%tuple
// CHECK:STDOUT:   %X.patt => constants.%tuple
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_type.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %tuple.type.c2c: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %X: %tuple.type.c2c = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %X.patt: %tuple.type.c2c = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.d0d: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_1.47b: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.c2c = tuple_value (%int_1.47b, %int_2.d0d) [concrete]
// CHECK:STDOUT:   %C.cf0: type = class_type @C, @C(%tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Implicit.a_ref = import_ref Implicit//default, a_ref, unloaded
// CHECK:STDOUT:   %Implicit.b_ref = import_ref Implicit//default, b_ref, unloaded
// CHECK:STDOUT:   %Implicit.C: %C.type = import_ref Implicit//default, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Implicit.F: %F.type = import_ref Implicit//default, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import_ref.48d: %tuple.type.c2c = import_ref Implicit//default, loc7_9, loaded [symbolic = @C.%X (constants.%X)]
// CHECK:STDOUT:   %Implicit.import_ref.8f2: <witness> = import_ref Implicit//default, loc7_26, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Implicit.import_ref.964 = import_ref Implicit//default, inst1174 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .a_ref = imports.%Implicit.a_ref
// CHECK:STDOUT:     .b_ref = imports.%Implicit.b_ref
// CHECK:STDOUT:     .C = imports.%Implicit.C
// CHECK:STDOUT:     .F = imports.%Implicit.F
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .c_bad = %c_bad
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c_bad.patt: <error> = binding_pattern c_bad
// CHECK:STDOUT:     %.loc12_1: <error> = var_pattern %c_bad.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_bad.var: ref <error> = var c_bad
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Implicit.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %.loc12_22: %tuple.type.37f = tuple_literal (%int_1, %int_2, %int_3)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_bad: <error> = bind_name c_bad, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Implicit.import_ref.48d: %tuple.type.c2c) [from "implicit.carbon"] {
// CHECK:STDOUT:   %X: %tuple.type.c2c = bind_symbolic_name X, 0 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %X.patt: %tuple.type.c2c = symbolic_binding_pattern X, 0 [symbolic = %X.patt (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Implicit.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Implicit.import_ref.964
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.cf0 [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Implicit.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc12: ref %C.cf0 = temporary_storage
// CHECK:STDOUT:   %F.call: init %C.cf0 = call %F.ref() to %.loc12
// CHECK:STDOUT:   assign file.%c_bad.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %X.patt => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple) {
// CHECK:STDOUT:   %X => constants.%tuple
// CHECK:STDOUT:   %X.patt => constants.%tuple
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_value.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %tuple.type.c2c: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %X: %tuple.type.c2c = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %X.patt: %tuple.type.c2c = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b9e: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.ea0: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.847: <witness> = impl_witness (imports.%Implicit.import_ref.773), @impl.a8d(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.e14: type = fn_type @Convert.2, @impl.a8d(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.4cb: %Convert.type.e14 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.b9e = facet_value Core.IntLiteral, (%impl_witness.847) [concrete]
// CHECK:STDOUT:   %.419: type = fn_type_with_self_type %Convert.type.ea0, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.a00: <bound method> = bound_method %int_3.1ba, %Convert.4cb [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.4cb, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.fc5: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.d47: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %Convert.bound.694: <bound method> = bound_method %int_4.0c1, %Convert.4cb [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_4.0c1, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.ea3: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %tuple.5de: %tuple.type.c2c = tuple_value (%int_3.d47, %int_4.ea3) [concrete]
// CHECK:STDOUT:   %C.ed5: type = class_type @C, @C(%tuple.5de) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_2: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_1: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.56c: %tuple.type.c2c = tuple_value (%int_1, %int_2) [concrete]
// CHECK:STDOUT:   %C.cf0: type = class_type @C, @C(%tuple.56c) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Implicit.a_ref = import_ref Implicit//default, a_ref, unloaded
// CHECK:STDOUT:   %Implicit.b_ref = import_ref Implicit//default, b_ref, unloaded
// CHECK:STDOUT:   %Implicit.C: %C.type = import_ref Implicit//default, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Implicit.F: %F.type = import_ref Implicit//default, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import_ref.48d: %tuple.type.c2c = import_ref Implicit//default, loc7_9, loaded [symbolic = @C.%X (constants.%X)]
// CHECK:STDOUT:   %Implicit.import_ref.8f2: <witness> = import_ref Implicit//default, loc7_26, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Implicit.import_ref.964 = import_ref Implicit//default, inst1174 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .a_ref = imports.%Implicit.a_ref
// CHECK:STDOUT:     .b_ref = imports.%Implicit.b_ref
// CHECK:STDOUT:     .C = imports.%Implicit.C
// CHECK:STDOUT:     .F = imports.%Implicit.F
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .c_bad = %c_bad
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c_bad.patt: %C.ed5 = binding_pattern c_bad
// CHECK:STDOUT:     %.loc11_1: %C.ed5 = var_pattern %c_bad.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_bad.var: ref %C.ed5 = var c_bad
// CHECK:STDOUT:   %.loc11_20.1: type = splice_block %C [concrete = constants.%C.ed5] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Implicit.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:     %.loc11_19.1: %tuple.type.f94 = tuple_literal (%int_3, %int_4)
// CHECK:STDOUT:     %impl.elem0.loc11_19.1: %.419 = impl_witness_access constants.%impl_witness.847, element0 [concrete = constants.%Convert.4cb]
// CHECK:STDOUT:     %bound_method.loc11_19.1: <bound method> = bound_method %int_3, %impl.elem0.loc11_19.1 [concrete = constants.%Convert.bound.a00]
// CHECK:STDOUT:     %specific_fn.loc11_19.1: <specific function> = specific_function %impl.elem0.loc11_19.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_19.2: <bound method> = bound_method %int_3, %specific_fn.loc11_19.1 [concrete = constants.%bound_method.fc5]
// CHECK:STDOUT:     %int.convert_checked.loc11_19.1: init %i32 = call %bound_method.loc11_19.2(%int_3) [concrete = constants.%int_3.d47]
// CHECK:STDOUT:     %.loc11_19.2: %i32 = value_of_initializer %int.convert_checked.loc11_19.1 [concrete = constants.%int_3.d47]
// CHECK:STDOUT:     %.loc11_19.3: %i32 = converted %int_3, %.loc11_19.2 [concrete = constants.%int_3.d47]
// CHECK:STDOUT:     %impl.elem0.loc11_19.2: %.419 = impl_witness_access constants.%impl_witness.847, element0 [concrete = constants.%Convert.4cb]
// CHECK:STDOUT:     %bound_method.loc11_19.3: <bound method> = bound_method %int_4, %impl.elem0.loc11_19.2 [concrete = constants.%Convert.bound.694]
// CHECK:STDOUT:     %specific_fn.loc11_19.2: <specific function> = specific_function %impl.elem0.loc11_19.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_19.4: <bound method> = bound_method %int_4, %specific_fn.loc11_19.2 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:     %int.convert_checked.loc11_19.2: init %i32 = call %bound_method.loc11_19.4(%int_4) [concrete = constants.%int_4.ea3]
// CHECK:STDOUT:     %.loc11_19.4: %i32 = value_of_initializer %int.convert_checked.loc11_19.2 [concrete = constants.%int_4.ea3]
// CHECK:STDOUT:     %.loc11_19.5: %i32 = converted %int_4, %.loc11_19.4 [concrete = constants.%int_4.ea3]
// CHECK:STDOUT:     %tuple: %tuple.type.c2c = tuple_value (%.loc11_19.3, %.loc11_19.5) [concrete = constants.%tuple.5de]
// CHECK:STDOUT:     %.loc11_20.2: %tuple.type.c2c = converted %.loc11_19.1, %tuple [concrete = constants.%tuple.5de]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%tuple.5de) [concrete = constants.%C.ed5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_bad: ref %C.ed5 = bind_name c_bad, %c_bad.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Implicit.import_ref.48d: %tuple.type.c2c) [from "implicit.carbon"] {
// CHECK:STDOUT:   %X: %tuple.type.c2c = bind_symbolic_name X, 0 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %X.patt: %tuple.type.c2c = symbolic_binding_pattern X, 0 [symbolic = %X.patt (constants.%X.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Implicit.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Implicit.import_ref.964
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.cf0 [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Implicit.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc11_26: ref %C.cf0 = temporary_storage
// CHECK:STDOUT:   %F.call: init %C.cf0 = call %F.ref() to %.loc11_26
// CHECK:STDOUT:   %.loc11_1: %C.ed5 = converted %F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%c_bad.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %X.patt => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.5de) {
// CHECK:STDOUT:   %X => constants.%tuple.5de
// CHECK:STDOUT:   %X.patt => constants.%tuple.5de
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%tuple.56c) {
// CHECK:STDOUT:   %X => constants.%tuple.56c
// CHECK:STDOUT:   %X.patt => constants.%tuple.56c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
