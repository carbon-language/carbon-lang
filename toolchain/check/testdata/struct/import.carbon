// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/import.carbon

// --- implicit.carbon

package Implicit;

var a_ref: {.a: i32} = {.a = 0};
var b_ref: {.a: {.b: i32, .c: (i32,)}, .d: i32} =
    {.a = {.b = 0, .c = (0,)}, .d = 0};

class C(S:! {.a: i32, .b: i32}) {}
fn F() -> C({.a = 1, .b = 2});

// --- implicit.impl.carbon

impl package Implicit;

var a: {.a: i32} = a_ref;
var b: {.a: {.b: i32, .c: (i32,)}, .d: i32} = b_ref;
var c: C({.a = 1, .b = 2}) = F();

// --- fail_bad_type.impl.carbon

impl package Implicit;
// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE+8]]:14: error: missing value for field `a` in struct initialization [StructInitMissingFieldInLiteral]
// CHECK:STDERR: var c_bad: C({.c = 1, .d = 2}) = F();
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE-4]]:1: in import [InImport]
// CHECK:STDERR: implicit.carbon:8:9: note: initializing generic parameter `S` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(S:! {.a: i32, .b: i32}) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
var c_bad: C({.c = 1, .d = 2}) = F();

// --- fail_bad_value.impl.carbon

impl package Implicit;
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `C({.a = 1, .b = 2})` to `C({.a = 3, .b = 4})` [ConversionFailure]
// CHECK:STDERR: var c_bad: C({.a = 3, .b = 4}) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+4]]:1: note: type `C({.a = 1, .b = 2})` does not implement interface `Core.ImplicitAs(C({.a = 3, .b = 4}))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var c_bad: C({.a = 3, .b = 4}) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var c_bad: C({.a = 3, .b = 4}) = F();

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.ba9: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %struct_type.a.a6c: type = struct_type {.a: Core.IntLiteral} [concrete]
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
// CHECK:STDOUT:   %struct.92d: %struct_type.a.ba9 = struct_value (%int_0.6a9) [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %struct_type.b.c.929: type = struct_type {.b: %i32, .c: %tuple.type.a1c} [concrete]
// CHECK:STDOUT:   %struct_type.a.d.3d9: type = struct_type {.a: %struct_type.b.c.929, .d: %i32} [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %struct_type.b.c.9af: type = struct_type {.b: Core.IntLiteral, .c: %tuple.type.985} [concrete]
// CHECK:STDOUT:   %struct_type.a.d.b82: type = struct_type {.a: %struct_type.b.c.9af, .d: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.a1c = tuple_value (%int_0.6a9) [concrete]
// CHECK:STDOUT:   %struct.381: %struct_type.b.c.929 = struct_value (%int_0.6a9, %tuple) [concrete]
// CHECK:STDOUT:   %struct.20d: %struct_type.a.d.3d9 = struct_value (%struct.381, %int_0.6a9) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.501: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %S: %struct_type.a.b.501 = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %S.patt: %struct_type.a.b.501 = symbolic_binding_pattern S, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.775: type = class_type @C, @C(%S) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %struct.ed5: %struct_type.a.b.501 = struct_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %C.c8f: type = class_type @C, @C(%struct.ed5) [concrete]
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
// CHECK:STDOUT:     %a_ref.patt: %struct_type.a.ba9 = binding_pattern a_ref
// CHECK:STDOUT:     %.loc4_1: %struct_type.a.ba9 = var_pattern %a_ref.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ref.var: ref %struct_type.a.ba9 = var a_ref
// CHECK:STDOUT:   %.loc4_20: type = splice_block %struct_type.a [concrete = constants.%struct_type.a.ba9] {
// CHECK:STDOUT:     %int_32.loc4: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a.ba9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ref: ref %struct_type.a.ba9 = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_ref.patt: %struct_type.a.d.3d9 = binding_pattern b_ref
// CHECK:STDOUT:     %.loc5_1: %struct_type.a.d.3d9 = var_pattern %b_ref.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_ref.var: ref %struct_type.a.d.3d9 = var b_ref
// CHECK:STDOUT:   %.loc5_47: type = splice_block %struct_type.a.d [concrete = constants.%struct_type.a.d.3d9] {
// CHECK:STDOUT:     %int_32.loc5_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_36.1: %tuple.type.85c = tuple_literal (%i32.loc5_32)
// CHECK:STDOUT:     %.loc5_36.2: type = converted %.loc5_36.1, constants.%tuple.type.a1c [concrete = constants.%tuple.type.a1c]
// CHECK:STDOUT:     %struct_type.b.c: type = struct_type {.b: %i32, .c: %tuple.type.a1c} [concrete = constants.%struct_type.b.c.929]
// CHECK:STDOUT:     %int_32.loc5_44: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_44: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.d: type = struct_type {.a: %struct_type.b.c.929, .d: %i32} [concrete = constants.%struct_type.a.d.3d9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_ref: ref %struct_type.a.d.3d9 = bind_name b_ref, %b_ref.var
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %S.patt.loc8_9.1: %struct_type.a.b.501 = symbolic_binding_pattern S, 0 [symbolic = %S.patt.loc8_9.2 (constants.%S.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b.501] {
// CHECK:STDOUT:       %int_32.loc8_18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc8_18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_32.loc8_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc8_27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.501]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %S.loc8_9.1: %struct_type.a.b.501 = bind_symbolic_name S, 0 [symbolic = %S.loc8_9.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %C.c8f = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C.c8f = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc9_28.1: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:     %impl.elem0.loc9_28.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc9_28.1: <bound method> = bound_method %int_1, %impl.elem0.loc9_28.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:     %specific_fn.loc9_28.1: <specific function> = specific_function %impl.elem0.loc9_28.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_28.2: <bound method> = bound_method %int_1, %specific_fn.loc9_28.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:     %int.convert_checked.loc9_28.1: init %i32 = call %bound_method.loc9_28.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc9_28.2: %i32 = value_of_initializer %int.convert_checked.loc9_28.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc9_28.3: %i32 = converted %int_1, %.loc9_28.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %impl.elem0.loc9_28.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc9_28.3: <bound method> = bound_method %int_2, %impl.elem0.loc9_28.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:     %specific_fn.loc9_28.2: <specific function> = specific_function %impl.elem0.loc9_28.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_28.4: <bound method> = bound_method %int_2, %specific_fn.loc9_28.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:     %int.convert_checked.loc9_28.2: init %i32 = call %bound_method.loc9_28.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc9_28.4: %i32 = value_of_initializer %int.convert_checked.loc9_28.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc9_28.5: %i32 = converted %int_2, %.loc9_28.4 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %struct: %struct_type.a.b.501 = struct_value (%.loc9_28.3, %.loc9_28.5) [concrete = constants.%struct.ed5]
// CHECK:STDOUT:     %.loc9_29: %struct_type.a.b.501 = converted %.loc9_28.1, %struct [concrete = constants.%struct.ed5]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%struct.ed5) [concrete = constants.%C.c8f]
// CHECK:STDOUT:     %return.param: ref %C.c8f = out_param call_param0
// CHECK:STDOUT:     %return: ref %C.c8f = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%S.loc8_9.1: %struct_type.a.b.501) {
// CHECK:STDOUT:   %S.loc8_9.2: %struct_type.a.b.501 = bind_symbolic_name S, 0 [symbolic = %S.loc8_9.2 (constants.%S)]
// CHECK:STDOUT:   %S.patt.loc8_9.2: %struct_type.a.b.501 = symbolic_binding_pattern S, 0 [symbolic = %S.patt.loc8_9.2 (constants.%S.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.775
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.c8f;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc4: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc4_31.1: %struct_type.a.a6c = struct_literal (%int_0.loc4)
// CHECK:STDOUT:   %impl.elem0.loc4: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc4_31.1: <bound method> = bound_method %int_0.loc4, %impl.elem0.loc4 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc4: <specific function> = specific_function %impl.elem0.loc4, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_31.2: <bound method> = bound_method %int_0.loc4, %specific_fn.loc4 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc4: init %i32 = call %bound_method.loc4_31.2(%int_0.loc4) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc4_31.2: init %i32 = converted %int_0.loc4, %int.convert_checked.loc4 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc4_31.3: init %struct_type.a.ba9 = struct_init (%.loc4_31.2) to file.%a_ref.var [concrete = constants.%struct.92d]
// CHECK:STDOUT:   %.loc4_1: init %struct_type.a.ba9 = converted %.loc4_31.1, %.loc4_31.3 [concrete = constants.%struct.92d]
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc4_1
// CHECK:STDOUT:   %int_0.loc6_17: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_0.loc6_26: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc6_28.1: %tuple.type.985 = tuple_literal (%int_0.loc6_26)
// CHECK:STDOUT:   %.loc6_29.1: %struct_type.b.c.9af = struct_literal (%int_0.loc6_17, %.loc6_28.1)
// CHECK:STDOUT:   %int_0.loc6_37: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc6_38.1: %struct_type.a.d.b82 = struct_literal (%.loc6_29.1, %int_0.loc6_37)
// CHECK:STDOUT:   %impl.elem0.loc6_29: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc6_29.1: <bound method> = bound_method %int_0.loc6_17, %impl.elem0.loc6_29 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc6_29: <specific function> = specific_function %impl.elem0.loc6_29, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_29.2: <bound method> = bound_method %int_0.loc6_17, %specific_fn.loc6_29 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc6_29: init %i32 = call %bound_method.loc6_29.2(%int_0.loc6_17) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_29.2: init %i32 = converted %int_0.loc6_17, %int.convert_checked.loc6_29 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_38.2: ref %struct_type.b.c.929 = struct_access file.%b_ref.var, element0
// CHECK:STDOUT:   %.loc6_29.3: ref %i32 = struct_access %.loc6_38.2, element0
// CHECK:STDOUT:   %.loc6_29.4: init %i32 = initialize_from %.loc6_29.2 to %.loc6_29.3 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %impl.elem0.loc6_28: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc6_28.1: <bound method> = bound_method %int_0.loc6_26, %impl.elem0.loc6_28 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc6_28: <specific function> = specific_function %impl.elem0.loc6_28, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_28.2: <bound method> = bound_method %int_0.loc6_26, %specific_fn.loc6_28 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc6_28: init %i32 = call %bound_method.loc6_28.2(%int_0.loc6_26) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_28.2: init %i32 = converted %int_0.loc6_26, %int.convert_checked.loc6_28 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_29.5: ref %tuple.type.a1c = struct_access %.loc6_38.2, element1
// CHECK:STDOUT:   %.loc6_28.3: init %tuple.type.a1c = tuple_init (%.loc6_28.2) to %.loc6_29.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.6: init %tuple.type.a1c = converted %.loc6_28.1, %.loc6_28.3 [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.7: init %tuple.type.a1c = initialize_from %.loc6_29.6 to %.loc6_29.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.8: init %struct_type.b.c.929 = struct_init (%.loc6_29.4, %.loc6_29.7) to %.loc6_38.2 [concrete = constants.%struct.381]
// CHECK:STDOUT:   %.loc6_38.3: init %struct_type.b.c.929 = converted %.loc6_29.1, %.loc6_29.8 [concrete = constants.%struct.381]
// CHECK:STDOUT:   %impl.elem0.loc6_38: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc6_38.1: <bound method> = bound_method %int_0.loc6_37, %impl.elem0.loc6_38 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc6_38: <specific function> = specific_function %impl.elem0.loc6_38, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_38.2: <bound method> = bound_method %int_0.loc6_37, %specific_fn.loc6_38 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc6_38: init %i32 = call %bound_method.loc6_38.2(%int_0.loc6_37) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_38.4: init %i32 = converted %int_0.loc6_37, %int.convert_checked.loc6_38 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_38.5: ref %i32 = struct_access file.%b_ref.var, element1
// CHECK:STDOUT:   %.loc6_38.6: init %i32 = initialize_from %.loc6_38.4 to %.loc6_38.5 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_38.7: init %struct_type.a.d.3d9 = struct_init (%.loc6_38.3, %.loc6_38.6) to file.%b_ref.var [concrete = constants.%struct.20d]
// CHECK:STDOUT:   %.loc5: init %struct_type.a.d.3d9 = converted %.loc6_38.1, %.loc6_38.7 [concrete = constants.%struct.20d]
// CHECK:STDOUT:   assign file.%b_ref.var, %.loc5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S.loc8_9.2 => constants.%S
// CHECK:STDOUT:   %S.patt.loc8_9.2 => constants.%S.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct.ed5) {
// CHECK:STDOUT:   %S.loc8_9.2 => constants.%struct.ed5
// CHECK:STDOUT:   %S.patt.loc8_9.2 => constants.%S.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.type.dd4: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %struct_type.b.c: type = struct_type {.b: %i32, .c: %tuple.type.dd4} [concrete]
// CHECK:STDOUT:   %struct_type.a.d.9db: type = struct_type {.a: %struct_type.b.c, .d: %i32} [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %struct_type.a.b.5ca: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %S: %struct_type.a.b.5ca = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %S.patt: %struct_type.a.b.5ca = symbolic_binding_pattern S, 0 [symbolic]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
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
// CHECK:STDOUT:   %struct: %struct_type.a.b.5ca = struct_value (%int_1.47b, %int_2.d0d) [concrete]
// CHECK:STDOUT:   %C.a8a: type = class_type @C, @C(%struct) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Implicit.a_ref: ref %struct_type.a = import_ref Implicit//default, a_ref, loaded
// CHECK:STDOUT:   %Implicit.b_ref: ref %struct_type.a.d.9db = import_ref Implicit//default, b_ref, loaded
// CHECK:STDOUT:   %Implicit.C: %C.type = import_ref Implicit//default, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Implicit.F: %F.type = import_ref Implicit//default, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import_ref.c81: %struct_type.a.b.5ca = import_ref Implicit//default, loc8_9, loaded [symbolic = @C.%S (constants.%S)]
// CHECK:STDOUT:   %Implicit.import_ref.8f2: <witness> = import_ref Implicit//default, loc8_34, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Implicit.import_ref.b8b = import_ref Implicit//default, inst1143 [no loc], unloaded
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
// CHECK:STDOUT:     %a.patt: %struct_type.a = binding_pattern a
// CHECK:STDOUT:     %.loc4_1: %struct_type.a = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %struct_type.a = var a
// CHECK:STDOUT:   %.loc4_16: type = splice_block %struct_type.a [concrete = constants.%struct_type.a] {
// CHECK:STDOUT:     %int_32.loc4: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %struct_type.a = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %struct_type.a.d.9db = binding_pattern b
// CHECK:STDOUT:     %.loc5_1: %struct_type.a.d.9db = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %struct_type.a.d.9db = var b
// CHECK:STDOUT:   %.loc5_43: type = splice_block %struct_type.a.d [concrete = constants.%struct_type.a.d.9db] {
// CHECK:STDOUT:     %int_32.loc5_18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_28: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_28: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_32.1: %tuple.type.85c = tuple_literal (%i32.loc5_28)
// CHECK:STDOUT:     %.loc5_32.2: type = converted %.loc5_32.1, constants.%tuple.type.dd4 [concrete = constants.%tuple.type.dd4]
// CHECK:STDOUT:     %struct_type.b.c: type = struct_type {.b: %i32, .c: %tuple.type.dd4} [concrete = constants.%struct_type.b.c]
// CHECK:STDOUT:     %int_32.loc5_40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_40: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.d: type = struct_type {.a: %struct_type.b.c, .d: %i32} [concrete = constants.%struct_type.a.d.9db]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %struct_type.a.d.9db = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C.a8a = binding_pattern c
// CHECK:STDOUT:     %.loc6_1: %C.a8a = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.a8a = var c
// CHECK:STDOUT:   %.loc6_26.1: type = splice_block %C [concrete = constants.%C.a8a] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Implicit.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc6_25.1: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:     %impl.elem0.loc6_25.1: %.419 = impl_witness_access constants.%impl_witness.847, element0 [concrete = constants.%Convert.4cb]
// CHECK:STDOUT:     %bound_method.loc6_25.1: <bound method> = bound_method %int_1, %impl.elem0.loc6_25.1 [concrete = constants.%Convert.bound.dc5]
// CHECK:STDOUT:     %specific_fn.loc6_25.1: <specific function> = specific_function %impl.elem0.loc6_25.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_25.2: <bound method> = bound_method %int_1, %specific_fn.loc6_25.1 [concrete = constants.%bound_method.c37]
// CHECK:STDOUT:     %int.convert_checked.loc6_25.1: init %i32 = call %bound_method.loc6_25.2(%int_1) [concrete = constants.%int_1.47b]
// CHECK:STDOUT:     %.loc6_25.2: %i32 = value_of_initializer %int.convert_checked.loc6_25.1 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:     %.loc6_25.3: %i32 = converted %int_1, %.loc6_25.2 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:     %impl.elem0.loc6_25.2: %.419 = impl_witness_access constants.%impl_witness.847, element0 [concrete = constants.%Convert.4cb]
// CHECK:STDOUT:     %bound_method.loc6_25.3: <bound method> = bound_method %int_2, %impl.elem0.loc6_25.2 [concrete = constants.%Convert.bound.30f]
// CHECK:STDOUT:     %specific_fn.loc6_25.2: <specific function> = specific_function %impl.elem0.loc6_25.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_25.4: <bound method> = bound_method %int_2, %specific_fn.loc6_25.2 [concrete = constants.%bound_method.62b]
// CHECK:STDOUT:     %int.convert_checked.loc6_25.2: init %i32 = call %bound_method.loc6_25.4(%int_2) [concrete = constants.%int_2.d0d]
// CHECK:STDOUT:     %.loc6_25.4: %i32 = value_of_initializer %int.convert_checked.loc6_25.2 [concrete = constants.%int_2.d0d]
// CHECK:STDOUT:     %.loc6_25.5: %i32 = converted %int_2, %.loc6_25.4 [concrete = constants.%int_2.d0d]
// CHECK:STDOUT:     %struct: %struct_type.a.b.5ca = struct_value (%.loc6_25.3, %.loc6_25.5) [concrete = constants.%struct]
// CHECK:STDOUT:     %.loc6_26.2: %struct_type.a.b.5ca = converted %.loc6_25.1, %struct [concrete = constants.%struct]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%struct) [concrete = constants.%C.a8a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.a8a = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Implicit.import_ref.c81: %struct_type.a.b.5ca) [from "implicit.carbon"] {
// CHECK:STDOUT:   %S: %struct_type.a.b.5ca = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:   %S.patt: %struct_type.a.b.5ca = symbolic_binding_pattern S, 0 [symbolic = %S.patt (constants.%S.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Implicit.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Implicit.import_ref.b8b
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.a8a [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %struct_type.a = name_ref a_ref, imports.%Implicit.a_ref
// CHECK:STDOUT:   %.loc4_20.1: ref %i32 = struct_access %a_ref.ref, element0
// CHECK:STDOUT:   %.loc4_20.2: %i32 = bind_value %.loc4_20.1
// CHECK:STDOUT:   %.loc4_20.3: init %struct_type.a = struct_init (%.loc4_20.2) to file.%a.var
// CHECK:STDOUT:   %.loc4_1: init %struct_type.a = converted %a_ref.ref, %.loc4_20.3
// CHECK:STDOUT:   assign file.%a.var, %.loc4_1
// CHECK:STDOUT:   %b_ref.ref: ref %struct_type.a.d.9db = name_ref b_ref, imports.%Implicit.b_ref
// CHECK:STDOUT:   %.loc5_47.1: ref %struct_type.b.c = struct_access %b_ref.ref, element0
// CHECK:STDOUT:   %.loc5_47.2: ref %i32 = struct_access %.loc5_47.1, element0
// CHECK:STDOUT:   %.loc5_47.3: %i32 = bind_value %.loc5_47.2
// CHECK:STDOUT:   %.loc5_47.4: ref %struct_type.b.c = struct_access file.%b.var, element0
// CHECK:STDOUT:   %.loc5_47.5: ref %i32 = struct_access %.loc5_47.4, element0
// CHECK:STDOUT:   %.loc5_47.6: init %i32 = initialize_from %.loc5_47.3 to %.loc5_47.5
// CHECK:STDOUT:   %.loc5_47.7: ref %tuple.type.dd4 = struct_access %.loc5_47.1, element1
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %.loc5_47.7, element0
// CHECK:STDOUT:   %.loc5_47.8: %i32 = bind_value %tuple.elem0
// CHECK:STDOUT:   %.loc5_47.9: ref %tuple.type.dd4 = struct_access %.loc5_47.4, element1
// CHECK:STDOUT:   %.loc5_47.10: init %tuple.type.dd4 = tuple_init (%.loc5_47.8) to %.loc5_47.9
// CHECK:STDOUT:   %.loc5_47.11: init %tuple.type.dd4 = converted %.loc5_47.7, %.loc5_47.10
// CHECK:STDOUT:   %.loc5_47.12: init %tuple.type.dd4 = initialize_from %.loc5_47.11 to %.loc5_47.9
// CHECK:STDOUT:   %.loc5_47.13: init %struct_type.b.c = struct_init (%.loc5_47.6, %.loc5_47.12) to %.loc5_47.4
// CHECK:STDOUT:   %.loc5_47.14: init %struct_type.b.c = converted %.loc5_47.1, %.loc5_47.13
// CHECK:STDOUT:   %.loc5_47.15: ref %i32 = struct_access %b_ref.ref, element1
// CHECK:STDOUT:   %.loc5_47.16: %i32 = bind_value %.loc5_47.15
// CHECK:STDOUT:   %.loc5_47.17: ref %i32 = struct_access file.%b.var, element1
// CHECK:STDOUT:   %.loc5_47.18: init %i32 = initialize_from %.loc5_47.16 to %.loc5_47.17
// CHECK:STDOUT:   %.loc5_47.19: init %struct_type.a.d.9db = struct_init (%.loc5_47.14, %.loc5_47.18) to file.%b.var
// CHECK:STDOUT:   %.loc5_1: init %struct_type.a.d.9db = converted %b_ref.ref, %.loc5_47.19
// CHECK:STDOUT:   assign file.%b.var, %.loc5_1
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Implicit.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc6: ref %C.a8a = splice_block file.%c.var {}
// CHECK:STDOUT:   %F.call: init %C.a8a = call %F.ref() to %.loc6
// CHECK:STDOUT:   assign file.%c.var, %F.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT:   %S.patt => constants.%S.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct) {
// CHECK:STDOUT:   %S => constants.%struct
// CHECK:STDOUT:   %S.patt => constants.%S.patt
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
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %S: %struct_type.a.b = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %S.patt: %struct_type.a.b = symbolic_binding_pattern S, 0 [symbolic]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.c.d: type = struct_type {.c: Core.IntLiteral, .d: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.d0d: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_1.47b: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b = struct_value (%int_1.47b, %int_2.d0d) [concrete]
// CHECK:STDOUT:   %C.a8a: type = class_type @C, @C(%struct) [concrete]
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
// CHECK:STDOUT:   %Implicit.import_ref.c81: %struct_type.a.b = import_ref Implicit//default, loc8_9, loaded [symbolic = @C.%S (constants.%S)]
// CHECK:STDOUT:   %Implicit.import_ref.8f2: <witness> = import_ref Implicit//default, loc8_34, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Implicit.import_ref.b8b = import_ref Implicit//default, inst1143 [no loc], unloaded
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
// CHECK:STDOUT:     %.loc11_1: <error> = var_pattern %c_bad.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_bad.var: ref <error> = var c_bad
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Implicit.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc11_29: %struct_type.c.d = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_bad: <error> = bind_name c_bad, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Implicit.import_ref.c81: %struct_type.a.b) [from "implicit.carbon"] {
// CHECK:STDOUT:   %S: %struct_type.a.b = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:   %S.patt: %struct_type.a.b = symbolic_binding_pattern S, 0 [symbolic = %S.patt (constants.%S.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Implicit.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Implicit.import_ref.b8b
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.a8a [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Implicit.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc11: ref %C.a8a = temporary_storage
// CHECK:STDOUT:   %F.call: init %C.a8a = call %F.ref() to %.loc11
// CHECK:STDOUT:   assign file.%c_bad.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT:   %S.patt => constants.%S.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct) {
// CHECK:STDOUT:   %S => constants.%struct
// CHECK:STDOUT:   %S.patt => constants.%S.patt
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
// CHECK:STDOUT:   %struct_type.a.b.5ca: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %S: %struct_type.a.b.5ca = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %S.patt: %struct_type.a.b.5ca = symbolic_binding_pattern S, 0 [symbolic]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
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
// CHECK:STDOUT:   %struct.8e6: %struct_type.a.b.5ca = struct_value (%int_3.d47, %int_4.ea3) [concrete]
// CHECK:STDOUT:   %C.5a7: type = class_type @C, @C(%struct.8e6) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_2: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_1: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct.cd9: %struct_type.a.b.5ca = struct_value (%int_1, %int_2) [concrete]
// CHECK:STDOUT:   %C.a8a: type = class_type @C, @C(%struct.cd9) [concrete]
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
// CHECK:STDOUT:   %Implicit.import_ref.c81: %struct_type.a.b.5ca = import_ref Implicit//default, loc8_9, loaded [symbolic = @C.%S (constants.%S)]
// CHECK:STDOUT:   %Implicit.import_ref.8f2: <witness> = import_ref Implicit//default, loc8_34, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Implicit.import_ref.b8b = import_ref Implicit//default, inst1143 [no loc], unloaded
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
// CHECK:STDOUT:     %c_bad.patt: %C.5a7 = binding_pattern c_bad
// CHECK:STDOUT:     %.loc10_1: %C.5a7 = var_pattern %c_bad.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_bad.var: ref %C.5a7 = var c_bad
// CHECK:STDOUT:   %.loc10_30.1: type = splice_block %C [concrete = constants.%C.5a7] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Implicit.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:     %.loc10_29.1: %struct_type.a.b.cfd = struct_literal (%int_3, %int_4)
// CHECK:STDOUT:     %impl.elem0.loc10_29.1: %.419 = impl_witness_access constants.%impl_witness.847, element0 [concrete = constants.%Convert.4cb]
// CHECK:STDOUT:     %bound_method.loc10_29.1: <bound method> = bound_method %int_3, %impl.elem0.loc10_29.1 [concrete = constants.%Convert.bound.a00]
// CHECK:STDOUT:     %specific_fn.loc10_29.1: <specific function> = specific_function %impl.elem0.loc10_29.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_29.2: <bound method> = bound_method %int_3, %specific_fn.loc10_29.1 [concrete = constants.%bound_method.fc5]
// CHECK:STDOUT:     %int.convert_checked.loc10_29.1: init %i32 = call %bound_method.loc10_29.2(%int_3) [concrete = constants.%int_3.d47]
// CHECK:STDOUT:     %.loc10_29.2: %i32 = value_of_initializer %int.convert_checked.loc10_29.1 [concrete = constants.%int_3.d47]
// CHECK:STDOUT:     %.loc10_29.3: %i32 = converted %int_3, %.loc10_29.2 [concrete = constants.%int_3.d47]
// CHECK:STDOUT:     %impl.elem0.loc10_29.2: %.419 = impl_witness_access constants.%impl_witness.847, element0 [concrete = constants.%Convert.4cb]
// CHECK:STDOUT:     %bound_method.loc10_29.3: <bound method> = bound_method %int_4, %impl.elem0.loc10_29.2 [concrete = constants.%Convert.bound.694]
// CHECK:STDOUT:     %specific_fn.loc10_29.2: <specific function> = specific_function %impl.elem0.loc10_29.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_29.4: <bound method> = bound_method %int_4, %specific_fn.loc10_29.2 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:     %int.convert_checked.loc10_29.2: init %i32 = call %bound_method.loc10_29.4(%int_4) [concrete = constants.%int_4.ea3]
// CHECK:STDOUT:     %.loc10_29.4: %i32 = value_of_initializer %int.convert_checked.loc10_29.2 [concrete = constants.%int_4.ea3]
// CHECK:STDOUT:     %.loc10_29.5: %i32 = converted %int_4, %.loc10_29.4 [concrete = constants.%int_4.ea3]
// CHECK:STDOUT:     %struct: %struct_type.a.b.5ca = struct_value (%.loc10_29.3, %.loc10_29.5) [concrete = constants.%struct.8e6]
// CHECK:STDOUT:     %.loc10_30.2: %struct_type.a.b.5ca = converted %.loc10_29.1, %struct [concrete = constants.%struct.8e6]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%struct.8e6) [concrete = constants.%C.5a7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_bad: ref %C.5a7 = bind_name c_bad, %c_bad.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Implicit.import_ref.c81: %struct_type.a.b.5ca) [from "implicit.carbon"] {
// CHECK:STDOUT:   %S: %struct_type.a.b.5ca = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:   %S.patt: %struct_type.a.b.5ca = symbolic_binding_pattern S, 0 [symbolic = %S.patt (constants.%S.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Implicit.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Implicit.import_ref.b8b
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.a8a [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Implicit.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc10_36: ref %C.a8a = temporary_storage
// CHECK:STDOUT:   %F.call: init %C.a8a = call %F.ref() to %.loc10_36
// CHECK:STDOUT:   %.loc10_1: %C.5a7 = converted %F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%c_bad.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT:   %S.patt => constants.%S.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct.8e6) {
// CHECK:STDOUT:   %S => constants.%struct.8e6
// CHECK:STDOUT:   %S.patt => constants.%S.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct.cd9) {
// CHECK:STDOUT:   %S => constants.%struct.cd9
// CHECK:STDOUT:   %S.patt => constants.%S.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
