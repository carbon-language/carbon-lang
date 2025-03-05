// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/fail_out_of_bound_non_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/fail_out_of_bound_non_literal.carbon

var a: array(i32, 3) = (1, 2, 3);
// CHECK:STDERR: fail_out_of_bound_non_literal.carbon:[[@LINE+4]]:16: error: array index `3` is past the end of type `[i32; 3]` [ArrayIndexOutOfBounds]
// CHECK:STDERR: var b: i32 = a[{.index = 3}.index];
// CHECK:STDERR:                ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var b: i32 = a[{.index = 3}.index];

// CHECK:STDOUT: --- fail_out_of_bound_non_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.5d2, %int_2.ef8, %int_3.822) [concrete]
// CHECK:STDOUT:   %struct_type.index: type = struct_type {.index: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.index = struct_value (%int_3.1ba) [concrete]
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
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type = binding_pattern a
// CHECK:STDOUT:     %.loc11_1: %array_type = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var a
// CHECK:STDOUT:   %.loc11_20: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %.loc16_1: %i32 = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %.loc16_8: type = splice_block %i32.loc16 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc11_25: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc11_28: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc11: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc11_32.1: %tuple.type = tuple_literal (%int_1.loc11_25, %int_2.loc11_28, %int_3.loc11)
// CHECK:STDOUT:   %impl.elem0.loc11_32.1: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_32.1: <bound method> = bound_method %int_1.loc11_25, %impl.elem0.loc11_32.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc11_32.1: <specific function> = specific_function %impl.elem0.loc11_32.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_32.2: <bound method> = bound_method %int_1.loc11_25, %specific_fn.loc11_32.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc11_32.1: init %i32 = call %bound_method.loc11_32.2(%int_1.loc11_25) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_32.2: init %i32 = converted %int_1.loc11_25, %int.convert_checked.loc11_32.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_32.3: ref %i32 = array_index file.%a.var, %int_0
// CHECK:STDOUT:   %.loc11_32.4: init %i32 = initialize_from %.loc11_32.2 to %.loc11_32.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc11_32.2: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_32.3: <bound method> = bound_method %int_2.loc11_28, %impl.elem0.loc11_32.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc11_32.2: <specific function> = specific_function %impl.elem0.loc11_32.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_32.4: <bound method> = bound_method %int_2.loc11_28, %specific_fn.loc11_32.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc11_32.2: init %i32 = call %bound_method.loc11_32.4(%int_2.loc11_28) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc11_32.5: init %i32 = converted %int_2.loc11_28, %int.convert_checked.loc11_32.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc11_32: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc11_32.6: ref %i32 = array_index file.%a.var, %int_1.loc11_32
// CHECK:STDOUT:   %.loc11_32.7: init %i32 = initialize_from %.loc11_32.5 to %.loc11_32.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc11_32.3: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_32.5: <bound method> = bound_method %int_3.loc11, %impl.elem0.loc11_32.3 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc11_32.3: <specific function> = specific_function %impl.elem0.loc11_32.3, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_32.6: <bound method> = bound_method %int_3.loc11, %specific_fn.loc11_32.3 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc11_32.3: init %i32 = call %bound_method.loc11_32.6(%int_3.loc11) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc11_32.8: init %i32 = converted %int_3.loc11, %int.convert_checked.loc11_32.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc11_32: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc11_32.9: ref %i32 = array_index file.%a.var, %int_2.loc11_32
// CHECK:STDOUT:   %.loc11_32.10: init %i32 = initialize_from %.loc11_32.8 to %.loc11_32.9 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc11_32.11: init %array_type = array_init (%.loc11_32.4, %.loc11_32.7, %.loc11_32.10) to file.%a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_1: init %array_type = converted %.loc11_32.1, %.loc11_32.11 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_1
// CHECK:STDOUT:   %a.ref: ref %array_type = name_ref a, file.%a
// CHECK:STDOUT:   %int_3.loc16: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc16_27.1: %struct_type.index = struct_literal (%int_3.loc16)
// CHECK:STDOUT:   %struct: %struct_type.index = struct_value (%int_3.loc16) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc16_27.2: %struct_type.index = converted %.loc16_27.1, %struct [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc16_28.1: Core.IntLiteral = struct_access %.loc16_27.2, element0 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc16: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc16_28.1: <bound method> = bound_method %.loc16_28.1, %impl.elem0.loc16 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_28.2: <bound method> = bound_method %.loc16_28.1, %specific_fn.loc16 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc16: init %i32 = call %bound_method.loc16_28.2(%.loc16_28.1) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc16_28.2: %i32 = value_of_initializer %int.convert_checked.loc16 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc16_28.3: %i32 = converted %.loc16_28.1, %.loc16_28.2 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc16_34.1: ref %i32 = array_index %a.ref, %.loc16_28.3 [concrete = <error>]
// CHECK:STDOUT:   %.loc16_34.2: %i32 = bind_value %.loc16_34.1 [concrete = <error>]
// CHECK:STDOUT:   assign file.%b.var, %.loc16_34.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
