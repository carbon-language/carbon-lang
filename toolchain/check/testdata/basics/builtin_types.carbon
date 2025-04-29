// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/builtin_types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/builtin_types.carbon

var test_i32: i32 = 0;
var test_f64: f64 = 0.1;
let test_str: String = "Test";
var test_type: type = i32;

// CHECK:STDOUT: --- builtin_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (imports.%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness %ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.3de: type = pattern_type f64 [concrete]
// CHECK:STDOUT:   %float: f64 = float_literal 0.10000000000000001 [concrete]
// CHECK:STDOUT:   %pattern_type.b05: type = pattern_type String [concrete]
// CHECK:STDOUT:   %str: String = string_literal "Test" [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .test_i32 = %test_i32
// CHECK:STDOUT:     .test_f64 = %test_f64
// CHECK:STDOUT:     .test_str = %test_str
// CHECK:STDOUT:     .test_type = %test_type
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_i32.patt: %pattern_type.7ce = binding_pattern test_i32 [concrete]
// CHECK:STDOUT:     %.loc11_1: %pattern_type.7ce = var_pattern %test_i32.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i32.var: ref %i32 = var test_i32
// CHECK:STDOUT:   %.loc11_15: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i32: ref %i32 = bind_name test_i32, %test_i32.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_f64.patt: %pattern_type.3de = binding_pattern test_f64 [concrete]
// CHECK:STDOUT:     %.loc12_1: %pattern_type.3de = var_pattern %test_f64.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_f64.var: ref f64 = var test_f64
// CHECK:STDOUT:   %.loc12_15.1: type = splice_block %.loc12_15.3 [concrete = f64] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %float.make_type: init type = call constants.%Float(%int_64) [concrete = f64]
// CHECK:STDOUT:     %.loc12_15.2: type = value_of_initializer %float.make_type [concrete = f64]
// CHECK:STDOUT:     %.loc12_15.3: type = converted %float.make_type, %.loc12_15.2 [concrete = f64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_f64: ref f64 = bind_name test_f64, %test_f64.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_str.patt: %pattern_type.b05 = binding_pattern test_str [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_str: String = bind_name test_str, @__global_init.%str
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_type.patt: %pattern_type.98f = binding_pattern test_type [concrete]
// CHECK:STDOUT:     %.loc14: %pattern_type.98f = var_pattern %test_type.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_type.var: ref type = var test_type
// CHECK:STDOUT:   %test_type: ref type = bind_name test_type, %test_type.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_1.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_1.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc11_1.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc11: init %i32 = converted %int_0, %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   assign file.%test_i32.var, %.loc11
// CHECK:STDOUT:   %float: f64 = float_literal 0.10000000000000001 [concrete = constants.%float]
// CHECK:STDOUT:   assign file.%test_f64.var, %float
// CHECK:STDOUT:   %str: String = string_literal "Test" [concrete = constants.%str]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   assign file.%test_type.var, %i32
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
