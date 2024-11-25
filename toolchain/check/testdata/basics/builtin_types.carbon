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
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %.30: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %.31: f64 = float_literal 0.10000000000000001 [template]
// CHECK:STDOUT:   %.33: String = string_literal "Test" [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     .Float = %import_ref.38
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .test_i32 = %test_i32
// CHECK:STDOUT:     .test_f64 = %test_f64
// CHECK:STDOUT:     .test_str = @__global_init.%test_str
// CHECK:STDOUT:     .test_type = %test_type
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc11_15.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc11_15.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_15.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_15.3: type = converted %int.make_type_signed, %.loc11_15.2 [template = constants.%i32]
// CHECK:STDOUT:   %test_i32.var: ref %i32 = var test_i32
// CHECK:STDOUT:   %test_i32: ref %i32 = bind_name test_i32, %test_i32.var
// CHECK:STDOUT:   %.loc12_15.1: Core.IntLiteral = int_value 64 [template = constants.%.30]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%.loc12_15.1) [template = f64]
// CHECK:STDOUT:   %.loc12_15.2: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc12_15.3: type = converted %float.make_type, %.loc12_15.2 [template = f64]
// CHECK:STDOUT:   %test_f64.var: ref f64 = var test_f64
// CHECK:STDOUT:   %test_f64: ref f64 = bind_name test_f64, %test_f64.var
// CHECK:STDOUT:   %test_type.var: ref type = var test_type
// CHECK:STDOUT:   %test_type: ref type = bind_name test_type, %test_type.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_21: Core.IntLiteral = int_value 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_22.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_22.2: <bound method> = bound_method %.loc11_21, %.loc11_22.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc11_22.3: <specific function> = specific_function %.loc11_22.2, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc11_22.3(%.loc11_21) [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_22.4: init %i32 = converted %.loc11_21, %int.convert_checked [template = constants.%.29]
// CHECK:STDOUT:   assign file.%test_i32.var, %.loc11_22.4
// CHECK:STDOUT:   %.loc12: f64 = float_literal 0.10000000000000001 [template = constants.%.31]
// CHECK:STDOUT:   assign file.%test_f64.var, %.loc12
// CHECK:STDOUT:   %.loc13: String = string_literal "Test" [template = constants.%.33]
// CHECK:STDOUT:   %test_str: String = bind_name test_str, %.loc13
// CHECK:STDOUT:   %.loc14: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc14) [template = constants.%i32]
// CHECK:STDOUT:   assign file.%test_type.var, %int.make_type_signed
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
