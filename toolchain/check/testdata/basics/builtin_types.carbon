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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %.4: f64 = float_literal 0.10000000000000001 [template]
// CHECK:STDOUT:   %.5: type = ptr_type String [template]
// CHECK:STDOUT:   %.6: String = string_literal "Test" [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .Float = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %Float.type = import_ref Core//prelude/types, inst+32, loaded [template = constants.%Float]
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
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_15.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc11_15.2: type = converted %int.make_type_32, %.loc11_15.1 [template = i32]
// CHECK:STDOUT:   %test_i32.var: ref i32 = var test_i32
// CHECK:STDOUT:   %test_i32: ref i32 = bind_name test_i32, %test_i32.var
// CHECK:STDOUT:   %.loc12_15.1: i32 = int_literal 64 [template = constants.%.3]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%.loc12_15.1) [template = f64]
// CHECK:STDOUT:   %.loc12_15.2: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc12_15.3: type = converted %float.make_type, %.loc12_15.2 [template = f64]
// CHECK:STDOUT:   %test_f64.var: ref f64 = var test_f64
// CHECK:STDOUT:   %test_f64: ref f64 = bind_name test_f64, %test_f64.var
// CHECK:STDOUT:   %test_type.var: ref type = var test_type
// CHECK:STDOUT:   %test_type: ref type = bind_name test_type, %test_type.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   assign file.%test_i32.var, %.loc11
// CHECK:STDOUT:   %.loc12: f64 = float_literal 0.10000000000000001 [template = constants.%.4]
// CHECK:STDOUT:   assign file.%test_f64.var, %.loc12
// CHECK:STDOUT:   %.loc13: String = string_literal "Test" [template = constants.%.6]
// CHECK:STDOUT:   %test_str: String = bind_name test_str, %.loc13
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   assign file.%test_type.var, %int.make_type_32
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
