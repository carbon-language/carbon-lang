// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/fail_numeric_literal_overflow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/fail_numeric_literal_overflow.carbon

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:14: error: integer literal with value 39999999999999999993 does not fit in i32
// CHECK:STDERR: let a: i32 = 39999999999999999993;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let a: i32 = 39999999999999999993;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:14: error: integer literal with value 2147483648 does not fit in i32
// CHECK:STDERR: let b: i32 = 2_147_483_648;
// CHECK:STDERR:              ^~~~~~~~~~~~~
// CHECK:STDERR:
let b: i32 = 2_147_483_648;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:14: error: integer literal with value 2147483648 does not fit in i32
// CHECK:STDERR: let c: i32 = 0x8000_0000;
// CHECK:STDERR:              ^~~~~~~~~~~
// CHECK:STDERR:
let c: i32 = 0x8000_0000;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:14: error: real mantissa with value 399999999999999999930 does not fit in i64
// CHECK:STDERR: let d: f64 = 39999999999999999993.0e3;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let d: f64 = 39999999999999999993.0e3;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+3]]:14: error: real exponent with value 39999999999999999992 does not fit in i64
// CHECK:STDERR: let e: f64 = 5.0e39999999999999999993;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~
let e: f64 = 5.0e39999999999999999993;

// CHECK:STDOUT: --- fail_numeric_literal_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
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
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:     .d = @__global_init.%d
// CHECK:STDOUT:     .e = @__global_init.%e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_8.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_8.2: type = converted %int.make_type_32.loc15, %.loc15_8.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc21_8.1: type = value_of_initializer %int.make_type_32.loc21 [template = i32]
// CHECK:STDOUT:   %.loc21_8.2: type = converted %int.make_type_32.loc21, %.loc21_8.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc27_8.1: type = value_of_initializer %int.make_type_32.loc27 [template = i32]
// CHECK:STDOUT:   %.loc27_8.2: type = converted %int.make_type_32.loc27, %.loc27_8.1 [template = i32]
// CHECK:STDOUT:   %.loc33_8.1: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:   %float.make_type.loc33: init type = call constants.%Float(%.loc33_8.1) [template = f64]
// CHECK:STDOUT:   %.loc33_8.2: type = value_of_initializer %float.make_type.loc33 [template = f64]
// CHECK:STDOUT:   %.loc33_8.3: type = converted %float.make_type.loc33, %.loc33_8.2 [template = f64]
// CHECK:STDOUT:   %.loc38_8.1: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:   %float.make_type.loc38: init type = call constants.%Float(%.loc38_8.1) [template = f64]
// CHECK:STDOUT:   %.loc38_8.2: type = value_of_initializer %float.make_type.loc38 [template = f64]
// CHECK:STDOUT:   %.loc38_8.3: type = converted %float.make_type.loc38, %.loc38_8.2 [template = f64]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a: i32 = bind_name a, <error>
// CHECK:STDOUT:   %b: i32 = bind_name b, <error>
// CHECK:STDOUT:   %c: i32 = bind_name c, <error>
// CHECK:STDOUT:   %d: f64 = bind_name d, <error>
// CHECK:STDOUT:   %e: f64 = bind_name e, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
