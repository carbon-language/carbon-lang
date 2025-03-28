// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/fail_member_access_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/fail_member_access_type.carbon

var x: {.a: f64} = {.a = 4.0};
// CHECK:STDERR: fail_member_access_type.carbon:[[@LINE+4]]:14: error: type `{.a: f64}` does not have a member `b` [QualifiedExprNameNotFound]
// CHECK:STDERR: var y: i32 = x.b;
// CHECK:STDERR:              ^~~
// CHECK:STDERR:
var y: i32 = x.b;

// CHECK:STDOUT: --- fail_member_access_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: f64} [concrete]
// CHECK:STDOUT:   %float: f64 = float_literal 4 [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a = struct_value (%float) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %struct_type.a = binding_pattern x
// CHECK:STDOUT:     %.loc11_1: %struct_type.a = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.a = var x
// CHECK:STDOUT:   %.loc11_16: type = splice_block %struct_type.a [concrete = constants.%struct_type.a] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %float.make_type: init type = call constants.%Float(%int_64) [concrete = f64]
// CHECK:STDOUT:     %.loc11_13.1: type = value_of_initializer %float.make_type [concrete = f64]
// CHECK:STDOUT:     %.loc11_13.2: type = converted %float.make_type, %.loc11_13.1 [concrete = f64]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: f64} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.a = bind_name x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %i32 = binding_pattern y
// CHECK:STDOUT:     %.loc16_1: %i32 = var_pattern %y.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %i32 = var y
// CHECK:STDOUT:   %.loc16_8: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %i32 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %float: f64 = float_literal 4 [concrete = constants.%float]
// CHECK:STDOUT:   %.loc11_29.1: %struct_type.a = struct_literal (%float)
// CHECK:STDOUT:   %.loc11_29.2: init %struct_type.a = struct_init (%float) to file.%x.var [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc11_1: init %struct_type.a = converted %.loc11_29.1, %.loc11_29.2 [concrete = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_1
// CHECK:STDOUT:   %x.ref: ref %struct_type.a = name_ref x, file.%x
// CHECK:STDOUT:   assign file.%y.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
