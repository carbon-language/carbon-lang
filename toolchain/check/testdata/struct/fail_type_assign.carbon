// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/fail_type_assign.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/fail_type_assign.carbon

// CHECK:STDERR: fail_type_assign.carbon:[[@LINE+7]]:1: error: cannot implicitly convert value of type `type` to `{.a: i32}` [ConversionFailure]
// CHECK:STDERR: var x: {.a: i32} = {.a: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_type_assign.carbon:[[@LINE+4]]:1: note: type `type` does not implement interface `Core.ImplicitAs({.a: i32})` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var x: {.a: i32} = {.a: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
var x: {.a: i32} = {.a: i32};

// CHECK:STDOUT: --- fail_type_assign.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [concrete]
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
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %struct_type.a = binding_pattern x
// CHECK:STDOUT:     %.loc18_1: %struct_type.a = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.a = var x
// CHECK:STDOUT:   %.loc18_16: type = splice_block %struct_type.a [concrete = constants.%struct_type.a] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.a = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:   %.loc18: %struct_type.a = converted %struct_type.a, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
