// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/fail_out_of_bound.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/fail_out_of_bound.carbon

// CHECK:STDERR: fail_out_of_bound.carbon:[[@LINE+3]]:19: error: cannot initialize array of 1 element from 3 initializers [ArrayInitFromLiteralArgCountMismatch]
// CHECK:STDERR: var a: [i32; 1] = (1, 2, 3);
// CHECK:STDERR:                   ^~~~~~~~~
var a: [i32; 1] = (1, 2, 3);

// CHECK:STDOUT: --- fail_out_of_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.1: type = array_type %int.1, %i32 [template]
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int.3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int.32) [template = constants.%i32]
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1]
// CHECK:STDOUT:   %.loc14_9.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_9.2: type = converted %int.make_type_signed, %.loc14_9.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_15: type = array_type %int.1, %i32 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref %.1 = var a
// CHECK:STDOUT:   %a: ref %.1 = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1]
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template = constants.%int.2]
// CHECK:STDOUT:   %int.3: Core.IntLiteral = int_value 3 [template = constants.%int.3]
// CHECK:STDOUT:   %.loc14: %tuple.type = tuple_literal (%int.1, %int.2, %int.3)
// CHECK:STDOUT:   assign file.%a.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
