// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/fail_not_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/fail_not_type.carbon

// CHECK:STDERR: fail_not_type.carbon:[[@LINE+6]]:19: error: cannot implicitly convert from `Core.IntLiteral` to `type` [ImplicitAsConversionFailure]
// CHECK:STDERR: let n: i32 = 1 as 2;
// CHECK:STDERR:                   ^
// CHECK:STDERR: fail_not_type.carbon:[[@LINE+3]]:19: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(type)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let n: i32 = 1 as 2;
// CHECK:STDERR:                   ^
let n: i32 = 1 as 2;

// CHECK:STDOUT: --- fail_not_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .n = @__global_init.%n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int.32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_8.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_8.2: type = converted %int.make_type_signed, %.loc17_8.1 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1]
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template = constants.%int.2]
// CHECK:STDOUT:   %.loc17: type = converted %int.2, <error> [template = <error>]
// CHECK:STDOUT:   %n: %i32 = bind_name n, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
