// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/fail_builtins.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/fail_builtins.carbon

// CHECK:STDERR: fail_builtins.carbon:[[@LINE+4]]:11: ERROR: Alias initializer must be a name reference.
// CHECK:STDERR: alias a = i32;
// CHECK:STDERR:           ^~~
// CHECK:STDERR:
alias a = i32;

// CHECK:STDERR: fail_builtins.carbon:[[@LINE+3]]:11: ERROR: Alias initializer must be a name reference.
// CHECK:STDERR: alias b = bool;
// CHECK:STDERR:           ^~~~
alias b = bool;

// CHECK:STDOUT: --- fail_builtins.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %Bool: type = fn_type @Bool [template]
// CHECK:STDOUT:   %struct.2: Bool = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %a: <error> = bind_alias a, <error> [template = <error>]
// CHECK:STDOUT:   %import_ref.2: Bool = import_ref ir7, inst+2, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %bool.make_type: init type = call constants.%struct.2() [template = bool]
// CHECK:STDOUT:   %b: <error> = bind_alias b, <error> [template = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
