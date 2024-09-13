// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/fail_param_redecl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/fail_param_redecl.carbon

// CHECK:STDERR: fail_param_redecl.carbon:[[@LINE+6]]:14: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn F(n: i32, n: i32);
// CHECK:STDERR:              ^
// CHECK:STDERR: fail_param_redecl.carbon:[[@LINE+3]]:6: Name is previously declared here.
// CHECK:STDERR: fn F(n: i32, n: i32);
// CHECK:STDERR:      ^
fn F(n: i32, n: i32);

// CHECK:STDOUT: --- fail_param_redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %int.make_type_32.loc17_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc17_9.1: type = value_of_initializer %int.make_type_32.loc17_9 [template = i32]
// CHECK:STDOUT:     %.loc17_9.2: type = converted %int.make_type_32.loc17_9, %.loc17_9.1 [template = i32]
// CHECK:STDOUT:     %n.loc17_6.1: i32 = param n, runtime_param0
// CHECK:STDOUT:     @F.%n.loc17_6: i32 = bind_name n, %n.loc17_6.1
// CHECK:STDOUT:     %int.make_type_32.loc17_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc17_17.1: type = value_of_initializer %int.make_type_32.loc17_17 [template = i32]
// CHECK:STDOUT:     %.loc17_17.2: type = converted %int.make_type_32.loc17_17, %.loc17_17.1 [template = i32]
// CHECK:STDOUT:     %n.loc17_14.1: i32 = param n, runtime_param1
// CHECK:STDOUT:     @F.%n.loc17_14: i32 = bind_name n, %n.loc17_14.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n.loc17_6: i32, %n.loc17_14: i32);
// CHECK:STDOUT:
