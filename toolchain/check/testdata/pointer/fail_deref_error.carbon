// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/fail_deref_error.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/fail_deref_error.carbon

// CHECK:STDERR: fail_deref_error.carbon:[[@LINE+4]]:15: error: name `undeclared` not found [NameNotFound]
// CHECK:STDERR: let n: i32 = *undeclared;
// CHECK:STDERR:               ^~~~~~~~~~
// CHECK:STDERR:
let n: i32 = *undeclared;
// CHECK:STDERR: fail_deref_error.carbon:[[@LINE+3]]:15: error: name `undeclared` not found [NameNotFound]
// CHECK:STDERR: let n2: i32 = undeclared->foo;
// CHECK:STDERR:               ^~~~~~~~~~
let n2: i32 = undeclared->foo;

// CHECK:STDOUT: --- fail_deref_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
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
// CHECK:STDOUT:     .n = @__global_init.%n
// CHECK:STDOUT:     .n2 = @__global_init.%n2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc15_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%.loc15_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_8.2: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_8.3: type = converted %int.make_type_signed.loc15, %.loc15_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_9.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc19: init type = call constants.%Int(%.loc19_9.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_9.2: type = value_of_initializer %int.make_type_signed.loc19 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_9.3: type = converted %int.make_type_signed.loc19, %.loc19_9.2 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %undeclared.ref.loc15: <error> = name_ref undeclared, <error> [template = <error>]
// CHECK:STDOUT:   %.loc15: ref <error> = deref <error>
// CHECK:STDOUT:   %n: %i32 = bind_name n, <error>
// CHECK:STDOUT:   %undeclared.ref.loc19: <error> = name_ref undeclared, <error> [template = <error>]
// CHECK:STDOUT:   %.loc19: ref <error> = deref <error>
// CHECK:STDOUT:   %foo.ref: <error> = name_ref foo, <error> [template = <error>]
// CHECK:STDOUT:   %n2: %i32 = bind_name n2, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
