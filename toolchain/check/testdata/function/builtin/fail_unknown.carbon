// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/fail_unknown.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/fail_unknown.carbon

// CHECK:STDERR: fail_unknown.carbon:[[@LINE+4]]:23: error: unknown builtin function name "unknown.builtin.name" [UnknownBuiltinFunctionName]
// CHECK:STDERR: fn UnknownBuiltin() = "unknown.builtin.name";
// CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn UnknownBuiltin() = "unknown.builtin.name";

// CHECK:STDOUT: --- fail_unknown.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UnknownBuiltin.type: type = fn_type @UnknownBuiltin [concrete]
// CHECK:STDOUT:   %UnknownBuiltin: %UnknownBuiltin.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .UnknownBuiltin = %UnknownBuiltin.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %UnknownBuiltin.decl: %UnknownBuiltin.type = fn_decl @UnknownBuiltin [concrete = constants.%UnknownBuiltin] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UnknownBuiltin();
// CHECK:STDOUT:
