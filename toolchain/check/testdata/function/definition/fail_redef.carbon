// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/definition/fail_redef.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/definition/fail_redef.carbon

fn F() {}
// CHECK:STDERR: fail_redef.carbon:[[@LINE+7]]:1: error: redefinition of `fn F` [RedeclRedef]
// CHECK:STDERR: fn F() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_redef.carbon:[[@LINE-4]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: fn F() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
fn F() {}

// CHECK:STDOUT: --- fail_redef.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type.b25846.1: type = fn_type @F.loc15 [concrete]
// CHECK:STDOUT:   %F.c41931.1: %F.type.b25846.1 = struct_value () [concrete]
// CHECK:STDOUT:   %F.type.b25846.2: type = fn_type @F.loc23 [concrete]
// CHECK:STDOUT:   %F.c41931.2: %F.type.b25846.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl.loc15
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc15: %F.type.b25846.1 = fn_decl @F.loc15 [concrete = constants.%F.c41931.1] {} {}
// CHECK:STDOUT:   %F.decl.loc23: %F.type.b25846.2 = fn_decl @F.loc23 [concrete = constants.%F.c41931.2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc15() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc23() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
