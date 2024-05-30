// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/global_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/global_decl.carbon

var x: {.v: ()};

// CHECK:STDOUT: --- global_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.v: ()} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_14.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc11_14.2: type = converted %.loc11_14.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_15: type = struct_type {.v: ()} [template = constants.%.2]
// CHECK:STDOUT:   %x.var: ref {.v: ()} = var x
// CHECK:STDOUT:   %x: ref {.v: ()} = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
