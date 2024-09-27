// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/fail_namespace_conflict.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/no_prelude/fail_namespace_conflict.carbon

namespace A;

// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE+7]]:5: error: duplicate name being declared in the same scope
// CHECK:STDERR: var A: ();
// CHECK:STDERR:     ^
// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE-5]]:1: note: name is previously declared here
// CHECK:STDERR: namespace A;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
var A: ();

// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE+6]]:5: error: duplicate name being declared in the same scope
// CHECK:STDERR: var A: () = ();
// CHECK:STDERR:     ^
// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE-14]]:1: note: name is previously declared here
// CHECK:STDERR: namespace A;
// CHECK:STDERR: ^~~~~~~~~~~~
var A: () = ();

// CHECK:STDOUT: --- fail_namespace_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.loc11
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.loc11: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc20_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc20_9.2: type = converted %.loc20_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %A.var.loc20: ref %.1 = var A
// CHECK:STDOUT:   %A.loc20: ref %.1 = bind_name A, %A.var.loc20
// CHECK:STDOUT:   %.loc28_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc28_9.2: type = converted %.loc28_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %A.var.loc28: ref %.1 = var A
// CHECK:STDOUT:   %A.loc28: ref %.1 = bind_name A, %A.var.loc28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc28_14.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc28_14.2: init %.1 = tuple_init () to file.%A.var.loc28 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc28_15: init %.1 = converted %.loc28_14.1, %.loc28_14.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%A.var.loc28, %.loc28_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
