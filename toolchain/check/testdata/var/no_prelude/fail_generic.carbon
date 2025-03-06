// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/fail_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/no_prelude/fail_generic.carbon

fn Main() {
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+8]]:7: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   var x:! () = ();
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+4]]:14: error: `var` pattern cannot declare a compile-time binding [CompileTimeBindingInVarDecl]
  // CHECK:STDERR:   var x:! () = ();
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  var x:! () = ();
}

// CHECK:STDOUT: --- fail_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main();
// CHECK:STDOUT:
