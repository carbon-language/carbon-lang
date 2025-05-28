// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/fail_in_interface.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/no_prelude/fail_in_interface.carbon

interface I {
  // CHECK:STDERR: fail_in_interface.carbon:[[@LINE+4]]:11: error: found runtime binding pattern in associated constant declaration; expected a `:!` binding [ExpectedSymbolicBindingInAssociatedConstant]
  // CHECK:STDERR:   let var a: ();
  // CHECK:STDERR:           ^~~~~
  // CHECK:STDERR:
  let var a: ();

  // CHECK:STDERR: fail_in_interface.carbon:[[@LINE+4]]:7: error: found runtime binding pattern in associated constant declaration; expected a `:!` binding [ExpectedSymbolicBindingInAssociatedConstant]
  // CHECK:STDERR:   var b: ();
  // CHECK:STDERR:       ^~~~~
  // CHECK:STDERR:
  var b: ();

  // CHECK:STDERR: fail_in_interface.carbon:[[@LINE+8]]:11: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   let var c:! ();
  // CHECK:STDERR:           ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_in_interface.carbon:[[@LINE+4]]:17: error: `var` pattern cannot declare a compile-time binding [CompileTimeBindingInVarDecl]
  // CHECK:STDERR:   let var c:! ();
  // CHECK:STDERR:                 ^
  // CHECK:STDERR:
  let var c:! ();

  // CHECK:STDERR: fail_in_interface.carbon:[[@LINE+4]]:13: error: `var` pattern cannot declare a compile-time binding [CompileTimeBindingInVarDecl]
  // CHECK:STDERR:   var d:! ();
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  var d:! ();
}
