// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/form.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/fail_todo_form_thunk.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/fail_todo_form_thunk.carbon

interface I {
  let T:! type;
  let Fm:! Core.Form();
  // CHECK:STDERR: fail_todo_form_thunk.carbon:[[@LINE+4]]:8: error: semantics TODO: `Support symbolic form bindings` [SemanticsTodo]
  // CHECK:STDERR:   fn F(x:? Fm, y: T);
  // CHECK:STDERR:        ^~~~~~
  // CHECK:STDERR:
  fn F(x:? Fm, y: T);
}

class C {
  impl as I where .T = () and .Fm = form(ref ()) {
    fn F(x:? form(ref ()), ());
  }
}
