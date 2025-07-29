// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/unsupported_decl_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/unsupported_decl_type.carbon

// --- template.h

template<typename T> class C {};

// --- fail_todo_use_template.carbon

library "[[@TEST_NAME]]";

import Cpp library "template.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_use_template.carbon:[[@LINE+4]]:10: error: name `C` not found [NameNotFound]
  // CHECK:STDERR:   var c: C({});
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  var c: C({});
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- fail_todo_use_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: <error> = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: <error> = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref <error> = var %c.var_patt [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %c: <error> = bind_name c, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
