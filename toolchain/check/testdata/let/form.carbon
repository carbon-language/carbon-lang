// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/form.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/form.carbon

// --- fail_todo_form_binding.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_form_binding.carbon:[[@LINE+5]]:9-10: error: cannot implicitly convert expression of type `()` to `Core.Form` [ConversionFailure]
// CHECK:STDERR: fail_todo_form_binding.carbon:[[@LINE+4]]:9-10: note: type `()` does not implement interface `Core.ImplicitAs(Core.Form)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_form_binding.carbon:[[@LINE+2]]:5-10: error: semantics TODO: `support local form bindings` [SemanticsTodo]
// CHECK:STDERR:
let x:? () = ();
