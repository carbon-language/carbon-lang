// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/let/fail_generic_ref.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/let/fail_generic_ref.carbon

// CHECK:STDERR: fail_generic_ref.carbon:[[@LINE+2]]:13-15: error: `ref` is only allowed on a runtime binding [ExpectedRuntimeBindingPatternAfterRef]
// CHECK:STDERR:
let generic ref X: i32 = 0;

// CHECK:STDOUT: - filename: fail_generic_ref.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameNotBeforeSignature 'X'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPatternTypeStart ':' has_error
// CHECK:STDOUT:   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ ├─CompileTimeBindingPattern ':' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ ├─IntLiteral '0'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
