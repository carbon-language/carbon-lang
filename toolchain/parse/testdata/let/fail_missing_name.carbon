// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/let/fail_missing_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/let/fail_missing_name.carbon

// --- fail_runtime_binding.carbon

// CHECK:STDERR: fail_runtime_binding.carbon:[[@LINE+2]]:5: error: expected pattern [ExpectedPattern]
// CHECK:STDERR:
let : i32 = 4;

// --- fail_compiletime_binding.carbon

// CHECK:STDERR: fail_compiletime_binding.carbon:[[@LINE+2]]:13: error: expected name in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR:
let generic : bool = true;

// CHECK:STDOUT: - filename: fail_runtime_binding.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ ├─InvalidParse ':' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ ├─IntLiteral '4'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_compiletime_binding.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameNotBeforeSignature ':' has_error
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPatternTypeStart ':' has_error
// CHECK:STDOUT:   │ │ ├─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │ ├─CompileTimeBindingPattern ':' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ ├─BoolLiteralTrue 'true'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
