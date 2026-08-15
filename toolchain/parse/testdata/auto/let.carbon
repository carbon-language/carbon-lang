// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/auto/let.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/auto/let.carbon

let x: auto = 1;

// CHECK:STDOUT: - filename: let.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ ├─AutoTypeLiteral 'auto'
// CHECK:STDOUT:   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ ├─IntLiteral '1'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
