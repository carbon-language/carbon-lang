// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/tuple/access/repeated.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/tuple/access/repeated.carbon

let x: i32 = tuple.0.1.2;

// CHECK:STDOUT: - filename: repeated.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'tuple'
// CHECK:STDOUT:   │ │     ├─IntLiteral '0'
// CHECK:STDOUT:   │ │   ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │   ├─IntLiteral '1'
// CHECK:STDOUT:   │ │ ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │ ├─IntLiteral '2'
// CHECK:STDOUT:   │ ├─MemberAccessExpr '.'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
