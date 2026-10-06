// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/impl/declaration.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/impl/declaration.carbon

impl bool as Interface;

// CHECK:STDOUT: - filename: declaration.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ │ ╭─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │ ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'Interface'
// CHECK:STDOUT:   ├─ImplDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
