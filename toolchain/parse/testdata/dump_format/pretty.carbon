// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --parse-dump-format=pretty
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/dump_format/pretty.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/dump_format/pretty.carbon

// --- fail_pretty.carbon

// CHECK:STDERR:        ╭─┤ fail_pretty.carbon:[[@LINE+7]]:9
// CHECK:STDERR: ->     │ var x: {,} = {};
// CHECK:STDERR:        ·         ┳
// CHECK:STDERR:        ·         ┗━━┫ error: expected `.field: field_type` or `.field = value`
// CHECK:STDERR:        ·            ┃        [ExpectedStructLiteralField]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var x: {,} = {};

// CHECK:STDOUT: - filename: fail_pretty.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │   │ ├─InvalidParse ',' has_error
// CHECK:STDOUT:   │ │   │ ├─InvalidParse ',' has_error
// CHECK:STDOUT:   │ │   ├─StructLiteral '}' has_error
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
