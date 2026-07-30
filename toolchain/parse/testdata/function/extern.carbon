// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/function/extern.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/function/extern.carbon

// --- extern.carbon

extern fn G();

// --- library.carbon

extern library "foo" fn G();

// --- default_library.carbon

extern library default fn G();

// --- fail_name_missing.carbon

// CHECK:STDERR: fail_name_missing.carbon:[[@LINE+4]]:16: error: expected `default` or a string literal to specify the library name [ExpectedLibraryNameOrDefault]
// CHECK:STDERR: extern library fn G();
// CHECK:STDERR:                ^~
// CHECK:STDERR:
extern library fn G();

// --- fail_identifier.carbon

// CHECK:STDERR: fail_identifier.carbon:[[@LINE+8]]:16: error: expected `default` or a string literal to specify the library name [ExpectedLibraryNameOrDefault]
// CHECK:STDERR: extern library foo fn G();
// CHECK:STDERR:                ^~~
// CHECK:STDERR:
// CHECK:STDERR: fail_identifier.carbon:[[@LINE+4]]:16: error: unrecognized declaration introducer [UnrecognizedDecl]
// CHECK:STDERR: extern library foo fn G();
// CHECK:STDERR:                ^~~
// CHECK:STDERR:
extern library foo fn G();

// --- extern_def.carbon

extern fn G() {}

// --- library_def.carbon

extern library "foo" fn G() {}

// CHECK:STDOUT: - filename: extern.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─ExternModifier 'extern'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'G'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: library.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │   ╭─LibraryName '"foo"'
// CHECK:STDOUT:   │ │ ╭─LibrarySpecifier 'library'
// CHECK:STDOUT:   │ ├─ExternModifierWithLibrary 'extern'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'G'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: default_library.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │   ╭─DefaultLibrary 'default'
// CHECK:STDOUT:   │ │ ╭─LibrarySpecifier 'library'
// CHECK:STDOUT:   │ ├─ExternModifierWithLibrary 'extern'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'G'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_name_missing.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │   ╭─LibraryName 'fn' has_error
// CHECK:STDOUT:   │ │ ╭─LibrarySpecifier 'library'
// CHECK:STDOUT:   │ ├─ExternModifierWithLibrary 'extern'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'G'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_identifier.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─InvalidParseStart 'foo' has_error
// CHECK:STDOUT:   │ │   ╭─LibraryName 'foo' has_error
// CHECK:STDOUT:   │ │ ╭─LibrarySpecifier 'library'
// CHECK:STDOUT:   │ ├─ExternModifierWithLibrary 'extern'
// CHECK:STDOUT:   ├─InvalidParseSubtree ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: extern_def.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─ExternModifier 'extern'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'G'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: library_def.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   │   ╭─LibraryName '"foo"'
// CHECK:STDOUT:   │   │ ╭─LibrarySpecifier 'library'
// CHECK:STDOUT:   │   ├─ExternModifierWithLibrary 'extern'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'G'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
