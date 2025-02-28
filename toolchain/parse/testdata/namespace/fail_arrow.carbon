// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/namespace/fail_arrow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/namespace/fail_arrow.carbon

namespace Foo;

// CHECK:STDERR: fail_arrow.carbon:[[@LINE+4]]:10: error: `class` declarations must either end with a `;` or have a `{ ... }` block for a definition [ExpectedDeclSemiOrDefinition]
// CHECK:STDERR: class Foo->Bar {}
// CHECK:STDERR:          ^~
// CHECK:STDERR:
class Foo->Bar {}

// CHECK:STDOUT: - filename: fail_arrow.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'Namespace', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: '}', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
