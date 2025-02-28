// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/fail_prefix_repeat.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/fail_prefix_repeat.carbon

// CHECK:STDERR: fail_prefix_repeat.carbon:[[@LINE+4]]:17: error: parentheses are required around this unary `const` operator [UnaryOperatorRequiresParentheses]
// CHECK:STDERR: fn F() -> const const i32* {
// CHECK:STDERR:                 ^~~~~
// CHECK:STDERR:
fn F() -> const const i32* {
}

// CHECK:STDOUT: - filename: fail_prefix_repeat.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:               {kind: 'PrefixOperatorConst', text: 'const', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'PrefixOperatorConst', text: 'const', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'PostfixOperatorStar', text: '*', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
