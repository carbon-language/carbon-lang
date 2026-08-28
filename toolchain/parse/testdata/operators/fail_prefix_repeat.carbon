// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/fail_prefix_repeat.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/fail_prefix_repeat.carbon

// CHECK:STDERR: fail_prefix_repeat.carbon:[[@LINE+2]]:17-21: error: parentheses are required around this unary `const` operator [UnaryOperatorRequiresParentheses]
// CHECK:STDERR:
fn F() -> const const i32* {
}

// CHECK:STDOUT: - filename: fail_prefix_repeat.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │   │       ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   │     ╭─PrefixOperatorConst 'const'
// CHECK:STDOUT:   │   │   ╭─PrefixOperatorConst 'const'
// CHECK:STDOUT:   │   │ ╭─PostfixOperatorStar '*'
// CHECK:STDOUT:   │   ├─ReturnType '->'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
