// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/match/fail_missing_cases_block.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/match/fail_missing_cases_block.carbon

fn f() -> i32 {
  match (1)
  // CHECK:STDERR: fail_missing_cases_block.carbon:[[@LINE+4]]:3: error: expected `{` starting block with cases [ExpectedMatchCasesBlock]
  // CHECK:STDERR:   return 0;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  return 0;
}

// CHECK:STDOUT: - filename: fail_missing_cases_block.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'f'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │   │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   ├─ReturnType '->'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─MatchIntroducer 'match'
// CHECK:STDOUT:   │ │   │ ╭─MatchConditionStart '('
// CHECK:STDOUT:   │ │   │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ │   ├─MatchCondition ')'
// CHECK:STDOUT:   │ │ ╭─MatchStatementStart 'return' ❌
// CHECK:STDOUT:   │ ├─MatchStatement 'return' ❌
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
