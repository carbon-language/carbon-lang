// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The missing `;` goes just past the closing `'''`.
inline Cpp '''
int n;
'''

// The CHECK lines go below, because a diagnostic on the closing `'''` line
// otherwise puts them inside the block string.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/import/fail_inline_cpp_no_semi.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/import/fail_inline_cpp_no_semi.carbon
// CHECK:STDERR: fail_inline_cpp_no_semi.carbon:[[@LINE-10]]:4: error: `inline` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR:

// CHECK:STDOUT: - filename: fail_inline_cpp_no_semi.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─InlineIntroducer 'inline'
// CHECK:STDOUT:   │ ├─CppNameExpr 'Cpp'
// CHECK:STDOUT:   │ ├─InlineImportBody ''''
// CHECK:STDOUT: int n;
// CHECK:STDOUT: ''''
// CHECK:STDOUT:   ├─InlineCppDecl ''''
// CHECK:STDOUT: int n;
// CHECK:STDOUT: '''' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
