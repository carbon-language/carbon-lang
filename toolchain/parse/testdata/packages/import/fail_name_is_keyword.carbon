// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/import/fail_name_is_keyword.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/import/fail_name_is_keyword.carbon

// CHECK:STDERR: fail_name_is_keyword.carbon:[[@LINE+4]]:8: error: expected identifier or `library` after `import` [ExpectedIdentifierAfterImport]
// CHECK:STDERR: import fn;
// CHECK:STDERR:        ^~
// CHECK:STDERR:
import fn;

// CHECK:STDOUT: - filename: fail_name_is_keyword.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   ├─ImportDecl ';' ❌
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
