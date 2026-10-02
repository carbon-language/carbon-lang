// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/import/fail_no_semi.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/import/fail_no_semi.carbon

import Geometry

// CHECK:STDERR: fail_no_semi.carbon:[[@LINE+8]]:26: error: `import` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR:
// CHECK:STDOUT: - filename: fail_no_semi.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Geometry'
// CHECK:STDOUT:   ├─ImportDecl 'Geometry' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
