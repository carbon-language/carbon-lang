// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/packages/import/fail_extra_string.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/packages/import/fail_extra_string.carbon

// CHECK:STDERR:        ╭─┤ fail_extra_string.carbon:[[@LINE+7]]:26
// CHECK:STDERR: ->     │ import Foo library "bar" "baz";
// CHECK:STDERR:        ·                          ━━┳━━
// CHECK:STDERR:        ·                            ┗━━┫ error: `import` declarations must end with a `;`
// CHECK:STDERR:        ·                               ┃        [ExpectedDeclSemi]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
import Foo library "bar" "baz";

// CHECK:STDOUT: - filename: fail_extra_string.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ImportIntroducer 'import'
// CHECK:STDOUT:   │ ├─IdentifierPackageName 'Foo'
// CHECK:STDOUT:   │ │ ╭─LibraryName '"bar"'
// CHECK:STDOUT:   │ ├─LibrarySpecifier 'library'
// CHECK:STDOUT:   ├─ImportDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
