// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/named_constraint/fail_incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/named_constraint/fail_incomplete.carbon

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+2]]:12-16: error: `constraint` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR:
constraint "Foo"

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+2]]:1-10: error: `constraint` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR:
constraint

// CHECK:STDOUT: - filename: fail_incomplete.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─NamedConstraintIntroducer 'constraint'
// CHECK:STDOUT:   │ ├─InvalidParse '"Foo"' has_error
// CHECK:STDOUT:   ├─NamedConstraintDecl '"Foo"' has_error
// CHECK:STDOUT:   │ ╭─NamedConstraintIntroducer 'constraint'
// CHECK:STDOUT:   │ ├─InvalidParse '' has_error
// CHECK:STDOUT:   ├─NamedConstraintDecl 'constraint' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
