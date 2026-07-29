// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/named_constraint/fail_incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/named_constraint/fail_incomplete.carbon

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+4]]:12: error: `constraint` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: constraint "Foo"
// CHECK:STDERR:            ^~~~~
// CHECK:STDERR:
constraint "Foo"

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+4]]:1: error: `constraint` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: constraint
// CHECK:STDERR: ^~~~~~~~~~
// CHECK:STDERR:
constraint

// CHECK:STDOUT: - filename: fail_incomplete.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─NamedConstraintIntroducer 'constraint'
// CHECK:STDOUT:   │ ├─InvalidParse '"Foo"' ❌
// CHECK:STDOUT:   ├─NamedConstraintDecl '"Foo"' ❌
// CHECK:STDOUT:   │ ╭─NamedConstraintIntroducer 'constraint'
// CHECK:STDOUT:   │ ├─InvalidParse '' ❌
// CHECK:STDOUT:   ├─NamedConstraintDecl 'constraint' ❌
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
