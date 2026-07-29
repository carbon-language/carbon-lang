// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/namespace/fail_incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/namespace/fail_incomplete.carbon

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+4]]:11: error: `namespace` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: namespace 123
// CHECK:STDERR:           ^~~
// CHECK:STDERR:
namespace 123

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+4]]:11: error: `namespace` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: namespace ""
// CHECK:STDERR:           ^~
// CHECK:STDERR:
namespace ""

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+4]]:11: error: `namespace` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: namespace +
// CHECK:STDERR:           ^
// CHECK:STDERR:
namespace +

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+4]]:11: error: `namespace` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: namespace bool
// CHECK:STDERR:           ^~~~
// CHECK:STDERR:
namespace bool

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+4]]:11: error: `namespace` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: namespace namespace namespace namespace namespace
// CHECK:STDERR:           ^~~~~~~~~
// CHECK:STDERR:
namespace namespace namespace namespace namespace

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+4]]:1: error: `namespace` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: namespace
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
namespace

// CHECK:STDOUT: - filename: fail_incomplete.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─NamespaceStart 'namespace'
// CHECK:STDOUT:   │ ├─InvalidParse '123' ❌
// CHECK:STDOUT:   ├─Namespace '123' ❌
// CHECK:STDOUT:   │ ╭─NamespaceStart 'namespace'
// CHECK:STDOUT:   │ ├─InvalidParse '""' ❌
// CHECK:STDOUT:   ├─Namespace '""' ❌
// CHECK:STDOUT:   │ ╭─NamespaceStart 'namespace'
// CHECK:STDOUT:   │ ├─InvalidParse '+' ❌
// CHECK:STDOUT:   ├─Namespace '+' ❌
// CHECK:STDOUT:   │ ╭─NamespaceStart 'namespace'
// CHECK:STDOUT:   │ ├─InvalidParse 'bool' ❌
// CHECK:STDOUT:   ├─Namespace 'bool' ❌
// CHECK:STDOUT:   │ ╭─NamespaceStart 'namespace'
// CHECK:STDOUT:   │ ├─InvalidParse 'namespace' ❌
// CHECK:STDOUT:   ├─Namespace 'namespace' ❌
// CHECK:STDOUT:   │ ╭─NamespaceStart 'namespace'
// CHECK:STDOUT:   │ ├─InvalidParse '' ❌
// CHECK:STDOUT:   ├─Namespace 'namespace' ❌
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
