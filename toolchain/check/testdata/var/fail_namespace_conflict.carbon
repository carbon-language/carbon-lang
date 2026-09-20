// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/fail_namespace_conflict.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/fail_namespace_conflict.carbon

namespace A;

// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE+3]]:5: error: duplicate name `A` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE-3]]:1-12: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:
var A: ();

// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE+3]]:5: error: duplicate name `A` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fail_namespace_conflict.carbon:[[@LINE-8]]:1-12: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:
var A: () = ();
