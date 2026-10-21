// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/import.carbon

// --- leaf.h

class Leaf {};

// --- middle.carbon
library "lib";

import Cpp library "leaf.h";

class Middle(T: type) {
  var l: Cpp.Leaf;
}

// --- fail_use.carbon

// This tests a situation in which an error occurs while importing a member of
// a generic's definition block. That shouldn't be possible: if the imported
// generic is valid in its source file, importing it shouldn't result in any
// errors. However, this TODO (and at least one other TODO as of this writing)
// can be triggered when importing locally-valid Carbon code. This is a
// regression test to validate that we don't crash when that happens.
// Consequently, this test can be removed if this TODO is resolved and there are
// no other ways of triggering an error while importing a valid generic.

// CHECK:STDERR: fail_use.carbon:[[@LINE+6]]:1: in import [InImport]
// CHECK:STDERR: middle.carbon:3:10: in file included here [InCppInclude]
// CHECK:STDERR: ./leaf.h:2:7: error: semantics TODO: `indirect import of C++ declaration with no direct Cpp import` [SemanticsTodo]
// CHECK:STDERR: class Leaf {};
// CHECK:STDERR:       ^
// CHECK:STDERR:
import library "lib";

alias M = Middle;
