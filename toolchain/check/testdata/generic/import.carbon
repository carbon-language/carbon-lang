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

// The specific error is not relevant; we just need importing an inst in
// the generic's definition block to fail, in order to test error recovery
// during generic import.

// CHECK:STDERR: fail_use.carbon:[[@LINE+6]]:1: in import [InImport]
// CHECK:STDERR: middle.carbon:3:10: in file included here [InCppInclude]
// CHECK:STDERR: ./leaf.h:2:7: error: semantics TODO: `indirect import of C++ declaration with no direct Cpp import` [SemanticsTodo]
// CHECK:STDERR: class Leaf {};
// CHECK:STDERR:       ^
// CHECK:STDERR:
import library "lib";

alias M = Middle;
