// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/primitive_types/void.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/primitive_types/void.carbon

// --- void_not_incomplete.carbon

library "[[@TEST_NAME]]";

import Cpp;

// This is valid, unlike in C++, because `Cpp.void` is complete but abstract in
// Carbon. Values of type `Cpp.void` exist, but objects of that type do not.
fn F(unused x: Cpp.void) {}

// --- fail_void_abstract.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn G() {
  // CHECK:STDERR: fail_void_abstract.carbon:[[@LINE+3]]:17-24: error: binding pattern has abstract type `Cpp.void` in `var` pattern [AbstractTypeInVarPattern]
  // CHECK:STDERR: {{.*}}/prelude/types/cpp/void.carbon:24:1-35: note: class was declared abstract here [ClassAbstractHere]
  // CHECK:STDERR:
  var unused x: Cpp.void;
}

// --- fail_void_argument.carbon

library "[[@TEST_NAME]]";

import Cpp inline "void F(int);";

fn G(v: Cpp.void) {
  // CHECK:STDERR: fail_void_argument.carbon:[[@LINE+4]]:3-7: error: no matching function for call to 'F' [CppInteropParseError]
  // CHECK:STDERR: fail_void_argument.carbon:[[@LINE-4]]:6: note: candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR: fail_void_argument.carbon:[[@LINE-5]]:8-10: note: cannot convert argument of incomplete type 'void' to 'int' for 1st argument [CppInteropParseNoteReason]
  // CHECK:STDERR:
  Cpp.F(v);
}

fn H(p: Cpp.void*) {
  // CHECK:STDERR: fail_void_argument.carbon:[[@LINE+4]]:3-7: error: no matching function for call to 'F' [CppInteropParseError]
  // CHECK:STDERR: fail_void_argument.carbon:[[@LINE-12]]:6: note: candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR: fail_void_argument.carbon:[[@LINE-13]]:8-10: note: cannot convert argument of incomplete type 'void' to 'int' for 1st argument [CppInteropParseNoteReason]
  // CHECK:STDERR:
  Cpp.F(*p);
}
