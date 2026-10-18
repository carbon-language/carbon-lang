// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/fail_multiple_vtables.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/fail_multiple_vtables.carbon

// --- fail_no_override.carbon
library "fail_no_override";

import Cpp;

inline Cpp '''
struct A { virtual void f(); };
struct B { virtual void g(); };
struct C : A, B {};
''';

// CHECK:STDERR: fail_no_override.carbon:[[@LINE+4]]:1: error: semantics TODO: `class deriving from C++ class with multiple virtual tables` [SemanticsTodo]
// CHECK:STDERR: class D {
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
class D {
  extend base: Cpp.C;
}

// --- fail_override.carbon
library "fail_override";

import Cpp;

inline Cpp '''
struct A { virtual void f(); };
struct B { virtual void g(); };
struct C : A, B {};
''';

// CHECK:STDERR: fail_override.carbon:[[@LINE+4]]:1: error: semantics TODO: `class deriving from C++ class with multiple virtual tables` [SemanticsTodo]
// CHECK:STDERR: class D {
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
class D {
  extend base: Cpp.C;
  override fn f(unused self) {}
}
