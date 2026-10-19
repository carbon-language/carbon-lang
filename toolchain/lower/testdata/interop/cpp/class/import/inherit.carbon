// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/import/inherit.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/import/inherit.carbon

// --- generic_class_inherit_from_cpp.carbon
library "[[@TEST_NAME]]";
import Cpp;

inline Cpp '''
struct Base {
  virtual void f() {}
  int field;
};
''';

// CHECK:STDERR: generic_class_inherit_from_cpp.carbon:[[@LINE+2]]:1-18: error: semantics TODO: `generic class deriving from C++ virtual class` [SemanticsTodo]
// CHECK:STDERR:
class C(T: type) {
  extend base: Cpp.Base;
}

fn Run() -> i32 {
  var c: C(i32) = {
   .base = Cpp.Base.Base(),
  };
  c.f();
  return c.field;
}

