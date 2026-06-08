// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/import/dynamic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/import/dynamic.carbon

// --- dynamic_base_from_cpp.carbon

library "[[@TEST_NAME]]";

import Cpp;

inline Cpp '''

struct FurtherBase {
  virtual auto further_base_func() -> void;
};

struct Base: FurtherBase {
  int i;
  virtual auto func() & -> void;
};

__attribute__((optnone)) auto Use(Base& b) {
}

''';

base class Derived {
  extend base: Cpp.Base;
  virtual fn other_func[unused self: Self]() {
  }
  override fn func[unused ref self: Self]() {
  }
}

fn DoThing() {
  // CHECK:STDERR: dynamic_base_from_cpp.carbon:[[@LINE+4]]:20: error: semantics TODO: `accessing vptr of custom layout class` [SemanticsTodo]
  // CHECK:STDERR:   var d: Derived = {.base = Cpp.Base.Base()};
  // CHECK:STDERR:                    ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var d: Derived = {.base = Cpp.Base.Base()};
  Cpp.Use(ref d);
}
