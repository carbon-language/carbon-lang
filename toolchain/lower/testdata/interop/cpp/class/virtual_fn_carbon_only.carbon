// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// EXTRA-ARGS: --clang-arg=-fno-exceptions
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/virtual_fn_carbon_only.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/virtual_fn_carbon_only.carbon

// --- non_generic.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
struct X { virtual void f() {} };
''';

class Z {
  extend base: Cpp.X;
  override fn f(unused self) {}
}

fn F() {
  var _: Z = {.base = Cpp.X.X()};
}

// --- generic.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
struct X { virtual void f() {} };
''';

class GenericZ(T: type) {
  extend base: Cpp.X;
  override fn f(unused self) {}
}

fn F() {
  var _: GenericZ(type) = {.base = Cpp.X.X()};
}
