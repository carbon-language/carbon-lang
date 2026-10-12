// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/import/custom_layout.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/import/custom_layout.carbon

// --- struct.h

struct Struct {
  int a;
  int b;
};

// --- a.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct.h";

class Adapter {
  adapt Cpp.Struct;
}

// --- b.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct.h";
import library "a";

fn F(unused a: Adapter) {}
