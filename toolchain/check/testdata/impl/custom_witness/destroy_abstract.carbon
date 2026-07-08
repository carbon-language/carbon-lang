// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destroy_abstract.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destroy_abstract.carbon

library "[[@TEST_NAME]]";

abstract class Otter {
  var whisker_count: i32;
  protected var fur_status: i32;
}

class C {
  extend base: AbstractBase;
}

fn Run() {
  var _: RiverOtter = {.base = {.whisker_count = 30, .fur_status = 1}};
}
