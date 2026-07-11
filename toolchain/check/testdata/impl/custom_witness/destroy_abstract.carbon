// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destroy_abstract.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destroy_abstract.carbon

abstract class AbstractBase {}

class Derived {
  extend base: AbstractBase;
}

// @dump-sem-ir-begin
fn Run() {
  var _: Derived = {.base = {}};
}
// @dump-sem-ir-end

fn F(T:! Core.Destroy) {}

fn G() {
  F(AbstractBase);
}
