// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destroy/arrays.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destroy/arrays.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- trivial_empty_array_with_trivial_element_type.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  var unused x: array(i32, 0);
}

// --- trivial_empty_array_with_nontrivial_element_type.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  var unused x: array(HasNonTrivialDestruction, 0);
}

// --- trivial_populated.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  var unused x: array(HasTrivialDestruction, 1);
}

// --- todo_nontrivial_populated.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  //@dump-sem-ir-begin
  var unused x: array(HasNonTrivialDestruction, 2);
  //@dump-sem-ir-end
}

// --- trivial_array_of_array.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  var unused x: array(array(HasTrivialDestruction, 3), 10);
}

// --- todo_nontrivial_array_of_array.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  //@dump-sem-ir-begin
  var unused x: array(array(HasNonTrivialDestruction, 4), 43);
  //@dump-sem-ir-end
}

// --- trivial_adapter.carbon

// Not in `adapters.carbon`, since that would expand our min_prelude from `destroy.carbon` to
// `array.carbon`.

library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt array(HasTrivialDestruction, 2);
}

fn Test() {
  var unused x: Adapter;
}

// --- todo_nontrivial_adapter.carbon

// Not in `adapters.carbon`, since that would expand our min_prelude from `destroy.carbon` to
// `array.carbon`.

library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt array(HasNonTrivialDestruction, 2);
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: Adapter;
  //@dump-sem-ir-end
}

// --- trivial_constrained_facet.carbon

// Not in `facets.carbon`, since that would expand our min_prelude from `destroy.carbon` to
// `array.carbon`.

library "[[@TEST_NAME]]";

fn Test(generic T: Core.Destroy) {
  var unused x: array(T, 2);
}