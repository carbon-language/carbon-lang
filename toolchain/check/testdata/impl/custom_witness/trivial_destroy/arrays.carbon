// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/arrays.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/arrays.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- todo_trivial_empty_array_with_trivial_element_type.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Array = array(i32, 0);
fn Make() -> Array;

fn Test() {
  Array as Core.Destroy;
  Array as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Array = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_empty_array_with_nontrivial_element_type.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Array =  array(HasNonTrivialDestruction, 0);
fn Make() -> Array;

fn Test() {
  Array as Core.Destroy;
  Array as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Array = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_populated.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Array = array(HasTrivialDestruction, 1);
fn Make() -> Array;

fn Test() {
  Array as Core.Destroy;
  Array as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Array = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_populated.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Array = array(HasNonTrivialDestruction, 2);

fn Test() {
  Array as Core.TrivialDestroy;
}

// --- todo_trivial_array_of_array.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Array = array(array(HasTrivialDestruction, 3), 10);
fn Make() -> Array;

fn Test() {
  Array as Core.Destroy;
  Array as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Array = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_array_of_array.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Array = array(array(HasNonTrivialDestruction, 4), 43);

fn Test() {
  Array as Core.TrivialDestroy;
}

// --- todo_trivial_adapter.carbon

// Not in `adapters.carbon`, since that would expand our min_prelude from `destroy.carbon` to
// `array.carbon`.

library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt array(HasTrivialDestruction, 2);
}

fn Make() -> Adapter;

fn Test() {
  Adapter as Core.Destroy;
  Adapter as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Adapter = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_adapter.carbon

library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt array(HasNonTrivialDestruction, 2);
}

fn Test() {
  Adapter as Core.TrivialDestroy;
}

// --- todo_trivial_constrained_facet.carbon

// Not in `facets.carbon`, since that would expand our min_prelude from `destroy.carbon` to
// `array.carbon`.

library "[[@TEST_NAME]]";

fn MakeElement(generic T: Core.TrivialDestroy) -> T {
  returned var result: T;
  return var;
}

fn MakeArray(generic T: Core.TrivialDestroy) -> array(T, 3) {
  return (MakeElement(T), MakeElement(T), MakeElement(T));
}

fn Test(generic T: Core.TrivialDestroy) {
  array(T, 3) as Core.Destroy;
  array(T, 3) as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: array(T, 3) = MakeArray(T);
  //@dump-sem-ir-end
}
