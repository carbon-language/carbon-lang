// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/tuples.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/tuples.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}

// Used when mixing with `ImplsDestroy`.
class HasTrivialDestruction {}

class ImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- todo_trivial_empty.carbon

library "[[@TEST_NAME]]";

alias Tuple = ();
fn Make() -> Tuple;

fn Test() {
  Tuple as Core.Destroy;
  Tuple as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Tuple = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_nonempty.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Tuple = (A, B);
fn Make() -> Tuple;

fn Test() {
  Tuple as Core.Destroy;
  Tuple as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Tuple = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_element_first.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Tuple = (ImplsDestroy, HasTrivialDestruction);

fn Test() {
  Tuple as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_element_second.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Tuple = (HasTrivialDestruction, ImplsDestroy);

fn Test() {
  Tuple as Core.TrivialDestroy;
}

// --- todo_trivial_nested.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Tuple = (A, (B, A));
fn Make() -> Tuple;

fn Test() {
  Tuple as Core.Destroy;
  Tuple as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Tuple = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_nested.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Tuple = ((ImplsDestroy,),);

fn Test() {
  Tuple as Core.TrivialDestroy;
}
