// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/structs.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/structs.carbon

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

alias Struct = {};
fn Make() -> Struct;

fn Test() {
  Struct as Core.Destroy;
  Struct as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Struct = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_nonempty.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: A, .b: B};
fn Make() -> Struct;

fn Test() {
  Struct as Core.Destroy;
  Struct as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Struct = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_nested_struct_with_trivial_members.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: {.b: B}};
fn Make() -> Struct;

fn Test() {
  Struct as Core.Destroy;
  Struct as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Struct = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_struct_nontrivial_field_first.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: ImplsDestroy, .b: HasTrivialDestruction};

fn Test() {
  Struct as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_struct_nontrivial_field_second.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: HasTrivialDestruction, .b: ImplsDestroy};

fn Test() {
  Struct as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_struct_nontrivial_field_nested.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: {.b: ImplsDestroy}};

fn Test() {
  Struct as Core.TrivialDestroy;
}
