// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/unformed.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destroy/adapters.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destroy/adapters.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

class ImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- trivial_class_adapter.carbon

library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt HasTrivialDestruction;
}

fn Test() {
  var unused x: Adapter;
}

// --- todo_nontrivial_class_adapter.carbon

library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt ImplsDestroy;
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: Adapter;
  //@dump-sem-ir-end
}

// --- trivial_destroy_adapter_with_destroy_impl.carbon

library "[[@TEST_NAME]]";


class HasTrivialDestruction {}

class Adapter {
  adapt HasTrivialDestruction;

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn Test() {
  var unused x: Adapter;
}

// --- trivial_struct_adapter.carbon

library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt {.x: HasTrivialDestruction};
}

fn Test() {
  var unused x: Adapter;
}

// --- todo_nontrivial_struct_adapter.carbon

library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt {.x: ImplsDestroy};
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: Adapter;
  //@dump-sem-ir-end
}

// --- trivial_tuple_adapter.carbon

library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt (HasTrivialDestruction, HasTrivialDestruction);
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: Adapter;
  //@dump-sem-ir-end
}

// --- todo_nontrivial_nontrivial_tuple_adapter.carbon

library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt (ImplsDestroy, HasTrivialDestruction);
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: Adapter;
  //@dump-sem-ir-end
}

// --- trivial_pointer.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Make() -> ImplsDestroy*;

fn Test() {
  var unused x: ImplsDestroy*;
}

// --- trivial_const.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  var unused x: const HasTrivialDestruction = {};
}

// --- todo_nontrivial_const.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  //@dump-sem-ir-begin
  var unused x: const ImplsDestroy = {};
  //@dump-sem-ir-end
}

// --- trivial_maybe_unformed.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  var unused x: Core.MaybeUnformed(HasTrivialDestruction);
}

// --- todo_nontrivial_maybe_unformed.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  //@dump-sem-ir-begin
  var unused x: Core.MaybeUnformed(ImplsDestroy);
  //@dump-sem-ir-end
}

