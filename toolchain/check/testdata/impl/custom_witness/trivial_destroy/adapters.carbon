// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/adapters.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/adapters.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

class ImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- todo_trivial_class_adapter.carbon

library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt HasTrivialDestruction;
}

fn Make() -> Adapter;

fn Test() {
  Adapter as Core.Destroy;
  Adapter as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Adapter = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_class_adapter.carbon

library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt ImplsDestroy;
}

fn Make() -> Adapter;

fn Test() {
  Adapter as Core.TrivialDestroy;
}

// --- todo_fail_trivial_destroy_adapter_with_destroy_impl.carbon

library "[[@TEST_NAME]]";


class HasTrivialDestruction {}

class Adapter {
  adapt HasTrivialDestruction;

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn Test() {
  Adapter as Core.TrivialDestroy;
}

// --- todo_trivial_struct_adapter.carbon

library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt {.n: HasTrivialDestruction};
}

fn Make() -> Adapter;

fn Test() {
  Adapter as Core.Destroy;
  Adapter as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Adapter = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_struct_adapter.carbon

library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt {.n: ImplsDestroy};
}

fn Test() {
  Adapter as Core.TrivialDestroy;
}

// --- todo_trivial_pointer.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Pointer = ImplsDestroy*;
fn Make() -> Pointer;

fn Test() {
  Pointer as Core.Destroy;
  Pointer as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Pointer = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_const.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Const = const HasTrivialDestruction;
fn Make() -> Const;

fn Test() {
  Const as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Const = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_const.carbon

library "[[@TEST_NAME]]";

import library "types";

alias Const = const ImplsDestroy;

fn Test() {
  Const as Core.TrivialDestroy;
}
