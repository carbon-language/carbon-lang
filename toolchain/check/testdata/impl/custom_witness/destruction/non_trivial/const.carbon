// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/non_trivial/const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/non_trivial/const.carbon

// --- types.carbon
library "[[@TEST_NAME]]";

// `HasNonTrivialDestruction.(SubobjectDestroy.Op)` is a no-op because it
// doesn't have any fields that require it to be non-trivial.
// Similarly for other types like this.
class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

base class NonTrivialDynamicClass {
  virtual fn F(self);

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- todo_nontrivial_const_element.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Tuple = (const HasNonTrivialDestruction, HasNonTrivialDestruction);
fn Make() -> Tuple;

fn Test() {
  //@dump-sem-ir-begin
  Tuple as Core.SubobjectDestroy;
  Tuple as Core.Destroy;

  var unused x: Tuple = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_const_adapter.carbon
library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt HasNonTrivialDestruction;
}

alias Const = const Adapter;
fn Make() -> Const;

fn Test() {
  //@dump-sem-ir-begin
  Const as Core.SubobjectDestroy;
  Const as Core.Destroy;

  var unused x: Const = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_const_partial.carbon
library "[[@TEST_NAME]]";

import library "types";

base class Base {
  var x: HasNonTrivialDestruction;
}

alias Const = const (partial Base);
fn Make() -> Const;

fn Test() {
  //@dump-sem-ir-begin
  Const as Core.SubobjectDestroy;
  Const as Core.Destroy;

  var unused x: Const = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_const_dynamic_class.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Class = const NonTrivialDynamicClass;
fn Make() -> Class;

fn Test() {
  //@dump-sem-ir-begin
  Class as Core.SubobjectDestroy;
  Class as Core.Destroy;

  var unused x: Class = Make();
  //@dump-sem-ir-end
}
