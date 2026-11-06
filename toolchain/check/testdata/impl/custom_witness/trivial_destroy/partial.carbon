// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/partial.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/partial.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

class ImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class AlsoImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- todo_trivial_empty_partial_class.carbon

library "[[@TEST_NAME]]";

base class Base {}
fn Make() -> Base;

fn Test() {
  partial Base as Core.Destroy;
  partial Base as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: partial Base = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_nonempty_partial_class.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}

base class Base {
  var a: A;
  var b: B;
}

fn Make() -> partial Base;

fn Test() {
  partial Base as Core.Destroy;
  partial Base as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: partial Base = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_member.carbon

library "[[@TEST_NAME]]";

import library "types";

base class Base {
  var x: ImplsDestroy;
  var y: AlsoImplsDestroy;
}

fn Make() -> Base;

fn Test() {
  partial Base as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_impls_destroy.carbon

library "[[@TEST_NAME]]";

base class Base {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn Test() {
  partial Base as Core.TrivialDestroy;
}

// --- todo_trivial_abstract.carbon

library "[[@TEST_NAME]]";

abstract class Abstract {}
fn Make() -> partial Abstract;

fn Test() {
  partial Abstract as Core.Destroy;
  partial Abstract as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: partial Abstract = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_abstract.carbon

library "[[@TEST_NAME]]";

import library "types";

abstract class Abstract {
  var x: ImplsDestroy;
  var y: AlsoImplsDestroy;
}

fn Test() {
  partial Abstract as Core.TrivialDestroy;
}
