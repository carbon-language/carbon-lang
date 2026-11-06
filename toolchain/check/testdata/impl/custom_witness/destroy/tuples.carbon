// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destroy/tuples.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destroy/tuples.carbon

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

// --- trivial_empty.carbon

library "[[@TEST_NAME]]";

fn Test() {
  var unused x: () = ();
}

// --- trivial_nonempty.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  var unused x: (A, B) = ({}, {});
}

// --- todo_nontrivial_element_first.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  //@dump-sem-ir-begin
  var unused x: (ImplsDestroy, HasTrivialDestruction) = ({}, {});
  //@dump-sem-ir-end
}

// --- todo_nontrivial_element_second.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  //@dump-sem-ir-begin
  var unused x: (HasTrivialDestruction, ImplsDestroy) = ({}, {});
  //@dump-sem-ir-end
}

// --- todo_nontrivial_all_nontrivial_elements.carbon

library "[[@TEST_NAME]]";

import library "types";

class AlsoImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused ref self) {}
  }
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: (ImplsDestroy, AlsoImplsDestroy, AlsoImplsDestroy, ImplsDestroy) = ({}, {}, {}, {});
  //@dump-sem-ir-end
}

// --- trivial_nested.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  var unused x: (A, (B, A)) = ({}, ({}, {}));
}

// --- todo_nontrivial_nested.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  //@dump-sem-ir-begin
  var unused x: ((ImplsDestroy,),) = (({},),);
  //@dump-sem-ir-end
}
