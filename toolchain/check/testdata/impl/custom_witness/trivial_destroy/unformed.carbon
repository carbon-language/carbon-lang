// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/unformed.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/adapters.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/adapters.carbon

// --- todo_trivial_maybe_unformed.carbon

library "[[@TEST_NAME]]";

import library "types";

class HasTrivialDestruction {}

alias MaybeUnformed = Core.MaybeUnformed(HasTrivialDestruction);
fn Make() -> MaybeUnformed;

fn Test() {
  MaybeUnformed as Core.Destroy;
  MaybeUnformed as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: MaybeUnformed = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_maybe_unformed.carbon

library "[[@TEST_NAME]]";

import library "types";

class ImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

alias MaybeUnformed = Core.MaybeUnformed(ImplsDestroy);

fn Test() {
  MaybeUnformed as Core.TrivialDestroy;
}
