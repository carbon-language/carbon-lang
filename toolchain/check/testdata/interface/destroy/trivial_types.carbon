// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/destroy/trivial_types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/destroy/trivial_types.carbon

// --- check_destroy_format.carbon

library "[[@TEST_NAME]]";

// Identifies whether a type can be destroyed.
//
// Use this function to confirm non-trivial destruction (pass) and types that
// can't be destroyed (fail).
fn CheckTypeHasDestruction(unused T: Core.Destroy) {}

// Identifies whether a type has trivial destruction.
//
// Use this function to confirm trivial destruction (pass), non-trivial
// destruction (fail), and types that can't be destroyed (fail).
//
// `CheckTypeHasTrivialDestruction` should only be combined with
// `CheckTypeHasDestruction` for types that are expected to not have trivial
// destruction. Using both on a trivially destroyable type (e.g. `i32`) is
// redundant.
fn CheckTypeHasTrivialDestruction(unused T: Core.TrivialDestroy) {}

// --- types.carbon

library "[[@TEST_NAME]]";

// Types used by the splits below.

base class TrivialBase {}

abstract class Abstract {}

// --- constrained_facet.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestDestroyAndTrivialDestroy(generic T: Core.Destroy & Core.TrivialDestroy) {
  CheckTypeHasDestruction(T);

  CheckTypeHasTrivialDestruction(T);
}

fn TestTrivialDestroy(generic T: Core.TrivialDestroy) {
  CheckTypeHasDestruction(T);
}

// --- facet_type.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

fn TestType() {
  CheckTypeHasTrivialDestruction(type);
}

fn TestFacetType() {
  CheckTypeHasTrivialDestruction(Core.Destroy);
}

// --- partial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestPartialTrivialBase() {
  CheckTypeHasTrivialDestruction(partial TrivialBase);
}

fn TestPartialAbstract() {
  CheckTypeHasTrivialDestruction(partial Abstract);
}
