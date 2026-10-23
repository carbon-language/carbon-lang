// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/destroy/non_trivial_types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/destroy/non_trivial_types.carbon

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

// Has non-trivial destruction because of its own `DestroyImpl.Op`.
class NonTrivial {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// Has non-trivial destruction because of its own `DestroyImpl.Op`.
base class NonTrivialBase {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// Has non-trivial destruction because of a member.
base class NonTrivialMemberBase {
  var n: NonTrivial;
}

// A generic class with a member of type `T`. Tests that the destruction of a
// generic class comes from its type argument: `DestroyableBox(i32)` has trivial
// destruction, and `DestroyableBox(NonTrivial)` has non-trivial destruction.
// The generics tests also use it with a symbolic argument.
class DestroyableBox(T: Core.Destroy) {
  var t: T;
}

// --- constrained_facet.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestDestroy(generic T: Core.Destroy) {
  CheckTypeHasDestruction(T);
}

fn TestStructOfDestroy(generic T: Core.Destroy) {
  CheckTypeHasDestruction({.a: T});
}

fn TestTupleOfDestroy(generic T: Core.Destroy) {
  CheckTypeHasDestruction((T, i32));
}

fn TestArrayOfDestroy(generic T: Core.Destroy) {
  CheckTypeHasDestruction(array(T, 2));
}

fn TestGenericClassOfDestroy(generic T: Core.Destroy) {
  CheckTypeHasDestruction(DestroyableBox(T));
}

// --- todo_fail_constrained_facet_trivial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

// TODO: This should fail, because `T` might not have trivial destruction.
fn TestDestroy(generic T: Core.Destroy) {
  CheckTypeHasTrivialDestruction(T);
}

// --- fail_partial.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

fn TestPartialNonTrivialMemberBase() {
  CheckTypeHasDestruction(partial NonTrivialMemberBase);

  // CHECK:STDERR: fail_partial.carbon:[[@LINE+4]]:34-61: error: cannot convert type `partial NonTrivialMemberBase` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: imported from: fail_partial.carbon:[[@LINE-7]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:19:42-63: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  CheckTypeHasTrivialDestruction(partial NonTrivialMemberBase);
}

// --- todo_fail_partial_destroy_impl.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import library "types";

// TODO: `CheckTypeHasTrivialDestruction` should fail once `DestroyImpl.Op`
// takes `partial Self` instead of `Self`.
fn TestPartialNonTrivialBase() {
  CheckTypeHasDestruction(partial NonTrivialBase);

  CheckTypeHasTrivialDestruction(partial NonTrivialBase);
}
