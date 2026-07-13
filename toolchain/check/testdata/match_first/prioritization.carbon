// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// ^ARGS: -v compile %s
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/match_first/prioritization.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/match_first/prioritization.carbon

// --- overlap_allowed_in_match_first.carbon
library "[[@TEST_NAME]]";

interface Z { let Z1: type; }
interface Y {}
interface X {}

impl forall [T: Y] T as Z where .Z1 = {.y: type} {}
impl forall [T: X] T as Z where .Z1 = {.x: type} {}

match_first {
  impl forall [T: Y] T as Z where .Z1 = {.y: type};
  impl forall [T: X] T as Z where .Z1 = {.x: type};
}

// --- fail_overlap_not_allowed_in_different_match_first.carbon
library "[[@TEST_NAME]]";

interface Z { let Z1: type; }
interface Y {}
interface X {}

impl forall [T: Y] T as Z where .Z1 = {.y: type} {}
// CHECK:STDERR: fail_overlap_not_allowed_in_different_match_first.carbon:[[@LINE+7]]:1: error: found non-final `impl` with the same type structure as another non-final `impl` [ImplNonFinalSameTypeStructure]
// CHECK:STDERR: impl forall [T: X] T as Z where .Z1 = {.x: type} {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_overlap_not_allowed_in_different_match_first.carbon:[[@LINE-4]]:1: note: other `impl` here [ImplNonFinalSameTypeStructureNote]
// CHECK:STDERR: impl forall [T: Y] T as Z where .Z1 = {.y: type} {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T: X] T as Z where .Z1 = {.x: type} {}

match_first {
  impl forall [T: Y] T as Z where .Z1 = {.y: type};
}
match_first {
  impl forall [T: X] T as Z where .Z1 = {.x: type};
}

// --- lookup_overlapping_impl.carbon
library "[[@TEST_NAME]]";

interface Z { let Z1: type; }
interface Y {}
interface X {}

impl forall [T: Y] T as Z where .Z1 = {.y: type} {}
impl forall [T: X] T as Z where .Z1 = {.x: type} {}

match_first {
  impl forall [T: Y] T as Z where .Z1 = {.y: type};
  impl forall [T: X] T as Z where .Z1 = {.x: type};
}

fn F() {
  class FC;
  impl FC as Y {}
  // Since GC impls Y and not X, the .y impl should be selected.
  {.y = type} as FC.(Z.Z1);
}

fn G() {
  class GC;
  impl GC as X {}
  // Since GC impls X and not Y, the .x impl should be selected.
  {.x = type} as GC.(Z.Z1);
}

// --- prioritization_first_declarared.carbon
library "[[@TEST_NAME]]";

interface Z { let Z1: type; }
interface Y {}
interface X {}

impl forall [T: Y] T as Z where .Z1 = {.y: type} {}
impl forall [T: X] T as Z where .Z1 = {.x: type} {}

match_first {
  // The match_first block is in the same order that the impls are introduced.
  // This makes us prioritize the first introduced impl.
  impl forall [T: Y] T as Z where .Z1 = {.y: type};
  impl forall [T: X] T as Z where .Z1 = {.x: type};
}

fn F() {
  class C;
  impl C as Y {}
  impl C as X {}
  // Since C impls X and Y, the first one in the match_first block
  // should be selected, which is the .y impl.
  {.y = type} as C.(Z.Z1);
}

// --- prioritization_second_declarared.carbon
library "[[@TEST_NAME]]";

interface Z { let Z1: type; }
interface Y {}
interface X {}

impl forall [T: Y] T as Z where .Z1 = {.y: type} {}
impl forall [T: X] T as Z where .Z1 = {.x: type} {}

match_first {
  // The match_first block is in the opposite order that the impls are
  // introduced. This makes us prioritize the second introduced impl.
  impl forall [T: X] T as Z where .Z1 = {.x: type};
  impl forall [T: Y] T as Z where .Z1 = {.y: type};
}

fn F() {
  class C;
  impl C as Y {}
  impl C as X {}
  // Since C impls X and Y, the first one in the match_first block
  // should be selected, which is the .x impl.
  {.x = type} as C.(Z.Z1);
}

// --- fail_prioritization_does_not_choose_second_declarared.carbon
library "[[@TEST_NAME]]";

interface Z { let Z1: type; }
interface Y {}
interface X {}

impl forall [T: Y] T as Z where .Z1 = {.y: type} {}
impl forall [T: X] T as Z where .Z1 = {.x: type} {}

match_first {
  // The match_first block is in the same order that the impls are introduced.
  // This makes us prioritize the first introduced impl.
  impl forall [T: Y] T as Z where .Z1 = {.y: type};
  impl forall [T: X] T as Z where .Z1 = {.x: type};
}

fn F() {
  class C;
  impl C as Y {}
  impl C as X {}
  // Since C impls X and Y, the first one in the match_first block
  // should be selected, which is the .y impl. So this should fail.
  // CHECK:STDERR: fail_prioritization_does_not_choose_second_declarared.carbon:[[@LINE+4]]:3: error: struct `{.y: type}` has no field named `x` [StructInitUnexpectedFieldInLiteral]
  // CHECK:STDERR:   {.x = type} as C.(Z.Z1);
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  {.x = type} as C.(Z.Z1);
}

// --- fail_prioritization_does_not_choose_first_declarared.carbon
library "[[@TEST_NAME]]";

interface Z { let Z1: type; }
interface Y {}
interface X {}

impl forall [T: Y] T as Z where .Z1 = {.y: type} {}
impl forall [T: X] T as Z where .Z1 = {.x: type} {}

match_first {
  // The match_first block is in the opposite order that the impls are
  // introduced. This makes us prioritize the second introduced impl.
  impl forall [T: X] T as Z where .Z1 = {.x: type};
  impl forall [T: Y] T as Z where .Z1 = {.y: type};
}

fn F() {
  class C;
  impl C as Y {}
  impl C as X {}
  // Since C impls X and Y, the first one in the match_first block
  // should be selected, which is the .x impl. So this should fail.
  // CHECK:STDERR: fail_prioritization_does_not_choose_first_declarared.carbon:[[@LINE+4]]:3: error: struct `{.x: type}` has no field named `y` [StructInitUnexpectedFieldInLiteral]
  // CHECK:STDERR:   {.y = type} as C.(Z.Z1);
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  {.y = type} as C.(Z.Z1);
}

// --- final_prioritization_first_declarared.carbon
library "[[@TEST_NAME]]";

interface Z { let Z1: type; }
interface Y {}
interface X {}

impl forall [T: Y] T as Z where .Z1 = {.y: type} {}
impl forall [T: X] T as Z where .Z1 = {.x: type} {}

final match_first {
  // The match_first block is in the same order that the impls are introduced.
  // This makes us prioritize the first introduced impl.
  impl forall [T: Y] T as Z where .Z1 = {.y: type};
  impl forall [T: X] T as Z where .Z1 = {.x: type};
}

fn F[U: type where .Self impls X and .Self impls Y]() {
  // Since U impls X and Y, the first one in the match_first block
  // should be selected, which is the .y impl.
  {.y = type} as U.(Z.Z1);
}

// --- final_prioritization_second_declarared.carbon
library "[[@TEST_NAME]]";

interface Z { let Z1: type; }
interface Y {}
interface X {}

impl forall [T: Y] T as Z where .Z1 = {.y: type} {}
impl forall [T: X] T as Z where .Z1 = {.x: type} {}

final match_first {
  // The match_first block is in the opposite order that the impls are
  // introduced. This makes us prioritize the second introduced impl.
  impl forall [T: X] T as Z where .Z1 = {.x: type};
  impl forall [T: Y] T as Z where .Z1 = {.y: type};
}

fn F[U: type where .Self impls X and .Self impls Y]() {
  // Since U impls X and Y, the first one in the match_first block
  // should be selected, which is the .x impl.
  {.x = type} as U.(Z.Z1);
}

// --- fail_match_first_does_not_imply_final.carbon
library "[[@TEST_NAME]]";

interface Z { let Z1: type; }
interface Y {}
interface X {}

impl forall [T: Y] T as Z where .Z1 = {.y: type} {}
impl forall [T: X] T as Z where .Z1 = {.x: type} {}

match_first {
  // The match_first block is in the same order that the impls are introduced.
  // This makes us prioritize the first introduced impl.
  impl forall [T: Y] T as Z where .Z1 = {.y: type};
  impl forall [T: X] T as Z where .Z1 = {.x: type};
}

fn F[U: type where .Self impls X and .Self impls Y]() {
  // The impls are not final, so we can't use the concrete `.Z1` value provided
  // by the impls here. This conversion should fail.
  // CHECK:STDERR: fail_match_first_does_not_imply_final.carbon:[[@LINE+4]]:3: error: `Core.As` implicitly referenced here, but package `Core` not found [CoreNotFound]
  // CHECK:STDERR:   {.y = type} as U.(Z.Z1);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  {.y = type} as U.(Z.Z1);
}

// --- non_final_more_specific_than_match_first.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) { let Z1: type; }
interface Y {}
interface X {}
interface W {}

class C(T: type);
impl forall [T: type] C(T) as W {}
impl forall [T: type] C(T) as Y {}

// These impls have different type structures, but they overlap. For non-final
// impls we want to choose the most specific one (which will be the middle one),
// even if there is a match_first block for the others (but not for the middle
// one).
impl forall [T: W] () as Z(T) where .Z1 = {.w: type} {}
impl forall [T: X] () as Z(C(T)) where .Z1 = {.x: type} {}
impl forall [T: Y] () as Z(T) where .Z1 = {.y: type} {}

match_first {
  impl forall [T: W] () as Z(T) where .Z1 = {.w: type};
  impl forall [T: Y] () as Z(T) where .Z1 = {.y: type};
}

fn F() {
  class D;
  impl D as X {}
  {.x = type} as ().(Z(C(D)).Z1);
}

// --- non_final_overlap_in_match_first.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) { let Z1: type; }
interface Y {}
interface X {}
interface W {}

class C(T: type);
impl forall [T: type] C(T) as W {}
impl forall [T: type] C(T) as Y {}

// These impls have different type structures, but they overlap. For non-final
// impls we would choose the most specific one (which will be the middle one),
// but then we find it is in a match_first block. So we use the first matching
// impl in that block instead.
impl forall [T: W] () as Z(T) where .Z1 = {.w: type} {}
impl forall [T: X] () as Z(C(T)) where .Z1 = {.x: type} {}
impl forall [T: Y] () as Z(T) where .Z1 = {.y: type} {}

match_first {
  impl forall [T: W] () as Z(T) where .Z1 = {.w: type};
  impl forall [T: X] () as Z(C(T)) where .Z1 = {.x: type};
  impl forall [T: Y] () as Z(T) where .Z1 = {.y: type};
}

fn F() {
  class D;
  impl D as X {}
  {.w = type} as ().(Z(C(D)).Z1);
}

// --- final_overlap_in_match_first.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) { let Z1: type; }
interface Y {}
interface X {}
interface W {}

class C(T: type);
impl forall [T: type] C(T) as W {}
impl forall [T: type] C(T) as Y {}

// These impls have different type structures, but they overlap. Overlapping
// final impls are always in a match_first block together, so we will use the
// first impl from that block, not the most specific one.
impl forall [T: W] () as Z(T) where .Z1 = {.w: type} {}
impl forall [T: X] () as Z(C(T)) where .Z1 = {.x: type} {}
impl forall [T: Y] () as Z(T) where .Z1 = {.y: type} {}

final match_first {
  impl forall [T: Y] () as Z(T) where .Z1 = {.y: type};
  impl forall [T: X] () as Z(C(T)) where .Z1 = {.x: type};
  impl forall [T: W] () as Z(T) where .Z1 = {.w: type};
}

fn F[U: X]() {
  {.y = type} as ().(Z(C(U)).Z1);
}
