// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/import_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/import_access.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- def.carbon

package Test library "[[@TEST_NAME]]";

private class Def {}

// --- forward_with_def.carbon

package Test library "[[@TEST_NAME]]";

private class ForwardWithDef;

class ForwardWithDef {}

// --- forward.carbon

package Test library "[[@TEST_NAME]]";

private class Forward;

// ============================================================================
// Test files
// ============================================================================

// --- def.impl.carbon

impl package Test library "[[@TEST_NAME]]";

var c: Def = {};

// --- fail_local_def.carbon

package Test library "[[@TEST_NAME]]";

import library "def";

// CHECK:STDERR: fail_local_def.carbon:[[@LINE+4]]:8: error: name `Def` not found [NameNotFound]
// CHECK:STDERR: var c: Def = {};
// CHECK:STDERR:        ^~~
// CHECK:STDERR:
var c: Def = {};

// --- fail_other_def.carbon

package Other library "[[@TEST_NAME]]";

import Test library "def";

// CHECK:STDERR: fail_other_def.carbon:[[@LINE+4]]:8: error: member name `Def` not found in `Test` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: var c: Test.Def = {};
// CHECK:STDERR:        ^~~~~~~~
// CHECK:STDERR:
var c: Test.Def = {};

// --- forward_with_def.impl.carbon

impl package Test library "[[@TEST_NAME]]";

var c: ForwardWithDef = {};

// --- fail_local_forward_with_def.carbon

package Test library "[[@TEST_NAME]]";

import library "forward_with_def";

// CHECK:STDERR: fail_local_forward_with_def.carbon:[[@LINE+4]]:8: error: name `ForwardWithDef` not found [NameNotFound]
// CHECK:STDERR: var c: ForwardWithDef = {};
// CHECK:STDERR:        ^~~~~~~~~~~~~~
// CHECK:STDERR:
var c: ForwardWithDef = {};

// --- fail_other_forward_with_def.carbon

package Other library "[[@TEST_NAME]]";

import Test library "forward_with_def";

// CHECK:STDERR: fail_other_forward_with_def.carbon:[[@LINE+4]]:8: error: member name `ForwardWithDef` not found in `Test` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: var c: Test.ForwardWithDef = {};
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var c: Test.ForwardWithDef = {};

// --- forward.impl.carbon

impl package Test library "[[@TEST_NAME]]";

fn F(c: Forward*) {}

class Forward {}

// --- fail_local_forward.carbon

package Test library "[[@TEST_NAME]]";

import library "forward";

// CHECK:STDERR: fail_local_forward.carbon:[[@LINE+4]]:9: error: name `Forward` not found [NameNotFound]
// CHECK:STDERR: fn F(c: Forward*) {}
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR:
fn F(c: Forward*) {}

// --- fail_other_forward.carbon

package Other library "[[@TEST_NAME]]";

import Test library "forward";

// CHECK:STDERR: fail_other_forward.carbon:[[@LINE+4]]:9: error: member name `Forward` not found in `Test` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: fn F(c: Test.Forward*) {}
// CHECK:STDERR:         ^~~~~~~~~~~~
// CHECK:STDERR:
fn F(c: Test.Forward*) {}

// --- todo_fail_private_on_redecl.carbon

library "[[@TEST_NAME]]";

private class Redecl;

private class Redecl {}
