// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/named_constraint/import_constraint_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/named_constraint/import_constraint_decl.carbon

// --- a.carbon
library "[[@TEST_NAME]]";

constraint A;

// --- a.impl.carbon
//@include-in-dumps
impl library "[[@TEST_NAME]]";

// --- b.carbon
library "[[@TEST_NAME]]";

interface I {}
constraint B {
  extend require impls I;
}

// --- b.impl.carbon
//@include-in-dumps
impl library "[[@TEST_NAME]]";

fn F(T:! B) {}

// --- fail_import_not_identified.carbon
library "[[@TEST_NAME]]";

import library "a";

class C {}
// CHECK:STDERR: fail_import_not_identified.carbon:[[@LINE+8]]:1: error: facet type `A` cannot be identified in `impl as` [ImplOfUnidentifiedFacetType]
// CHECK:STDERR: impl C as A;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR: fail_import_not_identified.carbon:[[@LINE-6]]:1: in import [InImport]
// CHECK:STDERR: a.carbon:3:1: note: constraint was forward declared here [NamedConstraintForwardDeclaredHere]
// CHECK:STDERR: constraint A;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C as A;

// --- fail_todo_import_complete.carbon
library "[[@TEST_NAME]]";

import library "b";

class C {}

// This requires that B is identified.
// TODO: This should work.
// CHECK:STDERR: fail_todo_import_complete.carbon:[[@LINE+4]]:1: error: impl as 0 interfaces, expected 1 [ImplOfNotOneInterface]
// CHECK:STDERR: impl C as B;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
impl C as B;

// This requires that B is complete.
// TODO: This should work.
// CHECK:STDERR: fail_todo_import_complete.carbon:[[@LINE+4]]:1: error: impl as 0 interfaces, expected 1 [ImplOfNotOneInterface]
// CHECK:STDERR: impl C as B {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C as B {}

// CHECK:STDOUT: --- a.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.A = import_ref Main//a, A, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_17.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_17.2 = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %T: %B.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %B.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I = import_ref Main//b, I, unloaded
// CHECK:STDOUT:   %Main.B: type = import_ref Main//b, B, loaded [concrete = constants.%B.type]
// CHECK:STDOUT:   %Main.import_ref = import_ref Main//b, loc4_14, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_17.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_17.2 = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %B.ref [concrete = constants.%B.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %B.ref: type = name_ref B, imports.%Main.B [concrete = constants.%B.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_6.2: %B.type = symbolic_binding T, 0 [symbolic = %T.loc4_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @B [from "b.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_6.2: %B.type) {
// CHECK:STDOUT:   %T.loc4_6.1: %B.type = symbolic_binding T, 0 [symbolic = %T.loc4_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc4_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
