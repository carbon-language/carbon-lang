// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/import_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/import_access.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- def.carbon

package Test library "[[@TEST_NAME]]";

private interface Def {}

// --- forward_with_def.carbon

package Test library "[[@TEST_NAME]]";

private interface ForwardWithDef;

interface ForwardWithDef {}

// --- forward.carbon

package Test library "[[@TEST_NAME]]";

// TODO: Import of forward declared interfaces crashes.
// private interface Forward;

// ============================================================================
// Test files
// ============================================================================

// --- def.impl.carbon

impl package Test library "[[@TEST_NAME]]";

fn F(i: Def) {}

// --- fail_local_def.carbon

package Test library "[[@TEST_NAME]]";

import library "def";

// CHECK:STDERR: fail_local_def.carbon:[[@LINE+4]]:9: error: name `Def` not found [NameNotFound]
// CHECK:STDERR: fn F(i: Def) {}
// CHECK:STDERR:         ^~~
// CHECK:STDERR:
fn F(i: Def) {}

// --- fail_other_def.carbon

package Other library "[[@TEST_NAME]]";

import Test library "def";

// CHECK:STDERR: fail_other_def.carbon:[[@LINE+4]]:9: error: member name `Def` not found in `Test` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: fn F(i: Test.Def) {}
// CHECK:STDERR:         ^~~~~~~~
// CHECK:STDERR:
fn F(i: Test.Def) {}

// --- forward_with_def.impl.carbon

impl package Test library "[[@TEST_NAME]]";

fn F(i: ForwardWithDef) {}

// --- fail_local_forward_with_def.carbon

package Test library "[[@TEST_NAME]]";

import library "forward_with_def";

// CHECK:STDERR: fail_local_forward_with_def.carbon:[[@LINE+4]]:9: error: name `ForwardWithDef` not found [NameNotFound]
// CHECK:STDERR: fn F(i: ForwardWithDef) {}
// CHECK:STDERR:         ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(i: ForwardWithDef) {}

// --- fail_other_forward_with_def.carbon

package Other library "[[@TEST_NAME]]";

import Test library "forward_with_def";

// CHECK:STDERR: fail_other_forward_with_def.carbon:[[@LINE+4]]:9: error: member name `ForwardWithDef` not found in `Test` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: fn F(i: Test.ForwardWithDef) {}
// CHECK:STDERR:         ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(i: Test.ForwardWithDef) {}

// --- fail_todo_forward.impl.carbon

impl package Test library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_forward.impl.carbon:[[@LINE+7]]:9: error: name `Forward` not found [NameNotFound]
// CHECK:STDERR: fn F(i: Forward*) {}
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_forward.impl.carbon:[[@LINE+3]]:9: error: name `Forward` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn F(i: Forward*) {}
// CHECK:STDERR:         ^~~~~~~
fn F(i: Forward*) {}

// CHECK:STDERR: fail_todo_forward.impl.carbon:[[@LINE+4]]:11: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: interface Forward {}
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR:
interface Forward {}

// --- fail_local_forward.carbon

package Test library "[[@TEST_NAME]]";

import library "forward";

// CHECK:STDERR: fail_local_forward.carbon:[[@LINE+4]]:9: error: name `Forward` not found [NameNotFound]
// CHECK:STDERR: fn F(i: Forward*) {}
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR:
fn F(i: Forward*) {}

// --- fail_other_forward.carbon

package Other library "[[@TEST_NAME]]";

import Test library "forward";

// CHECK:STDERR: fail_other_forward.carbon:[[@LINE+4]]:9: error: name `Forward` not found [NameNotFound]
// CHECK:STDERR: fn F(i: Forward*) {}
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR:
fn F(i: Forward*) {}

// --- todo_fail_private_on_redecl.carbon

library "[[@TEST_NAME]]";

private interface Redecl;

private interface Redecl {}

// CHECK:STDOUT: --- def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Def.type: type = facet_type <@Def> [concrete]
// CHECK:STDOUT:   %Self: %Def.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Def [private] = %Def.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Def.decl: type = interface_decl @Def [concrete = constants.%Def.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Def {
// CHECK:STDOUT:   %Self: %Def.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- forward_with_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ForwardWithDef.type: type = facet_type <@ForwardWithDef> [concrete]
// CHECK:STDOUT:   %Self: %ForwardWithDef.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .ForwardWithDef [private] = %ForwardWithDef.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ForwardWithDef.decl.loc4: type = interface_decl @ForwardWithDef [concrete = constants.%ForwardWithDef.type] {} {}
// CHECK:STDOUT:   %ForwardWithDef.decl.loc6: type = interface_decl @ForwardWithDef [concrete = constants.%ForwardWithDef.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardWithDef {
// CHECK:STDOUT:   %Self: %ForwardWithDef.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- forward.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- def.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Def.type: type = facet_type <@Def> [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Test.Def: type = import_ref Test//def, Def, loaded [concrete = constants.%Def.type]
// CHECK:STDOUT:   %Test.import_ref = import_ref Test//def, inst16 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Def [private] = imports.%Test.Def
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %i.patt: %Def.type = binding_pattern i
// CHECK:STDOUT:     %i.param_patt: %Def.type = value_param_pattern %i.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i.param: %Def.type = value_param call_param0
// CHECK:STDOUT:     %Def.ref: type = name_ref Def, imports.%Test.Def [concrete = constants.%Def.type]
// CHECK:STDOUT:     %i: %Def.type = bind_name i, %i.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Def [from "def.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Test.import_ref
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%i.param_patt: %Def.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Def = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %i.patt: <error> = binding_pattern i
// CHECK:STDOUT:     %i.param_patt: <error> = value_param_pattern %i.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i.param: <error> = value_param call_param0
// CHECK:STDOUT:     %Def.ref: <error> = name_ref Def, <error> [concrete = <error>]
// CHECK:STDOUT:     %i: <error> = bind_name i, %i.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%i.param_patt: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Test: <namespace> = namespace file.%Test.import, [concrete] {
// CHECK:STDOUT:     .Def = <poisoned>
// CHECK:STDOUT:     import Test//def
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Test = imports.%Test
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %i.patt: <error> = binding_pattern i
// CHECK:STDOUT:     %i.param_patt: <error> = value_param_pattern %i.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i.param: <error> = value_param call_param0
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %Test.ref: <namespace> = name_ref Test, imports.%Test [concrete = imports.%Test]
// CHECK:STDOUT:       %Def.ref: <error> = name_ref Def, <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %i: <error> = bind_name i, %i.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%i.param_patt: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- forward_with_def.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ForwardWithDef.type: type = facet_type <@ForwardWithDef> [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Test.ForwardWithDef: type = import_ref Test//forward_with_def, ForwardWithDef, loaded [concrete = constants.%ForwardWithDef.type]
// CHECK:STDOUT:   %Test.import_ref = import_ref Test//forward_with_def, inst17 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .ForwardWithDef [private] = imports.%Test.ForwardWithDef
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %i.patt: %ForwardWithDef.type = binding_pattern i
// CHECK:STDOUT:     %i.param_patt: %ForwardWithDef.type = value_param_pattern %i.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i.param: %ForwardWithDef.type = value_param call_param0
// CHECK:STDOUT:     %ForwardWithDef.ref: type = name_ref ForwardWithDef, imports.%Test.ForwardWithDef [concrete = constants.%ForwardWithDef.type]
// CHECK:STDOUT:     %i: %ForwardWithDef.type = bind_name i, %i.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardWithDef [from "forward_with_def.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Test.import_ref
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%i.param_patt: %ForwardWithDef.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_forward_with_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .ForwardWithDef = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %i.patt: <error> = binding_pattern i
// CHECK:STDOUT:     %i.param_patt: <error> = value_param_pattern %i.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i.param: <error> = value_param call_param0
// CHECK:STDOUT:     %ForwardWithDef.ref: <error> = name_ref ForwardWithDef, <error> [concrete = <error>]
// CHECK:STDOUT:     %i: <error> = bind_name i, %i.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%i.param_patt: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other_forward_with_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Test: <namespace> = namespace file.%Test.import, [concrete] {
// CHECK:STDOUT:     .ForwardWithDef = <poisoned>
// CHECK:STDOUT:     import Test//forward_with_def
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Test = imports.%Test
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %i.patt: <error> = binding_pattern i
// CHECK:STDOUT:     %i.param_patt: <error> = value_param_pattern %i.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i.param: <error> = value_param call_param0
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %Test.ref: <namespace> = name_ref Test, imports.%Test [concrete = imports.%Test]
// CHECK:STDOUT:       %ForwardWithDef.ref: <error> = name_ref ForwardWithDef, <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %i: <error> = bind_name i, %i.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%i.param_patt: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_forward.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Forward.type: type = facet_type <@Forward> [concrete]
// CHECK:STDOUT:   %Self: %Forward.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Forward = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %i.patt: <error> = binding_pattern i
// CHECK:STDOUT:     %i.param_patt: <error> = value_param_pattern %i.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i.param: <error> = value_param call_param0
// CHECK:STDOUT:     %.loc11: type = splice_block %ptr [concrete = <error>] {
// CHECK:STDOUT:       %Forward.ref: <error> = name_ref Forward, <error> [concrete = <error>]
// CHECK:STDOUT:       %ptr: type = ptr_type <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %i: <error> = bind_name i, %i.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Forward.decl: type = interface_decl @Forward [concrete = constants.%Forward.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Forward {
// CHECK:STDOUT:   %Self: %Forward.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%i.param_patt: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_forward.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Forward = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %i.patt: <error> = binding_pattern i
// CHECK:STDOUT:     %i.param_patt: <error> = value_param_pattern %i.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i.param: <error> = value_param call_param0
// CHECK:STDOUT:     %.loc10: type = splice_block %ptr [concrete = <error>] {
// CHECK:STDOUT:       %Forward.ref: <error> = name_ref Forward, <error> [concrete = <error>]
// CHECK:STDOUT:       %ptr: type = ptr_type <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %i: <error> = bind_name i, %i.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%i.param_patt: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other_forward.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Test: <namespace> = namespace file.%Test.import, [concrete] {
// CHECK:STDOUT:     import Test//forward
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Test = imports.%Test
// CHECK:STDOUT:     .Forward = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %i.patt: <error> = binding_pattern i
// CHECK:STDOUT:     %i.param_patt: <error> = value_param_pattern %i.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i.param: <error> = value_param call_param0
// CHECK:STDOUT:     %.loc10: type = splice_block %ptr [concrete = <error>] {
// CHECK:STDOUT:       %Forward.ref: <error> = name_ref Forward, <error> [concrete = <error>]
// CHECK:STDOUT:       %ptr: type = ptr_type <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %i: <error> = bind_name i, %i.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%i.param_patt: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_private_on_redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Redecl.type: type = facet_type <@Redecl> [concrete]
// CHECK:STDOUT:   %Self: %Redecl.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Redecl [private] = %Redecl.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Redecl.decl.loc4: type = interface_decl @Redecl [concrete = constants.%Redecl.type] {} {}
// CHECK:STDOUT:   %Redecl.decl.loc6: type = interface_decl @Redecl [concrete = constants.%Redecl.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Redecl {
// CHECK:STDOUT:   %Self: %Redecl.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
