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

// CHECK:STDERR: fail_local_def.carbon:[[@LINE+4]]:9: error: name `Def` not found
// CHECK:STDERR: fn F(i: Def) {}
// CHECK:STDERR:         ^~~
// CHECK:STDERR:
fn F(i: Def) {}

// --- fail_other_def.carbon

package Other library "[[@TEST_NAME]]";

import Test library "def";

// CHECK:STDERR: fail_other_def.carbon:[[@LINE+4]]:9: error: name `Def` not found
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

// CHECK:STDERR: fail_local_forward_with_def.carbon:[[@LINE+4]]:9: error: name `ForwardWithDef` not found
// CHECK:STDERR: fn F(i: ForwardWithDef) {}
// CHECK:STDERR:         ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(i: ForwardWithDef) {}

// --- fail_other_forward_with_def.carbon

package Other library "[[@TEST_NAME]]";

import Test library "forward_with_def";

// CHECK:STDERR: fail_other_forward_with_def.carbon:[[@LINE+4]]:9: error: name `ForwardWithDef` not found
// CHECK:STDERR: fn F(i: Test.ForwardWithDef) {}
// CHECK:STDERR:         ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(i: Test.ForwardWithDef) {}

// --- fail_todo_forward.impl.carbon

impl package Test library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_forward.impl.carbon:[[@LINE+4]]:9: error: name `Forward` not found
// CHECK:STDERR: fn F(i: Forward*) {}
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR:
fn F(i: Forward*) {}

interface Forward {}

// --- fail_local_forward.carbon

package Test library "[[@TEST_NAME]]";

import library "forward";

// CHECK:STDERR: fail_local_forward.carbon:[[@LINE+4]]:9: error: name `Forward` not found
// CHECK:STDERR: fn F(i: Forward*) {}
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR:
fn F(i: Forward*) {}

// --- fail_other_forward.carbon

package Other library "[[@TEST_NAME]]";

import Test library "forward";

// CHECK:STDERR: fail_other_forward.carbon:[[@LINE+3]]:9: error: name `Forward` not found
// CHECK:STDERR: fn F(i: Forward*) {}
// CHECK:STDERR:         ^~~~~~~
fn F(i: Forward*) {}

// --- todo_fail_private_on_redecl.carbon

library "[[@TEST_NAME]]";

private interface Redecl;

private interface Redecl {}

// CHECK:STDOUT: --- def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Def.type: type = interface_type @Def [template]
// CHECK:STDOUT:   %Self: %Def.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Def [private] = %Def.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Def.decl: type = interface_decl @Def [template = constants.%Def.type] {} {}
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
// CHECK:STDOUT:   %ForwardWithDef.type: type = interface_type @ForwardWithDef [template]
// CHECK:STDOUT:   %Self: %ForwardWithDef.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .ForwardWithDef [private] = %ForwardWithDef.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ForwardWithDef.decl.loc4: type = interface_decl @ForwardWithDef [template = constants.%ForwardWithDef.type] {} {}
// CHECK:STDOUT:   %ForwardWithDef.decl.loc6: type = interface_decl @ForwardWithDef [template = constants.%ForwardWithDef.type] {} {}
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
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- def.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Def.type: type = interface_type @Def [template]
// CHECK:STDOUT:   %Self: %Def.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Test//def, inst+1, loaded [template = constants.%Def.type]
// CHECK:STDOUT:   %import_ref.2 = import_ref Test//def, inst+3, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Def [private] = imports.%import_ref.1
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %i.patt: %Def.type = binding_pattern i
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Def.ref: type = name_ref Def, imports.%import_ref.1 [template = constants.%Def.type]
// CHECK:STDOUT:     %i.param: %Def.type = param i, runtime_param0
// CHECK:STDOUT:     %i: %Def.type = bind_name i, %i.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Def {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%i: %Def.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %i.patt: <error> = binding_pattern i
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Def.ref: <error> = name_ref Def, <error> [template = <error>]
// CHECK:STDOUT:     %i.param: <error> = param i, runtime_param0
// CHECK:STDOUT:     %i: <error> = bind_name i, %i.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%i: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Test: <namespace> = namespace file.%Test.import, [template] {
// CHECK:STDOUT:     import Test//def
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Test = imports.%Test
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %i.patt: <error> = binding_pattern i
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Test.ref: <namespace> = name_ref Test, imports.%Test [template = imports.%Test]
// CHECK:STDOUT:     %Def.ref: <error> = name_ref Def, <error> [template = <error>]
// CHECK:STDOUT:     %i.param: <error> = param i, runtime_param0
// CHECK:STDOUT:     %i: <error> = bind_name i, %i.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%i: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- forward_with_def.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ForwardWithDef.type: type = interface_type @ForwardWithDef [template]
// CHECK:STDOUT:   %Self: %ForwardWithDef.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Test//forward_with_def, inst+1, loaded [template = constants.%ForwardWithDef.type]
// CHECK:STDOUT:   %import_ref.2 = import_ref Test//forward_with_def, inst+4, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .ForwardWithDef [private] = imports.%import_ref.1
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %i.patt: %ForwardWithDef.type = binding_pattern i
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %ForwardWithDef.ref: type = name_ref ForwardWithDef, imports.%import_ref.1 [template = constants.%ForwardWithDef.type]
// CHECK:STDOUT:     %i.param: %ForwardWithDef.type = param i, runtime_param0
// CHECK:STDOUT:     %i: %ForwardWithDef.type = bind_name i, %i.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardWithDef {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%i: %ForwardWithDef.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_forward_with_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %i.patt: <error> = binding_pattern i
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %ForwardWithDef.ref: <error> = name_ref ForwardWithDef, <error> [template = <error>]
// CHECK:STDOUT:     %i.param: <error> = param i, runtime_param0
// CHECK:STDOUT:     %i: <error> = bind_name i, %i.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%i: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other_forward_with_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Test: <namespace> = namespace file.%Test.import, [template] {
// CHECK:STDOUT:     import Test//forward_with_def
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Test = imports.%Test
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %i.patt: <error> = binding_pattern i
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Test.ref: <namespace> = name_ref Test, imports.%Test [template = imports.%Test]
// CHECK:STDOUT:     %ForwardWithDef.ref: <error> = name_ref ForwardWithDef, <error> [template = <error>]
// CHECK:STDOUT:     %i.param: <error> = param i, runtime_param0
// CHECK:STDOUT:     %i: <error> = bind_name i, %i.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%i: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_forward.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Forward.type: type = interface_type @Forward [template]
// CHECK:STDOUT:   %Self: %Forward.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Forward = %Forward.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %i.patt: <error> = binding_pattern i
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Forward.ref: <error> = name_ref Forward, <error> [template = <error>]
// CHECK:STDOUT:     %.loc8: type = ptr_type <error> [template = <error>]
// CHECK:STDOUT:     %i.param: <error> = param i, runtime_param0
// CHECK:STDOUT:     %i: <error> = bind_name i, %i.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Forward.decl: type = interface_decl @Forward [template = constants.%Forward.type] {} {}
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
// CHECK:STDOUT: fn @F(%i: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_forward.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %i.patt: <error> = binding_pattern i
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Forward.ref: <error> = name_ref Forward, <error> [template = <error>]
// CHECK:STDOUT:     %.loc10: type = ptr_type <error> [template = <error>]
// CHECK:STDOUT:     %i.param: <error> = param i, runtime_param0
// CHECK:STDOUT:     %i: <error> = bind_name i, %i.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%i: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other_forward.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Test: <namespace> = namespace file.%Test.import, [template] {
// CHECK:STDOUT:     import Test//forward
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Test = imports.%Test
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %i.patt: <error> = binding_pattern i
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Forward.ref: <error> = name_ref Forward, <error> [template = <error>]
// CHECK:STDOUT:     %.loc9: type = ptr_type <error> [template = <error>]
// CHECK:STDOUT:     %i.param: <error> = param i, runtime_param0
// CHECK:STDOUT:     %i: <error> = bind_name i, %i.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%i: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_private_on_redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Redecl.type: type = interface_type @Redecl [template]
// CHECK:STDOUT:   %Self: %Redecl.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Redecl [private] = %Redecl.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Redecl.decl.loc4: type = interface_decl @Redecl [template = constants.%Redecl.type] {} {}
// CHECK:STDOUT:   %Redecl.decl.loc6: type = interface_decl @Redecl [template = constants.%Redecl.type] {} {}
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
