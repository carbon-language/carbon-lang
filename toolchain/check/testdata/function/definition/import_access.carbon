// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/definition/import_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/definition/import_access.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- def.carbon

package Test library "def";

private fn Def() {}

// --- forward_with_def.carbon

package Test library "forward_with_def";

private fn ForwardWithDef();

fn ForwardWithDef() {}

// --- forward.carbon

package Test library "forward";

private fn Forward();

// ============================================================================
// Test files
// ============================================================================

// --- fail_todo_def.impl.carbon

impl package Test library "def";

// CHECK:STDERR: fail_todo_def.impl.carbon:[[@LINE+4]]:13: ERROR: Name `Def` not found.
// CHECK:STDERR: var f: () = Def();
// CHECK:STDERR:             ^~~
// CHECK:STDERR:
var f: () = Def();

// --- fail_local_def.carbon

package Test library "fail_local_def";

import library "def";

// CHECK:STDERR: fail_local_def.carbon:[[@LINE+4]]:13: ERROR: Name `Def` not found.
// CHECK:STDERR: var f: () = Def();
// CHECK:STDERR:             ^~~
// CHECK:STDERR:
var f: () = Def();

// --- fail_other_def.carbon

package Other library "fail_other_def";

import Test library "def";

// CHECK:STDERR: fail_other_def.carbon:[[@LINE+4]]:13: ERROR: Name `Def` not found.
// CHECK:STDERR: var f: () = Test.Def();
// CHECK:STDERR:             ^~~~~~~~
// CHECK:STDERR:
var f: () = Test.Def();

// --- fail_todo_forward_with_def.impl.carbon

impl package Test library "forward_with_def";

// CHECK:STDERR: fail_todo_forward_with_def.impl.carbon:[[@LINE+4]]:13: ERROR: Name `ForwardWithDef` not found.
// CHECK:STDERR: var f: () = ForwardWithDef();
// CHECK:STDERR:             ^~~~~~~~~~~~~~
// CHECK:STDERR:
var f: () = ForwardWithDef();

// --- fail_local_forward_with_def.carbon

package Test library "fail_local_forward_with_def";

import library "forward_with_def";

// CHECK:STDERR: fail_local_forward_with_def.carbon:[[@LINE+4]]:13: ERROR: Name `ForwardWithDef` not found.
// CHECK:STDERR: var f: () = ForwardWithDef();
// CHECK:STDERR:             ^~~~~~~~~~~~~~
// CHECK:STDERR:
var f: () = ForwardWithDef();

// --- fail_other_forward_with_def.carbon

package Other library "fail_other_forward_with_def";

import Test library "forward_with_def";

// CHECK:STDERR: fail_other_forward_with_def.carbon:[[@LINE+4]]:13: ERROR: Name `ForwardWithDef` not found.
// CHECK:STDERR: var f: () = Test.ForwardWithDef();
// CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var f: () = Test.ForwardWithDef();

// --- fail_todo_forward.impl.carbon

impl package Test library "forward";

// CHECK:STDERR: fail_todo_forward.impl.carbon:[[@LINE+4]]:13: ERROR: Name `Forward` not found.
// CHECK:STDERR: var f: () = Forward();
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
var f: () = Forward();

fn Forward() {}

// --- fail_local_forward.carbon

package Test library "fail_local_forward";

import library "forward";

// CHECK:STDERR: fail_local_forward.carbon:[[@LINE+4]]:13: ERROR: Name `Forward` not found.
// CHECK:STDERR: var f: () = Forward();
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
var f: () = Forward();

// --- fail_other_forward.carbon

package Other library "fail_other_forward";

import Test library "forward";

// CHECK:STDERR: fail_other_forward.carbon:[[@LINE+3]]:13: ERROR: Name `Forward` not found.
// CHECK:STDERR: var f: () = Test.Forward();
// CHECK:STDERR:             ^~~~~~~~~~~~
var f: () = Test.Forward();

// --- todo_fail_private_on_redecl.carbon

library "fail_private_on_redecl";

private fn Redecl();

private fn Redecl() {}

// CHECK:STDOUT: --- def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Def.type: type = fn_type @Def [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Def: %Def.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Def [private] = %Def.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Def.decl: %Def.type = fn_decl @Def [template = constants.%Def] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Def() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- forward_with_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ForwardWithDef.type: type = fn_type @ForwardWithDef [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %ForwardWithDef: %ForwardWithDef.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .ForwardWithDef [private] = %ForwardWithDef.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %ForwardWithDef.decl.loc4: %ForwardWithDef.type = fn_decl @ForwardWithDef [template = constants.%ForwardWithDef] {}
// CHECK:STDOUT:   %ForwardWithDef.decl.loc6: %ForwardWithDef.type = fn_decl @ForwardWithDef [template = constants.%ForwardWithDef] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ForwardWithDef() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- forward.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Forward.type: type = fn_type @Forward [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Forward: %Forward.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Forward [private] = %Forward.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Forward.decl: %Forward.type = fn_decl @Forward [template = constants.%Forward] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Forward();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_def.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc8_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc8_9.2: type = converted %.loc8_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %f.var: ref %.1 = var f
// CHECK:STDOUT:   %f: ref %.1 = bind_runtime_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Def.ref: <error> = name_ref Def, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%f.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc10_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc10_9.2: type = converted %.loc10_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %f.var: ref %.1 = var f
// CHECK:STDOUT:   %f: ref %.1 = bind_runtime_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Def.ref: <error> = name_ref Def, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%f.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Test = %Test
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Test: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc10_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc10_9.2: type = converted %.loc10_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %f.var: ref %.1 = var f
// CHECK:STDOUT:   %f: ref %.1 = bind_runtime_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref: <namespace> = name_ref Test, file.%Test [template = file.%Test]
// CHECK:STDOUT:   %Def.ref: <error> = name_ref Def, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%f.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_forward_with_def.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc8_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc8_9.2: type = converted %.loc8_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %f.var: ref %.1 = var f
// CHECK:STDOUT:   %f: ref %.1 = bind_runtime_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ForwardWithDef.ref: <error> = name_ref ForwardWithDef, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%f.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_forward_with_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc10_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc10_9.2: type = converted %.loc10_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %f.var: ref %.1 = var f
// CHECK:STDOUT:   %f: ref %.1 = bind_runtime_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ForwardWithDef.ref: <error> = name_ref ForwardWithDef, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%f.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other_forward_with_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Test = %Test
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Test: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc10_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc10_9.2: type = converted %.loc10_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %f.var: ref %.1 = var f
// CHECK:STDOUT:   %f: ref %.1 = bind_runtime_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref: <namespace> = name_ref Test, file.%Test [template = file.%Test]
// CHECK:STDOUT:   %ForwardWithDef.ref: <error> = name_ref ForwardWithDef, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%f.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_forward.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Forward.type: type = fn_type @Forward [template]
// CHECK:STDOUT:   %Forward: %Forward.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:     .Forward = %Forward.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc8_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc8_9.2: type = converted %.loc8_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %f.var: ref %.1 = var f
// CHECK:STDOUT:   %f: ref %.1 = bind_runtime_name f, %f.var
// CHECK:STDOUT:   %Forward.decl: %Forward.type = fn_decl @Forward [template = constants.%Forward] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Forward() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Forward.ref: <error> = name_ref Forward, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%f.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_forward.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc10_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc10_9.2: type = converted %.loc10_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %f.var: ref %.1 = var f
// CHECK:STDOUT:   %f: ref %.1 = bind_runtime_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Forward.ref: <error> = name_ref Forward, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%f.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other_forward.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Test = %Test
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Test: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc9_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc9_9.2: type = converted %.loc9_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %f.var: ref %.1 = var f
// CHECK:STDOUT:   %f: ref %.1 = bind_runtime_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref: <namespace> = name_ref Test, file.%Test [template = file.%Test]
// CHECK:STDOUT:   %Forward.ref: <error> = name_ref Forward, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%f.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_private_on_redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Redecl.type: type = fn_type @Redecl [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Redecl: %Redecl.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Redecl [private] = %Redecl.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Redecl.decl.loc4: %Redecl.type = fn_decl @Redecl [template = constants.%Redecl] {}
// CHECK:STDOUT:   %Redecl.decl.loc6: %Redecl.type = fn_decl @Redecl [template = constants.%Redecl] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Redecl() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
