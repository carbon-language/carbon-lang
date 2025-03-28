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

package Test library "[[@TEST_NAME]]";

private fn Def() {}

// --- forward_with_def.carbon

package Test library "[[@TEST_NAME]]";

private fn ForwardWithDef();

fn ForwardWithDef() {}

// --- forward.carbon

package Test library "[[@TEST_NAME]]";

private fn Forward();

// ============================================================================
// Test files
// ============================================================================

// --- def.impl.carbon

impl package Test library "[[@TEST_NAME]]";

var f: () = Def();

// --- fail_local_def.carbon

package Test library "[[@TEST_NAME]]";

import library "def";

// CHECK:STDERR: fail_local_def.carbon:[[@LINE+4]]:13: error: name `Def` not found [NameNotFound]
// CHECK:STDERR: var f: () = Def();
// CHECK:STDERR:             ^~~
// CHECK:STDERR:
var f: () = Def();

// --- fail_other_def.carbon

package Other library "[[@TEST_NAME]]";

import Test library "def";

// CHECK:STDERR: fail_other_def.carbon:[[@LINE+4]]:13: error: member name `Def` not found in `Test` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: var f: () = Test.Def();
// CHECK:STDERR:             ^~~~~~~~
// CHECK:STDERR:
var f: () = Test.Def();

// --- forward_with_def.impl.carbon

impl package Test library "[[@TEST_NAME]]";

var f: () = ForwardWithDef();

// --- fail_local_forward_with_def.carbon

package Test library "[[@TEST_NAME]]";

import library "forward_with_def";

// CHECK:STDERR: fail_local_forward_with_def.carbon:[[@LINE+4]]:13: error: name `ForwardWithDef` not found [NameNotFound]
// CHECK:STDERR: var f: () = ForwardWithDef();
// CHECK:STDERR:             ^~~~~~~~~~~~~~
// CHECK:STDERR:
var f: () = ForwardWithDef();

// --- fail_other_forward_with_def.carbon

package Other library "[[@TEST_NAME]]";

import Test library "forward_with_def";

// CHECK:STDERR: fail_other_forward_with_def.carbon:[[@LINE+4]]:13: error: member name `ForwardWithDef` not found in `Test` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: var f: () = Test.ForwardWithDef();
// CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var f: () = Test.ForwardWithDef();

// --- forward.impl.carbon

impl package Test library "[[@TEST_NAME]]";

var f: () = Forward();

fn Forward() {}

// --- fail_local_forward.carbon

package Test library "[[@TEST_NAME]]";

import library "forward";

// CHECK:STDERR: fail_local_forward.carbon:[[@LINE+4]]:13: error: name `Forward` not found [NameNotFound]
// CHECK:STDERR: var f: () = Forward();
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
var f: () = Forward();

// --- fail_other_forward.carbon

package Other library "[[@TEST_NAME]]";

import Test library "forward";

// CHECK:STDERR: fail_other_forward.carbon:[[@LINE+4]]:13: error: member name `Forward` not found in `Test` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: var f: () = Test.Forward();
// CHECK:STDERR:             ^~~~~~~~~~~~
// CHECK:STDERR:
var f: () = Test.Forward();

// --- todo_fail_private_on_redecl.carbon

library "[[@TEST_NAME]]";

private fn Redecl();

private fn Redecl() {}

// CHECK:STDOUT: --- def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Def.type: type = fn_type @Def [concrete]
// CHECK:STDOUT:   %Def: %Def.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Def [private] = %Def.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Def.decl: %Def.type = fn_decl @Def [concrete = constants.%Def] {} {}
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
// CHECK:STDOUT:   %ForwardWithDef.type: type = fn_type @ForwardWithDef [concrete]
// CHECK:STDOUT:   %ForwardWithDef: %ForwardWithDef.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ForwardWithDef [private] = %ForwardWithDef.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ForwardWithDef.decl.loc4: %ForwardWithDef.type = fn_decl @ForwardWithDef [concrete = constants.%ForwardWithDef] {} {}
// CHECK:STDOUT:   %ForwardWithDef.decl.loc6: %ForwardWithDef.type = fn_decl @ForwardWithDef [concrete = constants.%ForwardWithDef] {} {}
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
// CHECK:STDOUT:   %Forward.type: type = fn_type @Forward [concrete]
// CHECK:STDOUT:   %Forward: %Forward.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Forward [private] = %Forward.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Forward.decl: %Forward.type = fn_decl @Forward [concrete = constants.%Forward] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Forward();
// CHECK:STDOUT:
// CHECK:STDOUT: --- def.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Def.type: type = fn_type @Def [concrete]
// CHECK:STDOUT:   %Def: %Def.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Test.Def: %Def.type = import_ref Test//def, Def, loaded [concrete = constants.%Def]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Def [private] = imports.%Test.Def
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %empty_tuple.type = binding_pattern f
// CHECK:STDOUT:     %.loc4_1: %empty_tuple.type = var_pattern %f.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f.var: ref %empty_tuple.type = var f
// CHECK:STDOUT:   %.loc4_9.1: type = splice_block %.loc4_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc4_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_9.3: type = converted %.loc4_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f: ref %empty_tuple.type = bind_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Def() [from "def.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Def.ref: %Def.type = name_ref Def, imports.%Test.Def [concrete = constants.%Def]
// CHECK:STDOUT:   %Def.call: init %empty_tuple.type = call %Def.ref()
// CHECK:STDOUT:   assign file.%f.var, %Def.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:     .Def = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %empty_tuple.type = binding_pattern f
// CHECK:STDOUT:     %.loc10_1: %empty_tuple.type = var_pattern %f.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f.var: ref %empty_tuple.type = var f
// CHECK:STDOUT:   %.loc10_9.1: type = splice_block %.loc10_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc10_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_9.3: type = converted %.loc10_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f: ref %empty_tuple.type = bind_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Def.ref: <error> = name_ref Def, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%f.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test: <namespace> = namespace file.%Test.import, [concrete] {
// CHECK:STDOUT:     .Def = <poisoned>
// CHECK:STDOUT:     import Test//def
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Test = imports.%Test
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %empty_tuple.type = binding_pattern f
// CHECK:STDOUT:     %.loc10_1: %empty_tuple.type = var_pattern %f.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f.var: ref %empty_tuple.type = var f
// CHECK:STDOUT:   %.loc10_9.1: type = splice_block %.loc10_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc10_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_9.3: type = converted %.loc10_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f: ref %empty_tuple.type = bind_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref: <namespace> = name_ref Test, imports.%Test [concrete = imports.%Test]
// CHECK:STDOUT:   %Def.ref: <error> = name_ref Def, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%f.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- forward_with_def.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ForwardWithDef.type: type = fn_type @ForwardWithDef [concrete]
// CHECK:STDOUT:   %ForwardWithDef: %ForwardWithDef.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Test.ForwardWithDef: %ForwardWithDef.type = import_ref Test//forward_with_def, ForwardWithDef, loaded [concrete = constants.%ForwardWithDef]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .ForwardWithDef [private] = imports.%Test.ForwardWithDef
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %empty_tuple.type = binding_pattern f
// CHECK:STDOUT:     %.loc4_1: %empty_tuple.type = var_pattern %f.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f.var: ref %empty_tuple.type = var f
// CHECK:STDOUT:   %.loc4_9.1: type = splice_block %.loc4_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc4_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_9.3: type = converted %.loc4_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f: ref %empty_tuple.type = bind_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ForwardWithDef() [from "forward_with_def.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ForwardWithDef.ref: %ForwardWithDef.type = name_ref ForwardWithDef, imports.%Test.ForwardWithDef [concrete = constants.%ForwardWithDef]
// CHECK:STDOUT:   %ForwardWithDef.call: init %empty_tuple.type = call %ForwardWithDef.ref()
// CHECK:STDOUT:   assign file.%f.var, %ForwardWithDef.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_forward_with_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:     .ForwardWithDef = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %empty_tuple.type = binding_pattern f
// CHECK:STDOUT:     %.loc10_1: %empty_tuple.type = var_pattern %f.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f.var: ref %empty_tuple.type = var f
// CHECK:STDOUT:   %.loc10_9.1: type = splice_block %.loc10_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc10_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_9.3: type = converted %.loc10_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f: ref %empty_tuple.type = bind_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ForwardWithDef.ref: <error> = name_ref ForwardWithDef, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%f.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other_forward_with_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test: <namespace> = namespace file.%Test.import, [concrete] {
// CHECK:STDOUT:     .ForwardWithDef = <poisoned>
// CHECK:STDOUT:     import Test//forward_with_def
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Test = imports.%Test
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %empty_tuple.type = binding_pattern f
// CHECK:STDOUT:     %.loc10_1: %empty_tuple.type = var_pattern %f.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f.var: ref %empty_tuple.type = var f
// CHECK:STDOUT:   %.loc10_9.1: type = splice_block %.loc10_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc10_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_9.3: type = converted %.loc10_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f: ref %empty_tuple.type = bind_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref: <namespace> = name_ref Test, imports.%Test [concrete = imports.%Test]
// CHECK:STDOUT:   %ForwardWithDef.ref: <error> = name_ref ForwardWithDef, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%f.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- forward.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Forward.type: type = fn_type @Forward [concrete]
// CHECK:STDOUT:   %Forward: %Forward.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Test.Forward: %Forward.type = import_ref Test//forward, Forward, loaded [concrete = constants.%Forward]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Forward [private] = %Forward.decl
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %empty_tuple.type = binding_pattern f
// CHECK:STDOUT:     %.loc4_1: %empty_tuple.type = var_pattern %f.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f.var: ref %empty_tuple.type = var f
// CHECK:STDOUT:   %.loc4_9.1: type = splice_block %.loc4_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc4_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_9.3: type = converted %.loc4_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f: ref %empty_tuple.type = bind_name f, %f.var
// CHECK:STDOUT:   %Forward.decl: %Forward.type = fn_decl @Forward [concrete = constants.%Forward] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Forward() [from "forward.carbon"] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Forward.ref: %Forward.type = name_ref Forward, imports.%Test.Forward [concrete = constants.%Forward]
// CHECK:STDOUT:   %Forward.call: init %empty_tuple.type = call %Forward.ref()
// CHECK:STDOUT:   assign file.%f.var, %Forward.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_forward.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:     .Forward = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %empty_tuple.type = binding_pattern f
// CHECK:STDOUT:     %.loc10_1: %empty_tuple.type = var_pattern %f.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f.var: ref %empty_tuple.type = var f
// CHECK:STDOUT:   %.loc10_9.1: type = splice_block %.loc10_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc10_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_9.3: type = converted %.loc10_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f: ref %empty_tuple.type = bind_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Forward.ref: <error> = name_ref Forward, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%f.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other_forward.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test: <namespace> = namespace file.%Test.import, [concrete] {
// CHECK:STDOUT:     .Forward = <poisoned>
// CHECK:STDOUT:     import Test//forward
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Test = imports.%Test
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %empty_tuple.type = binding_pattern f
// CHECK:STDOUT:     %.loc10_1: %empty_tuple.type = var_pattern %f.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f.var: ref %empty_tuple.type = var f
// CHECK:STDOUT:   %.loc10_9.1: type = splice_block %.loc10_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc10_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_9.3: type = converted %.loc10_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f: ref %empty_tuple.type = bind_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref: <namespace> = name_ref Test, imports.%Test [concrete = imports.%Test]
// CHECK:STDOUT:   %Forward.ref: <error> = name_ref Forward, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%f.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_private_on_redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Redecl.type: type = fn_type @Redecl [concrete]
// CHECK:STDOUT:   %Redecl: %Redecl.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Redecl [private] = %Redecl.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Redecl.decl.loc4: %Redecl.type = fn_decl @Redecl [concrete = constants.%Redecl] {} {}
// CHECK:STDOUT:   %Redecl.decl.loc6: %Redecl.type = fn_decl @Redecl [concrete = constants.%Redecl] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Redecl() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
