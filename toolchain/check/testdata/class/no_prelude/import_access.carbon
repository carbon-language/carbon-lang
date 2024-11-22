// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// CHECK:STDERR: fail_other_def.carbon:[[@LINE+4]]:8: error: name `Def` not found [NameNotFound]
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

// CHECK:STDERR: fail_other_forward_with_def.carbon:[[@LINE+4]]:8: error: name `ForwardWithDef` not found [NameNotFound]
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

// CHECK:STDERR: fail_other_forward.carbon:[[@LINE+3]]:9: error: name `Forward` not found [NameNotFound]
// CHECK:STDERR: fn F(c: Test.Forward*) {}
// CHECK:STDERR:         ^~~~~~~~~~~~
fn F(c: Test.Forward*) {}

// --- todo_fail_private_on_redecl.carbon

library "[[@TEST_NAME]]";

private class Redecl;

private class Redecl {}

// CHECK:STDOUT: --- def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Def: type = class_type @Def [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Def [private] = %Def.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Def.decl: type = class_decl @Def [template = constants.%Def] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Def {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Def
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- forward_with_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ForwardWithDef: type = class_type @ForwardWithDef [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .ForwardWithDef [private] = %ForwardWithDef.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ForwardWithDef.decl.loc4: type = class_decl @ForwardWithDef [template = constants.%ForwardWithDef] {} {}
// CHECK:STDOUT:   %ForwardWithDef.decl.loc6: type = class_decl @ForwardWithDef [template = constants.%ForwardWithDef] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ForwardWithDef {
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ForwardWithDef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- forward.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Forward: type = class_type @Forward [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Forward [private] = %Forward.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Forward.decl: type = class_decl @Forward [template = constants.%Forward] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Forward;
// CHECK:STDOUT:
// CHECK:STDOUT: --- def.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Def: type = class_type @Def [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %struct: %Def = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Test//def, inst+1, loaded [template = constants.%Def]
// CHECK:STDOUT:   %import_ref.2 = import_ref Test//def, inst+2, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Def [private] = imports.%import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Def.ref: type = name_ref Def, imports.%import_ref.1 [template = constants.%Def]
// CHECK:STDOUT:   %c.var: ref %Def = var c
// CHECK:STDOUT:   %c: ref %Def = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Def {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_15.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc4_15.2: init %Def = class_init (), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc4_16: init %Def = converted %.loc4_15.1, %.loc4_15.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc4_16
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Def.ref: <error> = name_ref Def, <error> [template = <error>]
// CHECK:STDOUT:   %c.var: ref <error> = var c
// CHECK:STDOUT:   %c: ref <error> = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10: %.1 = struct_literal ()
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
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
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %Test.ref: <namespace> = name_ref Test, imports.%Test [template = imports.%Test]
// CHECK:STDOUT:   %Def.ref: <error> = name_ref Def, <error> [template = <error>]
// CHECK:STDOUT:   %c.var: ref <error> = var c
// CHECK:STDOUT:   %c: ref <error> = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10: %.1 = struct_literal ()
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- forward_with_def.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ForwardWithDef: type = class_type @ForwardWithDef [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %struct: %ForwardWithDef = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Test//forward_with_def, inst+1, loaded [template = constants.%ForwardWithDef]
// CHECK:STDOUT:   %import_ref.2 = import_ref Test//forward_with_def, inst+2, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .ForwardWithDef [private] = imports.%import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %ForwardWithDef.ref: type = name_ref ForwardWithDef, imports.%import_ref.1 [template = constants.%ForwardWithDef]
// CHECK:STDOUT:   %c.var: ref %ForwardWithDef = var c
// CHECK:STDOUT:   %c: ref %ForwardWithDef = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ForwardWithDef {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_26.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc4_26.2: init %ForwardWithDef = class_init (), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc4_27: init %ForwardWithDef = converted %.loc4_26.1, %.loc4_26.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc4_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_forward_with_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %ForwardWithDef.ref: <error> = name_ref ForwardWithDef, <error> [template = <error>]
// CHECK:STDOUT:   %c.var: ref <error> = var c
// CHECK:STDOUT:   %c: ref <error> = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10: %.1 = struct_literal ()
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other_forward_with_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
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
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %Test.ref: <namespace> = name_ref Test, imports.%Test [template = imports.%Test]
// CHECK:STDOUT:   %ForwardWithDef.ref: <error> = name_ref ForwardWithDef, <error> [template = <error>]
// CHECK:STDOUT:   %c.var: ref <error> = var c
// CHECK:STDOUT:   %c: ref <error> = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10: %.1 = struct_literal ()
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- forward.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Forward: type = class_type @Forward [template]
// CHECK:STDOUT:   %.1: type = ptr_type %Forward [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref: type = import_ref Test//forward, inst+1, loaded [template = constants.%Forward]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Forward [private] = %Forward.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %c.patt: %.1 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %.1 = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Forward.ref: type = name_ref Forward, imports.%import_ref [template = constants.%Forward]
// CHECK:STDOUT:     %.loc4: type = ptr_type %Forward [template = constants.%.1]
// CHECK:STDOUT:     %c.param: %.1 = value_param runtime_param0
// CHECK:STDOUT:     %c: %.1 = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Forward.decl: type = class_decl @Forward [template = constants.%Forward] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Forward {
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Forward
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param_patt: %.1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_forward.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %c.patt: <error> = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: <error> = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Forward.ref: <error> = name_ref Forward, <error> [template = <error>]
// CHECK:STDOUT:     %.loc10: type = ptr_type <error> [template = <error>]
// CHECK:STDOUT:     %c.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %c: <error> = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param_patt: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other_forward.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
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
// CHECK:STDOUT:     %c.patt: <error> = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: <error> = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Test.ref: <namespace> = name_ref Test, imports.%Test [template = imports.%Test]
// CHECK:STDOUT:     %Forward.ref: <error> = name_ref Forward, <error> [template = <error>]
// CHECK:STDOUT:     %.loc9: type = ptr_type <error> [template = <error>]
// CHECK:STDOUT:     %c.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %c: <error> = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param_patt: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_private_on_redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Redecl: type = class_type @Redecl [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Redecl [private] = %Redecl.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Redecl.decl.loc4: type = class_decl @Redecl [template = constants.%Redecl] {} {}
// CHECK:STDOUT:   %Redecl.decl.loc6: type = class_decl @Redecl [template = constants.%Redecl] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Redecl {
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Redecl
// CHECK:STDOUT: }
// CHECK:STDOUT:
