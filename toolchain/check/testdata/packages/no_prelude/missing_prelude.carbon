// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/no_prelude/missing_prelude.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/no_prelude/missing_prelude.carbon

// --- fail_missing_prelude.carbon

library "fail_missing_prelude";

// CHECK:STDERR: fail_missing_prelude.carbon:[[@LINE+4]]:8: ERROR: Package `Core` implicitly referenced here, but not found.
// CHECK:STDERR: var n: i32;
// CHECK:STDERR:        ^~~
// CHECK:STDERR:
var n: i32;

// --- prelude_empty.carbon

package Core library "empty";

// --- fail_missing_prelude_member.carbon

library "fail_missing_prelude_member";

import Core library "empty";

// CHECK:STDERR: fail_missing_prelude_member.carbon:[[@LINE+4]]:8: ERROR: Name `Core.Int32` implicitly referenced here, but not found.
// CHECK:STDERR: var n: i32;
// CHECK:STDERR:        ^~~
// CHECK:STDERR:
var n: i32;

// --- prelude_fake_int.carbon

package Core library "fake_int";

fn Int32() -> {} { return {}; }

// --- use_fake_int.carbon

library "use_fake_int";

import Core library "fake_int";

var n: {} = i32;

// --- prelude_use_in_prelude.carbon

package Core library "prelude_use_in_prelude";

// Core is not an imported package here.

// Note that we are intentionally using a different type here from other cases
// to test that this function is called, not the one from prelude_fake_int.
fn Int32() -> () { return (); }

var n: () = i32;

// --- prelude_as_namespace.carbon

library "prelude_as_namespace";

// TODO: Decide whether we want to accept this.
namespace Core;

fn Core.Int32() -> {} { return {}; }

var n: {} = i32;

// --- fail_prelude_as_class.carbon

library "prelude_as_class";

// TODO: Decide whether we want to accept this.
class Core {
  fn Int32() -> {} { return {}; }
}

// CHECK:STDERR: fail_prelude_as_class.carbon:[[@LINE+3]]:13: ERROR: Package `Core` implicitly referenced here, but not found.
// CHECK:STDERR: var n: {} = i32;
// CHECK:STDERR:             ^~~
var n: {} = i32;

// CHECK:STDOUT: --- fail_missing_prelude.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- prelude_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_missing_prelude_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//empty
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- prelude_fake_int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %struct: %.1 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int32 = %Int32.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32.decl: %Int32.type = fn_decl @Int32 [template = constants.%Int32] {
// CHECK:STDOUT:     %.loc4_16.1: %.1 = struct_literal ()
// CHECK:STDOUT:     %.loc4_16.2: type = converted %.loc4_16.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @Int32.%return: ref %.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_28: %.1 = struct_literal ()
// CHECK:STDOUT:   %struct: %.1 = struct_value () [template = constants.%struct]
// CHECK:STDOUT:   %.loc4_29: %.1 = converted %.loc4_28, %struct [template = constants.%struct]
// CHECK:STDOUT:   return %.loc4_29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_fake_int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//fake_int
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//fake_int, inst+5, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc6_9.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %n.var: ref %.1 = var n
// CHECK:STDOUT:   %n: ref %.1 = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> %.1;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32.call: init %.1 = call constants.%Int32()
// CHECK:STDOUT:   assign file.%n.var, %Int32.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- prelude_use_in_prelude.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int32 = %Int32.decl
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32.decl: %Int32.type = fn_decl @Int32 [template = constants.%Int32] {
// CHECK:STDOUT:     %.loc8_16.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc8_16.2: type = converted %.loc8_16.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @Int32.%return: ref %.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc10_9.2: type = converted %.loc10_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %n.var: ref %.1 = var n
// CHECK:STDOUT:   %n: ref %.1 = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_28: %.1 = tuple_literal ()
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc8_29: %.1 = converted %.loc8_28, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc8_29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32.call: init %.1 = call constants.%Int32()
// CHECK:STDOUT:   assign file.%n.var, %Int32.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- prelude_as_namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %struct: %.1 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int32 = %Int32.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32.decl: %Int32.type = fn_decl @Int32 [template = constants.%Int32] {
// CHECK:STDOUT:     %.loc7_21.1: %.1 = struct_literal ()
// CHECK:STDOUT:     %.loc7_21.2: type = converted %.loc7_21.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @Int32.%return: ref %.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_9.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc9_9.2: type = converted %.loc9_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %n.var: ref %.1 = var n
// CHECK:STDOUT:   %n: ref %.1 = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_33: %.1 = struct_literal ()
// CHECK:STDOUT:   %struct: %.1 = struct_value () [template = constants.%struct]
// CHECK:STDOUT:   %.loc7_34: %.1 = converted %.loc7_33, %struct [template = constants.%struct]
// CHECK:STDOUT:   return %.loc7_34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32.call: init %.1 = call constants.%Int32()
// CHECK:STDOUT:   assign file.%n.var, %Int32.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_prelude_as_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Core: type = class_type @Core [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %struct: %.1 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core.decl
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.decl: type = class_decl @Core [template = constants.%Core] {}
// CHECK:STDOUT:   %.loc12_9.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc12_9.2: type = converted %.loc12_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %n.var: ref %.1 = var n
// CHECK:STDOUT:   %n: ref %.1 = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Core {
// CHECK:STDOUT:   %Int32.decl: %Int32.type = fn_decl @Int32 [template = constants.%Int32] {
// CHECK:STDOUT:     %.loc6_18.1: %.1 = struct_literal ()
// CHECK:STDOUT:     %.loc6_18.2: type = converted %.loc6_18.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %return.var: ref %.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Core
// CHECK:STDOUT:   .Int32 = %Int32.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_30: %.1 = struct_literal ()
// CHECK:STDOUT:   %struct: %.1 = struct_value () [template = constants.%struct]
// CHECK:STDOUT:   %.loc6_31: %.1 = converted %.loc6_30, %struct [template = constants.%struct]
// CHECK:STDOUT:   return %.loc6_31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   assign file.%n.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
