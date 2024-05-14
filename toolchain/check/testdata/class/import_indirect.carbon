// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// Triangle:
//
//     a    <-- Define
//     |\
//     | b  <-- Re-export
//     |/
//     X    <-- Use both
//
// Diamond:
//
//     a    <-- Define
//    / \
//   b   c  <-- Re-export
//    \ /
//     X    <-- Use both

// ============================================================================
// Setup files
// ============================================================================

// --- a.carbon

library "a" api;

class C {}

// --- b.carbon

library "b" api;

import library "a";

alias D = C;

var b_val: C = {};
var b_ptr: D* = &b_val;

// --- c.carbon

library "c" api;

import library "a";

alias E = C;

var c_val: C = {};
var c_ptr: E* = &c_val;

// ============================================================================
// Test files
// ============================================================================

// --- triangle.carbon

library "triangle" api;

import library "a";
import library "b";

var val: C = {};
var ptr: D* = &val;

// --- triangle_reverse.carbon

library "triangle_reverse" api;

import library "b";
import library "a";

var val: C = {};
var ptr: D* = &val;

// --- diamond.carbon

library "diamond" api;

import library "b";
import library "c";

var val: D = {};
var ptr: E* = &val;

// --- diamond_reverse.carbon

library "diamond_reverse" api;

import library "c";
import library "b";

var val: D = {};
var ptr: E* = &val;

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %struct: C = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .b_val = %b_val
// CHECK:STDOUT:     .b_ptr = %b_ptr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+2, loaded [template = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+3, unloaded
// CHECK:STDOUT:   %C.ref.loc6: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %D: type = bind_alias D, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %b_val.var: ref C = var b_val
// CHECK:STDOUT:   %b_val: ref C = bind_name b_val, %b_val.var
// CHECK:STDOUT:   %D.ref: type = name_ref D, %D [template = constants.%C]
// CHECK:STDOUT:   %.loc9: type = ptr_type C [template = constants.%.4]
// CHECK:STDOUT:   %b_ptr.var: ref C* = var b_ptr
// CHECK:STDOUT:   %b_ptr: ref C* = bind_name b_ptr, %b_ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_17.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc8_17.2: init C = class_init (), file.%b_val.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc8_18: init C = converted %.loc8_17.1, %.loc8_17.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%b_val.var, %.loc8_18
// CHECK:STDOUT:   %b_val.ref: ref C = name_ref b_val, file.%b_val
// CHECK:STDOUT:   %.loc9: C* = addr_of %b_val.ref
// CHECK:STDOUT:   assign file.%b_ptr.var, %.loc9
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- c.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %struct: C = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .E = %E
// CHECK:STDOUT:     .c_val = %c_val
// CHECK:STDOUT:     .c_ptr = %c_ptr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+2, loaded [template = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+3, unloaded
// CHECK:STDOUT:   %C.ref.loc6: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %E: type = bind_alias E, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c_val.var: ref C = var c_val
// CHECK:STDOUT:   %c_val: ref C = bind_name c_val, %c_val.var
// CHECK:STDOUT:   %E.ref: type = name_ref E, %E [template = constants.%C]
// CHECK:STDOUT:   %.loc9: type = ptr_type C [template = constants.%.4]
// CHECK:STDOUT:   %c_ptr.var: ref C* = var c_ptr
// CHECK:STDOUT:   %c_ptr: ref C* = bind_name c_ptr, %c_ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_17.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc8_17.2: init C = class_init (), file.%c_val.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc8_18: init C = converted %.loc8_17.1, %.loc8_17.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c_val.var, %.loc8_18
// CHECK:STDOUT:   %c_val.ref: ref C = name_ref c_val, file.%c_val
// CHECK:STDOUT:   %.loc9: C* = addr_of %c_val.ref
// CHECK:STDOUT:   assign file.%c_ptr.var, %.loc9
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- triangle.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %struct: C = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .D = %import_ref.2
// CHECK:STDOUT:     .b_val = %import_ref.3
// CHECK:STDOUT:     .b_ptr = %import_ref.4
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .val = %val
// CHECK:STDOUT:     .ptr = %ptr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+2, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: type = import_ref ir2, inst+8, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir2, inst+23, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.5 = import_ref ir1, inst+3, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %val.var: ref C = var val
// CHECK:STDOUT:   %val: ref C = bind_name val, %val.var
// CHECK:STDOUT:   %D.ref: type = name_ref D, %import_ref.2 [template = constants.%C]
// CHECK:STDOUT:   %.loc8: type = ptr_type C [template = constants.%.4]
// CHECK:STDOUT:   %ptr.var: ref C* = var ptr
// CHECK:STDOUT:   %ptr: ref C* = bind_name ptr, %ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_15.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc7_15.2: init C = class_init (), file.%val.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc7_16: init C = converted %.loc7_15.1, %.loc7_15.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%val.var, %.loc7_16
// CHECK:STDOUT:   %val.ref: ref C = name_ref val, file.%val
// CHECK:STDOUT:   %.loc8: C* = addr_of %val.ref
// CHECK:STDOUT:   assign file.%ptr.var, %.loc8
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- triangle_reverse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %struct: C = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .D = %import_ref.1
// CHECK:STDOUT:     .b_val = %import_ref.2
// CHECK:STDOUT:     .b_ptr = %import_ref.3
// CHECK:STDOUT:     .C = %import_ref.4
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .val = %val
// CHECK:STDOUT:     .ptr = %ptr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+8, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.4: type = import_ref ir2, inst+2, loaded [template = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.5 = import_ref ir2, inst+3, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.4 [template = constants.%C]
// CHECK:STDOUT:   %val.var: ref C = var val
// CHECK:STDOUT:   %val: ref C = bind_name val, %val.var
// CHECK:STDOUT:   %D.ref: type = name_ref D, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %.loc8: type = ptr_type C [template = constants.%.4]
// CHECK:STDOUT:   %ptr.var: ref C* = var ptr
// CHECK:STDOUT:   %ptr: ref C* = bind_name ptr, %ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_15.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc7_15.2: init C = class_init (), file.%val.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc7_16: init C = converted %.loc7_15.1, %.loc7_15.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%val.var, %.loc7_16
// CHECK:STDOUT:   %val.ref: ref C = name_ref val, file.%val
// CHECK:STDOUT:   %.loc8: C* = addr_of %val.ref
// CHECK:STDOUT:   assign file.%ptr.var, %.loc8
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- diamond.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %struct: C = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .D = %import_ref.1
// CHECK:STDOUT:     .b_val = %import_ref.2
// CHECK:STDOUT:     .b_ptr = %import_ref.3
// CHECK:STDOUT:     .E = %import_ref.4
// CHECK:STDOUT:     .c_val = %import_ref.5
// CHECK:STDOUT:     .c_ptr = %import_ref.6
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .val = %val
// CHECK:STDOUT:     .ptr = %ptr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+8, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.4: type = import_ref ir2, inst+8, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.5 = import_ref ir2, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref ir2, inst+23, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.7 = import_ref ir1, inst+6, unloaded
// CHECK:STDOUT:   %D.ref: type = name_ref D, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %val.var: ref C = var val
// CHECK:STDOUT:   %val: ref C = bind_name val, %val.var
// CHECK:STDOUT:   %E.ref: type = name_ref E, %import_ref.4 [template = constants.%C]
// CHECK:STDOUT:   %.loc8: type = ptr_type C [template = constants.%.4]
// CHECK:STDOUT:   %ptr.var: ref C* = var ptr
// CHECK:STDOUT:   %ptr: ref C* = bind_name ptr, %ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_15.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc7_15.2: init C = class_init (), file.%val.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc7_16: init C = converted %.loc7_15.1, %.loc7_15.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%val.var, %.loc7_16
// CHECK:STDOUT:   %val.ref: ref C = name_ref val, file.%val
// CHECK:STDOUT:   %.loc8: C* = addr_of %val.ref
// CHECK:STDOUT:   assign file.%ptr.var, %.loc8
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- diamond_reverse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %struct: C = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .E = %import_ref.1
// CHECK:STDOUT:     .c_val = %import_ref.2
// CHECK:STDOUT:     .c_ptr = %import_ref.3
// CHECK:STDOUT:     .D = %import_ref.4
// CHECK:STDOUT:     .b_val = %import_ref.5
// CHECK:STDOUT:     .b_ptr = %import_ref.6
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .val = %val
// CHECK:STDOUT:     .ptr = %ptr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+8, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.4: type = import_ref ir2, inst+8, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.5 = import_ref ir2, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref ir2, inst+23, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.7 = import_ref ir2, inst+6, unloaded
// CHECK:STDOUT:   %D.ref: type = name_ref D, %import_ref.4 [template = constants.%C]
// CHECK:STDOUT:   %val.var: ref C = var val
// CHECK:STDOUT:   %val: ref C = bind_name val, %val.var
// CHECK:STDOUT:   %E.ref: type = name_ref E, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %.loc8: type = ptr_type C [template = constants.%.4]
// CHECK:STDOUT:   %ptr.var: ref C* = var ptr
// CHECK:STDOUT:   %ptr: ref C* = bind_name ptr, %ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_15.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc7_15.2: init C = class_init (), file.%val.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc7_16: init C = converted %.loc7_15.1, %.loc7_15.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%val.var, %.loc7_16
// CHECK:STDOUT:   %val.ref: ref C = name_ref val, file.%val
// CHECK:STDOUT:   %.loc8: C* = addr_of %val.ref
// CHECK:STDOUT:   assign file.%ptr.var, %.loc8
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
