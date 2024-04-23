// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon

library "a" api;

class C {}

// --- b.carbon

library "b" api;

import library "a";

alias D = C;

var c1: C = {};
var d1: D* = &c1;

// --- fail_c.carbon

library "c" api;

import library "a";
import library "b";

var c2: C = {};
// CHECK:STDERR: fail_c.carbon:[[@LINE+3]]:1: ERROR: Cannot implicitly convert from `C*` to `C*`.
// CHECK:STDERR: var d2: D* = &c2;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
var d2: D* = &c2;

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
// CHECK:STDOUT:   %.4: C = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .c1 = %c1
// CHECK:STDOUT:     .d1 = %d1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+2, loc_12 [template = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+3, unloaded
// CHECK:STDOUT:   %C.ref.loc6: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %D: type = bind_alias D, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c1.var: ref C = var c1
// CHECK:STDOUT:   %c1: ref C = bind_name c1, %c1.var
// CHECK:STDOUT:   %D.ref: type = name_ref D, %D [template = constants.%C]
// CHECK:STDOUT:   %.loc9: type = ptr_type C [template = constants.%.5]
// CHECK:STDOUT:   %d1.var: ref C* = var d1
// CHECK:STDOUT:   %d1: ref C* = bind_name d1, %d1.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_14.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc8_14.2: init C = class_init (), file.%c1.var [template = constants.%.4]
// CHECK:STDOUT:   %.loc8_15: init C = converted %.loc8_14.1, %.loc8_14.2 [template = constants.%.4]
// CHECK:STDOUT:   assign file.%c1.var, %.loc8_15
// CHECK:STDOUT:   %c1.ref: ref C = name_ref c1, file.%c1
// CHECK:STDOUT:   %.loc9: C* = addr_of %c1.ref
// CHECK:STDOUT:   assign file.%d1.var, %.loc9
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_c.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.1: type = class_type @C.1 [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: C = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C.2 [template]
// CHECK:STDOUT:   %.5: type = ptr_type C [template]
// CHECK:STDOUT:   %.6: type = ptr_type C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .D = %import_ref.2
// CHECK:STDOUT:     .c1 = %import_ref.3
// CHECK:STDOUT:     .d1 = %import_ref.4
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .c2 = %c2
// CHECK:STDOUT:     .d2 = %d2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+2, loc_15 [template = constants.%C.1]
// CHECK:STDOUT:   %import_ref.2: type = import_ref ir2, inst+8, loc_23 [template = constants.%C.2]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir2, inst+23, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl.1: type = class_decl @C.1 [template = constants.%C.1] {}
// CHECK:STDOUT:   %import_ref.5 = import_ref ir1, inst+3, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C.1]
// CHECK:STDOUT:   %c2.var: ref C = var c2
// CHECK:STDOUT:   %c2: ref C = bind_name c2, %c2.var
// CHECK:STDOUT:   %C.decl.2: type = class_decl @C.2 [template = constants.%C.2] {}
// CHECK:STDOUT:   %import_ref.6 = import_ref ir2, inst+6, unloaded
// CHECK:STDOUT:   %D.ref: type = name_ref D, %import_ref.2 [template = constants.%C.2]
// CHECK:STDOUT:   %.loc11: type = ptr_type C [template = constants.%.5]
// CHECK:STDOUT:   %d2.var: ref C* = var d2
// CHECK:STDOUT:   %d2: ref C* = bind_name d2, %d2.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.1 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.2 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_14.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc7_14.2: init C = class_init (), file.%c2.var [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_15: init C = converted %.loc7_14.1, %.loc7_14.2 [template = constants.%.4]
// CHECK:STDOUT:   assign file.%c2.var, %.loc7_15
// CHECK:STDOUT:   %c2.ref: ref C = name_ref c2, file.%c2
// CHECK:STDOUT:   %.loc11: C* = addr_of %c2.ref
// CHECK:STDOUT:   assign file.%d2.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
