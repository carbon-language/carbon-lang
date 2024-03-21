// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon

library "a" api;

class Cycle;

var a: {.b: Cycle*};

class Cycle {
  // The type here is equivalent to the `a` above, but on import can be resolved first.
  var c: {.b: Cycle*};
}

// --- b.carbon

library "b" api;

import library "a";

fn Run() {
  a.b = (*a.b).c.b;
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cycle: type = class_type @Cycle [template]
// CHECK:STDOUT:   %.1: type = ptr_type Cycle [template]
// CHECK:STDOUT:   %.2: type = struct_type {.b: Cycle*} [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type Cycle, {.b: Cycle*} [template]
// CHECK:STDOUT:   %.4: type = struct_type {.c: {.b: Cycle*}} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Cycle = %Cycle.decl.loc4
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cycle.decl.loc4: type = class_decl @Cycle [template = constants.%Cycle] {}
// CHECK:STDOUT:   %Cycle.ref: type = name_ref Cycle, %Cycle.decl.loc4 [template = constants.%Cycle]
// CHECK:STDOUT:   %.loc6_18: type = ptr_type Cycle [template = constants.%.1]
// CHECK:STDOUT:   %.loc6_19: type = struct_type {.b: Cycle*} [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref {.b: Cycle*} = var a
// CHECK:STDOUT:   %a: ref {.b: Cycle*} = bind_name a, %a.var
// CHECK:STDOUT:   %Cycle.decl.loc8: type = class_decl @Cycle [template = constants.%Cycle] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Cycle {
// CHECK:STDOUT:   %Cycle.ref: type = name_ref Cycle, file.%Cycle.decl.loc4 [template = constants.%Cycle]
// CHECK:STDOUT:   %.loc10_20: type = ptr_type Cycle [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_21: type = struct_type {.b: Cycle*} [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_8: <unbound element of class Cycle> = field_decl c, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Cycle
// CHECK:STDOUT:   .c = %.loc10_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cycle: type = class_type @Cycle [template]
// CHECK:STDOUT:   %.1: type = ptr_type Cycle [template]
// CHECK:STDOUT:   %.2: type = struct_type {.b: Cycle*} [template]
// CHECK:STDOUT:   %.3: type = struct_type {.c: {.b: Cycle*}} [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.c: {.b: Cycle*}} [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type Cycle, {.b: Cycle*} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Cycle = %import_ref.1
// CHECK:STDOUT:     .a = %import_ref.2
// CHECK:STDOUT:     .Run = %Run
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+1, unused
// CHECK:STDOUT:   %import_ref.2: ref {.b: Cycle*} = import_ref ir1, inst+11, used
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run [template] {}
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+2, unused
// CHECK:STDOUT:   %import_ref.4: <unbound element of class Cycle> = import_ref ir1, inst+18, used [template = imports.%.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Cycle {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT:   .c = file.%import_ref.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cycle.decl: invalid = class_decl @Cycle [template = constants.%Cycle] {}
// CHECK:STDOUT:   %a.ref.loc7_3: ref {.b: Cycle*} = name_ref a, file.%import_ref.2
// CHECK:STDOUT:   %.loc7_4: ref Cycle* = struct_access %a.ref.loc7_3, element0
// CHECK:STDOUT:   %a.ref.loc7_11: ref {.b: Cycle*} = name_ref a, file.%import_ref.2
// CHECK:STDOUT:   %.loc7_12.1: ref Cycle* = struct_access %a.ref.loc7_11, element0
// CHECK:STDOUT:   %.loc7_12.2: Cycle* = bind_value %.loc7_12.1
// CHECK:STDOUT:   %.loc7_10: ref Cycle = deref %.loc7_12.2
// CHECK:STDOUT:   %c.ref: <unbound element of class Cycle> = name_ref c, file.%import_ref.4 [template = imports.%.1]
// CHECK:STDOUT:   %.loc7_15: ref {.b: Cycle*} = class_element_access %.loc7_10, element0
// CHECK:STDOUT:   %.loc7_17.1: ref Cycle* = struct_access %.loc7_15, element0
// CHECK:STDOUT:   %.loc7_17.2: Cycle* = bind_value %.loc7_17.1
// CHECK:STDOUT:   assign %.loc7_4, %.loc7_17.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
