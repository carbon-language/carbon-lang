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
  a.b;
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
// CHECK:STDOUT:   package: <namespace> = namespace {.Cycle = %Cycle.decl.loc4, .a = %a} [template]
// CHECK:STDOUT:   %Cycle.decl.loc4 = class_decl @Cycle, ()
// CHECK:STDOUT:   %Cycle.ref: type = name_ref Cycle, constants.%Cycle [template = constants.%Cycle]
// CHECK:STDOUT:   %.loc6_18: type = ptr_type Cycle [template = constants.%.1]
// CHECK:STDOUT:   %.loc6_19: type = struct_type {.b: Cycle*} [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref {.b: Cycle*} = var a
// CHECK:STDOUT:   %a: ref {.b: Cycle*} = bind_name a, %a.var
// CHECK:STDOUT:   %Cycle.decl.loc8 = class_decl @Cycle, ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Cycle {
// CHECK:STDOUT:   %Cycle.ref: type = name_ref Cycle, constants.%Cycle [template = constants.%Cycle]
// CHECK:STDOUT:   %.loc10_20: type = ptr_type Cycle [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_21: type = struct_type {.b: Cycle*} [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_8: <unbound element of class Cycle> = field_decl c, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Cycle = %import_ref.1, .a = %import_ref.2, .Run = %Run} [template]
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+1, unused
// CHECK:STDOUT:   %import_ref.2: ref {.b: Cycle*} = import_ref ir1, inst+11, used
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Cycle {
// CHECK:STDOUT:   %import_ref = import_ref ir1, inst+18, unused
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .c = %import_ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cycle.decl = class_decl @Cycle, ()
// CHECK:STDOUT:   %a.ref: ref {.b: Cycle*} = name_ref a, file.%import_ref.2
// CHECK:STDOUT:   %.loc7: ref Cycle* = struct_access %a.ref, element0
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
