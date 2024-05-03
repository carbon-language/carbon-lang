// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon

library "a" api;

class Cycle {
  var a: Cycle*;
}

// --- b.carbon

library "b" api;

import library "a";

fn Run() {
  var a: Cycle*;
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cycle: type = class_type @Cycle [template]
// CHECK:STDOUT:   %.1: type = ptr_type Cycle [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type Cycle, Cycle* [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: Cycle*} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Cycle = %Cycle.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Cycle.decl: type = class_decl @Cycle [template = constants.%Cycle] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Cycle {
// CHECK:STDOUT:   %Cycle.ref: type = name_ref Cycle, file.%Cycle.decl [template = constants.%Cycle]
// CHECK:STDOUT:   %.loc5_15: type = ptr_type Cycle [template = constants.%.1]
// CHECK:STDOUT:   %.loc5_8: <unbound element of class Cycle> = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Cycle
// CHECK:STDOUT:   .a = %.loc5_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run: type = fn_type @Run [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Run = struct_value () [template]
// CHECK:STDOUT:   %Cycle: type = class_type @Cycle [template]
// CHECK:STDOUT:   %.2: type = ptr_type Cycle [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: Cycle*} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Cycle = %import_ref.1
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+2, loaded [template = constants.%Cycle]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Run.decl: Run = fn_decl @Run [template = constants.%struct] {}
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+8, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+3, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Cycle {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cycle.ref: type = name_ref Cycle, file.%import_ref.1 [template = constants.%Cycle]
// CHECK:STDOUT:   %.loc7: type = ptr_type Cycle [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref Cycle* = var a
// CHECK:STDOUT:   %a: ref Cycle* = bind_name a, %a.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
