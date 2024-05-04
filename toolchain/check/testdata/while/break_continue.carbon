// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A() -> bool;
fn B() -> bool;
fn C() -> bool;
fn D() -> bool;
fn E() -> bool;
fn F() -> bool;
fn G() -> bool;
fn H() -> bool;

fn While() {
  while (A()) {
    if (B()) { continue; }
    if (C()) { break; }
    while (D()) {
      if (E()) { continue; }
      if (F()) { break; }
    }
    if (G()) { continue; }
    if (H()) { break; }
  }
}

// CHECK:STDOUT: --- break_continue.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.2: B = struct_value () [template]
// CHECK:STDOUT:   %C: type = fn_type @C [template]
// CHECK:STDOUT:   %struct.3: C = struct_value () [template]
// CHECK:STDOUT:   %D: type = fn_type @D [template]
// CHECK:STDOUT:   %struct.4: D = struct_value () [template]
// CHECK:STDOUT:   %E: type = fn_type @E [template]
// CHECK:STDOUT:   %struct.5: E = struct_value () [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.6: F = struct_value () [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.7: G = struct_value () [template]
// CHECK:STDOUT:   %H: type = fn_type @H [template]
// CHECK:STDOUT:   %struct.8: H = struct_value () [template]
// CHECK:STDOUT:   %While: type = fn_type @While [template]
// CHECK:STDOUT:   %struct.9: While = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:     .While = %While.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct.1] {
// CHECK:STDOUT:     @A.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: B = fn_decl @B [template = constants.%struct.2] {
// CHECK:STDOUT:     @B.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: C = fn_decl @C [template = constants.%struct.3] {
// CHECK:STDOUT:     @C.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: D = fn_decl @D [template = constants.%struct.4] {
// CHECK:STDOUT:     @D.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.decl: E = fn_decl @E [template = constants.%struct.5] {
// CHECK:STDOUT:     @E.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.6] {
// CHECK:STDOUT:     @F.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.7] {
// CHECK:STDOUT:     @G.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: H = fn_decl @H [template = constants.%struct.8] {
// CHECK:STDOUT:     @H.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %While.decl: While = fn_decl @While [template = constants.%struct.9] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @While() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   br !while.cond.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond.loc17:
// CHECK:STDOUT:   %A.ref: A = name_ref A, file.%A.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %A.call: init bool = call %A.ref()
// CHECK:STDOUT:   %.loc17_13.1: bool = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc17_13.2: bool = converted %A.call, %.loc17_13.1
// CHECK:STDOUT:   if %.loc17_13.2 br !while.body.loc17 else br !while.done.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body.loc17:
// CHECK:STDOUT:   %B.ref: B = name_ref B, file.%B.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %B.call: init bool = call %B.ref()
// CHECK:STDOUT:   %.loc18_12.1: bool = value_of_initializer %B.call
// CHECK:STDOUT:   %.loc18_12.2: bool = converted %B.call, %.loc18_12.1
// CHECK:STDOUT:   if %.loc18_12.2 br !if.then.loc18 else br !if.else.loc18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc18:
// CHECK:STDOUT:   br !while.cond.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc18:
// CHECK:STDOUT:   %C.ref: C = name_ref C, file.%C.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %C.call: init bool = call %C.ref()
// CHECK:STDOUT:   %.loc19_12.1: bool = value_of_initializer %C.call
// CHECK:STDOUT:   %.loc19_12.2: bool = converted %C.call, %.loc19_12.1
// CHECK:STDOUT:   if %.loc19_12.2 br !if.then.loc19 else br !if.else.loc19
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc19:
// CHECK:STDOUT:   br !while.done.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc19:
// CHECK:STDOUT:   br !while.cond.loc20
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond.loc20:
// CHECK:STDOUT:   %D.ref: D = name_ref D, file.%D.decl [template = constants.%struct.4]
// CHECK:STDOUT:   %D.call: init bool = call %D.ref()
// CHECK:STDOUT:   %.loc20_15.1: bool = value_of_initializer %D.call
// CHECK:STDOUT:   %.loc20_15.2: bool = converted %D.call, %.loc20_15.1
// CHECK:STDOUT:   if %.loc20_15.2 br !while.body.loc20 else br !while.done.loc20
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body.loc20:
// CHECK:STDOUT:   %E.ref: E = name_ref E, file.%E.decl [template = constants.%struct.5]
// CHECK:STDOUT:   %E.call: init bool = call %E.ref()
// CHECK:STDOUT:   %.loc21_14.1: bool = value_of_initializer %E.call
// CHECK:STDOUT:   %.loc21_14.2: bool = converted %E.call, %.loc21_14.1
// CHECK:STDOUT:   if %.loc21_14.2 br !if.then.loc21 else br !if.else.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc21:
// CHECK:STDOUT:   br !while.cond.loc20
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc21:
// CHECK:STDOUT:   %F.ref: F = name_ref F, file.%F.decl [template = constants.%struct.6]
// CHECK:STDOUT:   %F.call: init bool = call %F.ref()
// CHECK:STDOUT:   %.loc22_14.1: bool = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc22_14.2: bool = converted %F.call, %.loc22_14.1
// CHECK:STDOUT:   if %.loc22_14.2 br !if.then.loc22 else br !if.else.loc22
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc22:
// CHECK:STDOUT:   br !while.done.loc20
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc22:
// CHECK:STDOUT:   br !while.cond.loc20
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done.loc20:
// CHECK:STDOUT:   %G.ref: G = name_ref G, file.%G.decl [template = constants.%struct.7]
// CHECK:STDOUT:   %G.call: init bool = call %G.ref()
// CHECK:STDOUT:   %.loc24_12.1: bool = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc24_12.2: bool = converted %G.call, %.loc24_12.1
// CHECK:STDOUT:   if %.loc24_12.2 br !if.then.loc24 else br !if.else.loc24
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc24:
// CHECK:STDOUT:   br !while.cond.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc24:
// CHECK:STDOUT:   %H.ref: H = name_ref H, file.%H.decl [template = constants.%struct.8]
// CHECK:STDOUT:   %H.call: init bool = call %H.ref()
// CHECK:STDOUT:   %.loc25_12.1: bool = value_of_initializer %H.call
// CHECK:STDOUT:   %.loc25_12.2: bool = converted %H.call, %.loc25_12.1
// CHECK:STDOUT:   if %.loc25_12.2 br !if.then.loc25 else br !if.else.loc25
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc25:
// CHECK:STDOUT:   br !while.done.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc25:
// CHECK:STDOUT:   br !while.cond.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done.loc17:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
