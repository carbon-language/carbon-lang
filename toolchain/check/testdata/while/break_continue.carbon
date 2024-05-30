// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/while/break_continue.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/while/break_continue.carbon

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
// CHECK:STDOUT:   %Bool: type = fn_type @Bool [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Bool = struct_value () [template]
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %struct.2: A = struct_value () [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.3: B = struct_value () [template]
// CHECK:STDOUT:   %C: type = fn_type @C [template]
// CHECK:STDOUT:   %struct.4: C = struct_value () [template]
// CHECK:STDOUT:   %D: type = fn_type @D [template]
// CHECK:STDOUT:   %struct.5: D = struct_value () [template]
// CHECK:STDOUT:   %E: type = fn_type @E [template]
// CHECK:STDOUT:   %struct.6: E = struct_value () [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.7: F = struct_value () [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.8: G = struct_value () [template]
// CHECK:STDOUT:   %H: type = fn_type @H [template]
// CHECK:STDOUT:   %struct.9: H = struct_value () [template]
// CHECK:STDOUT:   %While: type = fn_type @While [template]
// CHECK:STDOUT:   %struct.10: While = struct_value () [template]
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
// CHECK:STDOUT:   %import_ref.1: Bool = import_ref ir7, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct.2] {
// CHECK:STDOUT:     %bool.make_type.loc11: init type = call constants.%struct.1() [template = bool]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %bool.make_type.loc11 [template = bool]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %bool.make_type.loc11, %.loc11_11.1 [template = bool]
// CHECK:STDOUT:     @A.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: Bool = import_ref ir7, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %B.decl: B = fn_decl @B [template = constants.%struct.3] {
// CHECK:STDOUT:     %bool.make_type.loc12: init type = call constants.%struct.1() [template = bool]
// CHECK:STDOUT:     %.loc12_11.1: type = value_of_initializer %bool.make_type.loc12 [template = bool]
// CHECK:STDOUT:     %.loc12_11.2: type = converted %bool.make_type.loc12, %.loc12_11.1 [template = bool]
// CHECK:STDOUT:     @B.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: Bool = import_ref ir7, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %C.decl: C = fn_decl @C [template = constants.%struct.4] {
// CHECK:STDOUT:     %bool.make_type.loc13: init type = call constants.%struct.1() [template = bool]
// CHECK:STDOUT:     %.loc13_11.1: type = value_of_initializer %bool.make_type.loc13 [template = bool]
// CHECK:STDOUT:     %.loc13_11.2: type = converted %bool.make_type.loc13, %.loc13_11.1 [template = bool]
// CHECK:STDOUT:     @C.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.4: Bool = import_ref ir7, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %D.decl: D = fn_decl @D [template = constants.%struct.5] {
// CHECK:STDOUT:     %bool.make_type.loc14: init type = call constants.%struct.1() [template = bool]
// CHECK:STDOUT:     %.loc14_11.1: type = value_of_initializer %bool.make_type.loc14 [template = bool]
// CHECK:STDOUT:     %.loc14_11.2: type = converted %bool.make_type.loc14, %.loc14_11.1 [template = bool]
// CHECK:STDOUT:     @D.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: Bool = import_ref ir7, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %E.decl: E = fn_decl @E [template = constants.%struct.6] {
// CHECK:STDOUT:     %bool.make_type.loc15: init type = call constants.%struct.1() [template = bool]
// CHECK:STDOUT:     %.loc15_11.1: type = value_of_initializer %bool.make_type.loc15 [template = bool]
// CHECK:STDOUT:     %.loc15_11.2: type = converted %bool.make_type.loc15, %.loc15_11.1 [template = bool]
// CHECK:STDOUT:     @E.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: Bool = import_ref ir7, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.7] {
// CHECK:STDOUT:     %bool.make_type.loc16: init type = call constants.%struct.1() [template = bool]
// CHECK:STDOUT:     %.loc16_11.1: type = value_of_initializer %bool.make_type.loc16 [template = bool]
// CHECK:STDOUT:     %.loc16_11.2: type = converted %bool.make_type.loc16, %.loc16_11.1 [template = bool]
// CHECK:STDOUT:     @F.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.7: Bool = import_ref ir7, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.8] {
// CHECK:STDOUT:     %bool.make_type.loc17: init type = call constants.%struct.1() [template = bool]
// CHECK:STDOUT:     %.loc17_11.1: type = value_of_initializer %bool.make_type.loc17 [template = bool]
// CHECK:STDOUT:     %.loc17_11.2: type = converted %bool.make_type.loc17, %.loc17_11.1 [template = bool]
// CHECK:STDOUT:     @G.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8: Bool = import_ref ir7, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %H.decl: H = fn_decl @H [template = constants.%struct.9] {
// CHECK:STDOUT:     %bool.make_type.loc18: init type = call constants.%struct.1() [template = bool]
// CHECK:STDOUT:     %.loc18_11.1: type = value_of_initializer %bool.make_type.loc18 [template = bool]
// CHECK:STDOUT:     %.loc18_11.2: type = converted %bool.make_type.loc18, %.loc18_11.1 [template = bool]
// CHECK:STDOUT:     @H.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %While.decl: While = fn_decl @While [template = constants.%struct.10] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
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
// CHECK:STDOUT:   br !while.cond.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond.loc21:
// CHECK:STDOUT:   %A.ref: A = name_ref A, file.%A.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %A.call: init bool = call %A.ref()
// CHECK:STDOUT:   %.loc21_13.1: bool = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc21_13.2: bool = converted %A.call, %.loc21_13.1
// CHECK:STDOUT:   if %.loc21_13.2 br !while.body.loc21 else br !while.done.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body.loc21:
// CHECK:STDOUT:   %B.ref: B = name_ref B, file.%B.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %B.call: init bool = call %B.ref()
// CHECK:STDOUT:   %.loc22_12.1: bool = value_of_initializer %B.call
// CHECK:STDOUT:   %.loc22_12.2: bool = converted %B.call, %.loc22_12.1
// CHECK:STDOUT:   if %.loc22_12.2 br !if.then.loc22 else br !if.else.loc22
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc22:
// CHECK:STDOUT:   br !while.cond.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc22:
// CHECK:STDOUT:   %C.ref: C = name_ref C, file.%C.decl [template = constants.%struct.4]
// CHECK:STDOUT:   %C.call: init bool = call %C.ref()
// CHECK:STDOUT:   %.loc23_12.1: bool = value_of_initializer %C.call
// CHECK:STDOUT:   %.loc23_12.2: bool = converted %C.call, %.loc23_12.1
// CHECK:STDOUT:   if %.loc23_12.2 br !if.then.loc23 else br !if.else.loc23
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc23:
// CHECK:STDOUT:   br !while.done.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc23:
// CHECK:STDOUT:   br !while.cond.loc24
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond.loc24:
// CHECK:STDOUT:   %D.ref: D = name_ref D, file.%D.decl [template = constants.%struct.5]
// CHECK:STDOUT:   %D.call: init bool = call %D.ref()
// CHECK:STDOUT:   %.loc24_15.1: bool = value_of_initializer %D.call
// CHECK:STDOUT:   %.loc24_15.2: bool = converted %D.call, %.loc24_15.1
// CHECK:STDOUT:   if %.loc24_15.2 br !while.body.loc24 else br !while.done.loc24
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body.loc24:
// CHECK:STDOUT:   %E.ref: E = name_ref E, file.%E.decl [template = constants.%struct.6]
// CHECK:STDOUT:   %E.call: init bool = call %E.ref()
// CHECK:STDOUT:   %.loc25_14.1: bool = value_of_initializer %E.call
// CHECK:STDOUT:   %.loc25_14.2: bool = converted %E.call, %.loc25_14.1
// CHECK:STDOUT:   if %.loc25_14.2 br !if.then.loc25 else br !if.else.loc25
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc25:
// CHECK:STDOUT:   br !while.cond.loc24
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc25:
// CHECK:STDOUT:   %F.ref: F = name_ref F, file.%F.decl [template = constants.%struct.7]
// CHECK:STDOUT:   %F.call: init bool = call %F.ref()
// CHECK:STDOUT:   %.loc26_14.1: bool = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc26_14.2: bool = converted %F.call, %.loc26_14.1
// CHECK:STDOUT:   if %.loc26_14.2 br !if.then.loc26 else br !if.else.loc26
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc26:
// CHECK:STDOUT:   br !while.done.loc24
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc26:
// CHECK:STDOUT:   br !while.cond.loc24
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done.loc24:
// CHECK:STDOUT:   %G.ref: G = name_ref G, file.%G.decl [template = constants.%struct.8]
// CHECK:STDOUT:   %G.call: init bool = call %G.ref()
// CHECK:STDOUT:   %.loc28_12.1: bool = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc28_12.2: bool = converted %G.call, %.loc28_12.1
// CHECK:STDOUT:   if %.loc28_12.2 br !if.then.loc28 else br !if.else.loc28
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc28:
// CHECK:STDOUT:   br !while.cond.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc28:
// CHECK:STDOUT:   %H.ref: H = name_ref H, file.%H.decl [template = constants.%struct.9]
// CHECK:STDOUT:   %H.call: init bool = call %H.ref()
// CHECK:STDOUT:   %.loc29_12.1: bool = value_of_initializer %H.call
// CHECK:STDOUT:   %.loc29_12.2: bool = converted %H.call, %.loc29_12.1
// CHECK:STDOUT:   if %.loc29_12.2 br !if.then.loc29 else br !if.else.loc29
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc29:
// CHECK:STDOUT:   br !while.done.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc29:
// CHECK:STDOUT:   br !while.cond.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done.loc21:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
