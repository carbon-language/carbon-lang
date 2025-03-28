// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/while/break_continue.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/while/break_continue.carbon

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
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.type: type = fn_type @C [concrete]
// CHECK:STDOUT:   %C: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.type: type = fn_type @D [concrete]
// CHECK:STDOUT:   %D: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %E.type: type = fn_type @E [concrete]
// CHECK:STDOUT:   %E: %E.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %While.type: type = fn_type @While [concrete]
// CHECK:STDOUT:   %While: %While.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
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
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %bool.make_type, %.loc11_11.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc12_11.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc12_11.2: type = converted %bool.make_type, %.loc12_11.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [concrete = constants.%C] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc13_11.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc13_11.2: type = converted %bool.make_type, %.loc13_11.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = fn_decl @D [concrete = constants.%D] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc14_11.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc14_11.2: type = converted %bool.make_type, %.loc14_11.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.decl: %E.type = fn_decl @E [concrete = constants.%E] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc15_11.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc15_11.2: type = converted %bool.make_type, %.loc15_11.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc16_11.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc16_11.2: type = converted %bool.make_type, %.loc16_11.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc17_11.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc17_11.2: type = converted %bool.make_type, %.loc17_11.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc18_11.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc18_11.2: type = converted %bool.make_type, %.loc18_11.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %While.decl: %While.type = fn_decl @While [concrete = constants.%While] {} {}
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
// CHECK:STDOUT:   br !while.cond.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond.loc21:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init bool = call %A.ref()
// CHECK:STDOUT:   %.loc21_13.1: bool = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc21_13.2: bool = converted %A.call, %.loc21_13.1
// CHECK:STDOUT:   if %.loc21_13.2 br !while.body.loc21 else br !while.done.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body.loc21:
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %B.call: init bool = call %B.ref()
// CHECK:STDOUT:   %.loc22_12.1: bool = value_of_initializer %B.call
// CHECK:STDOUT:   %.loc22_12.2: bool = converted %B.call, %.loc22_12.1
// CHECK:STDOUT:   if %.loc22_12.2 br !if.then.loc22 else br !if.else.loc22
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc22:
// CHECK:STDOUT:   br !while.cond.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc22:
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C]
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
// CHECK:STDOUT:   %D.ref: %D.type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %D.call: init bool = call %D.ref()
// CHECK:STDOUT:   %.loc24_15.1: bool = value_of_initializer %D.call
// CHECK:STDOUT:   %.loc24_15.2: bool = converted %D.call, %.loc24_15.1
// CHECK:STDOUT:   if %.loc24_15.2 br !while.body.loc24 else br !while.done.loc24
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body.loc24:
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %E.call: init bool = call %E.ref()
// CHECK:STDOUT:   %.loc25_14.1: bool = value_of_initializer %E.call
// CHECK:STDOUT:   %.loc25_14.2: bool = converted %E.call, %.loc25_14.1
// CHECK:STDOUT:   if %.loc25_14.2 br !if.then.loc25 else br !if.else.loc25
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc25:
// CHECK:STDOUT:   br !while.cond.loc24
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc25:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
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
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init bool = call %G.ref()
// CHECK:STDOUT:   %.loc28_12.1: bool = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc28_12.2: bool = converted %G.call, %.loc28_12.1
// CHECK:STDOUT:   if %.loc28_12.2 br !if.then.loc28 else br !if.else.loc28
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc28:
// CHECK:STDOUT:   br !while.cond.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc28:
// CHECK:STDOUT:   %H.ref: %H.type = name_ref H, file.%H.decl [concrete = constants.%H]
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
