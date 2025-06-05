// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// INCLUDE-FILE: toolchain/testing/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/while/min_prelude/while.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/while/min_prelude/while.carbon

// --- while.carbon
library "[[@TEST_NAME]]";

fn Cond() -> bool;

fn F();
fn G();
fn H();

//@dump-sem-ir-begin
fn While() {
  F();
  while (Cond()) {
    G();
  }
  H();
}
//@dump-sem-ir-end

// --- unreachable_end.carbon
library "[[@TEST_NAME]]";

fn Cond() -> bool;

fn F();
fn G();
fn H();

//@dump-sem-ir-begin
fn While() {
  F();
  while (Cond()) {
    G();
    return;
  }
  H();
}
//@dump-sem-ir-end

// --- break_continue.carbon
library "[[@TEST_NAME]]";

fn A() -> bool;
fn B() -> bool;
fn C() -> bool;
fn D() -> bool;
fn E() -> bool;
fn F() -> bool;
fn G() -> bool;
fn H() -> bool;

//@dump-sem-ir-begin
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
//@dump-sem-ir-end

// --- fail_bad_condition.carbon
library "[[@TEST_NAME]]";

fn While() {
  // CHECK:STDERR: fail_bad_condition.carbon:[[@LINE+7]]:9: error: cannot implicitly convert expression of type `String` to `bool` [ConversionFailure]
  // CHECK:STDERR:   while ("Hello") {}
  // CHECK:STDERR:         ^~~~~~~~~
  // CHECK:STDERR: fail_bad_condition.carbon:[[@LINE+4]]:9: note: type `String` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   while ("Hello") {}
  // CHECK:STDERR:         ^~~~~~~~~
  // CHECK:STDERR:
  while ("Hello") {}
}

// --- fail_bad_break_continue.carbon
library "[[@TEST_NAME]]";

fn While() {
  // CHECK:STDERR: fail_bad_break_continue.carbon:[[@LINE+4]]:3: error: `continue` can only be used in a loop [ContinueOutsideLoop]
  // CHECK:STDERR:   continue;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  continue;
  // CHECK:STDERR: fail_bad_break_continue.carbon:[[@LINE+4]]:3: error: `break` can only be used in a loop [BreakOutsideLoop]
  // CHECK:STDERR:   break;
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  break;
  while (false) {
    continue;
    break;
  }
  // CHECK:STDERR: fail_bad_break_continue.carbon:[[@LINE+4]]:3: error: `continue` can only be used in a loop [ContinueOutsideLoop]
  // CHECK:STDERR:   continue;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  continue;
  // CHECK:STDERR: fail_bad_break_continue.carbon:[[@LINE+4]]:3: error: `break` can only be used in a loop [BreakOutsideLoop]
  // CHECK:STDERR:   break;
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  break;
}

// CHECK:STDOUT: --- while.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cond.type: type = fn_type @Cond [concrete]
// CHECK:STDOUT:   %Cond: %Cond.type = struct_value () [concrete]
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %While.decl: %While.type = fn_decl @While [concrete = constants.%While] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @While() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond:
// CHECK:STDOUT:   %Cond.ref: %Cond.type = name_ref Cond, file.%Cond.decl [concrete = constants.%Cond]
// CHECK:STDOUT:   %Cond.call: init bool = call %Cond.ref()
// CHECK:STDOUT:   %.loc12_16.1: bool = value_of_initializer %Cond.call
// CHECK:STDOUT:   %.loc12_16.2: bool = converted %Cond.call, %.loc12_16.1
// CHECK:STDOUT:   if %.loc12_16.2 br !while.body else br !while.done
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.ref()
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done:
// CHECK:STDOUT:   %H.ref: %H.type = name_ref H, file.%H.decl [concrete = constants.%H]
// CHECK:STDOUT:   %H.call: init %empty_tuple.type = call %H.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unreachable_end.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cond.type: type = fn_type @Cond [concrete]
// CHECK:STDOUT:   %Cond: %Cond.type = struct_value () [concrete]
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %While.decl: %While.type = fn_decl @While [concrete = constants.%While] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @While() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond:
// CHECK:STDOUT:   %Cond.ref: %Cond.type = name_ref Cond, file.%Cond.decl [concrete = constants.%Cond]
// CHECK:STDOUT:   %Cond.call: init bool = call %Cond.ref()
// CHECK:STDOUT:   %.loc12_16.1: bool = value_of_initializer %Cond.call
// CHECK:STDOUT:   %.loc12_16.2: bool = converted %Cond.call, %.loc12_16.1
// CHECK:STDOUT:   if %.loc12_16.2 br !while.body else br !while.done
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done:
// CHECK:STDOUT:   %H.ref: %H.type = name_ref H, file.%H.decl [concrete = constants.%H]
// CHECK:STDOUT:   %H.call: init %empty_tuple.type = call %H.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- break_continue.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %While.decl: %While.type = fn_decl @While [concrete = constants.%While] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @While() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   br !while.cond.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond.loc14:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init bool = call %A.ref()
// CHECK:STDOUT:   %.loc14_13.1: bool = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc14_13.2: bool = converted %A.call, %.loc14_13.1
// CHECK:STDOUT:   if %.loc14_13.2 br !while.body.loc14 else br !while.done.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body.loc14:
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %B.call: init bool = call %B.ref()
// CHECK:STDOUT:   %.loc15_12.1: bool = value_of_initializer %B.call
// CHECK:STDOUT:   %.loc15_12.2: bool = converted %B.call, %.loc15_12.1
// CHECK:STDOUT:   if %.loc15_12.2 br !if.then.loc15 else br !if.else.loc15
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc15:
// CHECK:STDOUT:   br !while.cond.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc15:
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.call: init bool = call %C.ref()
// CHECK:STDOUT:   %.loc16_12.1: bool = value_of_initializer %C.call
// CHECK:STDOUT:   %.loc16_12.2: bool = converted %C.call, %.loc16_12.1
// CHECK:STDOUT:   if %.loc16_12.2 br !if.then.loc16 else br !if.else.loc16
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc16:
// CHECK:STDOUT:   br !while.done.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc16:
// CHECK:STDOUT:   br !while.cond.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond.loc17:
// CHECK:STDOUT:   %D.ref: %D.type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %D.call: init bool = call %D.ref()
// CHECK:STDOUT:   %.loc17_15.1: bool = value_of_initializer %D.call
// CHECK:STDOUT:   %.loc17_15.2: bool = converted %D.call, %.loc17_15.1
// CHECK:STDOUT:   if %.loc17_15.2 br !while.body.loc17 else br !while.done.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body.loc17:
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %E.call: init bool = call %E.ref()
// CHECK:STDOUT:   %.loc18_14.1: bool = value_of_initializer %E.call
// CHECK:STDOUT:   %.loc18_14.2: bool = converted %E.call, %.loc18_14.1
// CHECK:STDOUT:   if %.loc18_14.2 br !if.then.loc18 else br !if.else.loc18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc18:
// CHECK:STDOUT:   br !while.cond.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc18:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init bool = call %F.ref()
// CHECK:STDOUT:   %.loc19_14.1: bool = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc19_14.2: bool = converted %F.call, %.loc19_14.1
// CHECK:STDOUT:   if %.loc19_14.2 br !if.then.loc19 else br !if.else.loc19
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc19:
// CHECK:STDOUT:   br !while.done.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc19:
// CHECK:STDOUT:   br !while.cond.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done.loc17:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init bool = call %G.ref()
// CHECK:STDOUT:   %.loc21_12.1: bool = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc21_12.2: bool = converted %G.call, %.loc21_12.1
// CHECK:STDOUT:   if %.loc21_12.2 br !if.then.loc21 else br !if.else.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc21:
// CHECK:STDOUT:   br !while.cond.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc21:
// CHECK:STDOUT:   %H.ref: %H.type = name_ref H, file.%H.decl [concrete = constants.%H]
// CHECK:STDOUT:   %H.call: init bool = call %H.ref()
// CHECK:STDOUT:   %.loc22_12.1: bool = value_of_initializer %H.call
// CHECK:STDOUT:   %.loc22_12.2: bool = converted %H.call, %.loc22_12.1
// CHECK:STDOUT:   if %.loc22_12.2 br !if.then.loc22 else br !if.else.loc22
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc22:
// CHECK:STDOUT:   br !while.done.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc22:
// CHECK:STDOUT:   br !while.cond.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done.loc14:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
