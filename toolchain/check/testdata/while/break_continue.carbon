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

// CHECK:STDOUT: file "break_continue.carbon" {
// CHECK:STDOUT:   %A: <function> = fn_decl @A
// CHECK:STDOUT:   %B: <function> = fn_decl @B
// CHECK:STDOUT:   %C: <function> = fn_decl @C
// CHECK:STDOUT:   %D: <function> = fn_decl @D
// CHECK:STDOUT:   %E: <function> = fn_decl @E
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:   %H: <function> = fn_decl @H
// CHECK:STDOUT:   %While: <function> = fn_decl @While
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
// CHECK:STDOUT:   %A.ref: <function> = name_reference "A", file.%A
// CHECK:STDOUT:   %.loc17_11: init bool = call %A.ref()
// CHECK:STDOUT:   %.loc17_13: bool = value_of_initializer %.loc17_11
// CHECK:STDOUT:   if %.loc17_13 br !while.body.loc17 else br !while.done.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body.loc17:
// CHECK:STDOUT:   %B.ref: <function> = name_reference "B", file.%B
// CHECK:STDOUT:   %.loc18_10: init bool = call %B.ref()
// CHECK:STDOUT:   %.loc18_12: bool = value_of_initializer %.loc18_10
// CHECK:STDOUT:   if %.loc18_12 br !if.then.loc18 else br !if.else.loc18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc18:
// CHECK:STDOUT:   br !while.cond.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc18:
// CHECK:STDOUT:   %C.ref: <function> = name_reference "C", file.%C
// CHECK:STDOUT:   %.loc19_10: init bool = call %C.ref()
// CHECK:STDOUT:   %.loc19_12: bool = value_of_initializer %.loc19_10
// CHECK:STDOUT:   if %.loc19_12 br !if.then.loc19 else br !if.else.loc19
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc19:
// CHECK:STDOUT:   br !while.done.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc19:
// CHECK:STDOUT:   br !while.cond.loc20
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond.loc20:
// CHECK:STDOUT:   %D.ref: <function> = name_reference "D", file.%D
// CHECK:STDOUT:   %.loc20_13: init bool = call %D.ref()
// CHECK:STDOUT:   %.loc20_15: bool = value_of_initializer %.loc20_13
// CHECK:STDOUT:   if %.loc20_15 br !while.body.loc20 else br !while.done.loc20
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body.loc20:
// CHECK:STDOUT:   %E.ref: <function> = name_reference "E", file.%E
// CHECK:STDOUT:   %.loc21_12: init bool = call %E.ref()
// CHECK:STDOUT:   %.loc21_14: bool = value_of_initializer %.loc21_12
// CHECK:STDOUT:   if %.loc21_14 br !if.then.loc21 else br !if.else.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc21:
// CHECK:STDOUT:   br !while.cond.loc20
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc21:
// CHECK:STDOUT:   %F.ref: <function> = name_reference "F", file.%F
// CHECK:STDOUT:   %.loc22_12: init bool = call %F.ref()
// CHECK:STDOUT:   %.loc22_14: bool = value_of_initializer %.loc22_12
// CHECK:STDOUT:   if %.loc22_14 br !if.then.loc22 else br !if.else.loc22
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc22:
// CHECK:STDOUT:   br !while.done.loc20
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc22:
// CHECK:STDOUT:   br !while.cond.loc20
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done.loc20:
// CHECK:STDOUT:   %G.ref: <function> = name_reference "G", file.%G
// CHECK:STDOUT:   %.loc24_10: init bool = call %G.ref()
// CHECK:STDOUT:   %.loc24_12: bool = value_of_initializer %.loc24_10
// CHECK:STDOUT:   if %.loc24_12 br !if.then.loc24 else br !if.else.loc24
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc24:
// CHECK:STDOUT:   br !while.cond.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc24:
// CHECK:STDOUT:   %H.ref: <function> = name_reference "H", file.%H
// CHECK:STDOUT:   %.loc25_10: init bool = call %H.ref()
// CHECK:STDOUT:   %.loc25_12: bool = value_of_initializer %.loc25_10
// CHECK:STDOUT:   if %.loc25_12 br !if.then.loc25 else br !if.else.loc25
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
