// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Cond() -> bool;

fn F();
fn G();
fn H();

fn While() {
  F();
  while (Cond()) {
    G();
    return;
  }
  H();
}

// CHECK:STDOUT: file "unreachable_end.carbon" {
// CHECK:STDOUT:   %Cond: <function> = fn_decl @Cond
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:   %H: <function> = fn_decl @H
// CHECK:STDOUT:   %While: <function> = fn_decl @While
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Cond() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @While() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <function> = name_reference "F", file.%F
// CHECK:STDOUT:   %.loc14_4.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc14_4.2: init () = call %F.ref()
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond:
// CHECK:STDOUT:   %Cond.ref: <function> = name_reference "Cond", file.%Cond
// CHECK:STDOUT:   %.loc15_14: init bool = call %Cond.ref()
// CHECK:STDOUT:   %.loc15_16: bool = value_of_initializer %.loc15_14
// CHECK:STDOUT:   if %.loc15_16 br !while.body else br !while.done
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body:
// CHECK:STDOUT:   %G.ref: <function> = name_reference "G", file.%G
// CHECK:STDOUT:   %.loc16: init () = call %G.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done:
// CHECK:STDOUT:   %H.ref: <function> = name_reference "H", file.%H
// CHECK:STDOUT:   %.loc19: init () = call %H.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
