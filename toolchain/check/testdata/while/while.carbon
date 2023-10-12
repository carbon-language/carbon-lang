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
  }
  H();
}

// CHECK:STDOUT: file "while.carbon" {
// CHECK:STDOUT:   %Cond = fn_decl @Cond
// CHECK:STDOUT:   %F = fn_decl @F
// CHECK:STDOUT:   %G = fn_decl @G
// CHECK:STDOUT:   %H = fn_decl @H
// CHECK:STDOUT:   %While = fn_decl @While
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
// CHECK:STDOUT:   %F.ref = name_reference_untyped "F", package.%F
// CHECK:STDOUT:   %.loc14_4.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc14_4.2: init () = call @F()
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond:
// CHECK:STDOUT:   %Cond.ref = name_reference_untyped "Cond", package.%Cond
// CHECK:STDOUT:   %.loc15_14.1: init bool = call @Cond()
// CHECK:STDOUT:   %.loc15_14.2: ref bool = temporary_storage
// CHECK:STDOUT:   %.loc15_14.3: ref bool = temporary %.loc15_14.2, %.loc15_14.1
// CHECK:STDOUT:   %.loc15_14.4: bool = bind_value %.loc15_14.3
// CHECK:STDOUT:   if %.loc15_14.4 br !while.body else br !while.done
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body:
// CHECK:STDOUT:   %G.ref = name_reference_untyped "G", package.%G
// CHECK:STDOUT:   %.loc16: init () = call @G()
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done:
// CHECK:STDOUT:   %H.ref = name_reference_untyped "H", package.%H
// CHECK:STDOUT:   %.loc18: init () = call @H()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
