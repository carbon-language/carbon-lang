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

// CHECK:STDOUT: --- unreachable_end.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc14: type = tuple_type ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Cond = %Cond, .F = %F, .G = %G, .H = %H, .While = %While}
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
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, file.%F
// CHECK:STDOUT:   %.loc14_3: <function> = reify_constant %F.ref, file.%F
// CHECK:STDOUT:   %.loc14_4: init () = call file.%F()
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond:
// CHECK:STDOUT:   %Cond.ref: <function> = name_ref Cond, file.%Cond
// CHECK:STDOUT:   %.loc15_10: <function> = reify_constant %Cond.ref, file.%Cond
// CHECK:STDOUT:   %.loc15_14: init bool = call file.%Cond()
// CHECK:STDOUT:   %.loc15_16.1: bool = value_of_initializer %.loc15_14
// CHECK:STDOUT:   %.loc15_16.2: bool = converted %.loc15_14, %.loc15_16.1
// CHECK:STDOUT:   if %.loc15_16.2 br !while.body else br !while.done
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body:
// CHECK:STDOUT:   %G.ref: <function> = name_ref G, file.%G
// CHECK:STDOUT:   %.loc16_5: <function> = reify_constant %G.ref, file.%G
// CHECK:STDOUT:   %.loc16_6: init () = call file.%G()
// CHECK:STDOUT:   return
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done:
// CHECK:STDOUT:   %H.ref: <function> = name_ref H, file.%H
// CHECK:STDOUT:   %.loc19_3: <function> = reify_constant %H.ref, file.%H
// CHECK:STDOUT:   %.loc19_4: init () = call file.%H()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
