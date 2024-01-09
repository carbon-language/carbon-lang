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
// CHECK:STDOUT:   %Cond: <function> = fn_decl @Cond, const
// CHECK:STDOUT:   %F: <function> = fn_decl @F, const
// CHECK:STDOUT:   %G: <function> = fn_decl @G, const
// CHECK:STDOUT:   %H: <function> = fn_decl @H, const
// CHECK:STDOUT:   %While: <function> = fn_decl @While, const
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
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, file.%F, const = file.%F
// CHECK:STDOUT:   %.loc14: init () = call %F.ref()
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond:
// CHECK:STDOUT:   %Cond.ref: <function> = name_ref Cond, file.%Cond, const = file.%Cond
// CHECK:STDOUT:   %.loc15_14: init bool = call %Cond.ref()
// CHECK:STDOUT:   %.loc15_16.1: bool = value_of_initializer %.loc15_14
// CHECK:STDOUT:   %.loc15_16.2: bool = converted %.loc15_14, %.loc15_16.1
// CHECK:STDOUT:   if %.loc15_16.2 br !while.body else br !while.done
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body:
// CHECK:STDOUT:   %G.ref: <function> = name_ref G, file.%G, const = file.%G
// CHECK:STDOUT:   %.loc16: init () = call %G.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done:
// CHECK:STDOUT:   %H.ref: <function> = name_ref H, file.%H, const = file.%H
// CHECK:STDOUT:   %.loc19: init () = call %H.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
