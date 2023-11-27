// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace OuterN;
namespace OuterN.InnerN;

fn A();
fn OuterN.B();
fn OuterN.InnerN.C();

fn CallA() {
  A();
}

fn OuterN.CallAB() {
  A();
  B();
}

fn OuterN.InnerN.CallABC() {
  A();
  B();
  C();
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc15: type = tuple_type ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "unqualified_lookup.carbon" {
// CHECK:STDOUT:   %.loc7: <namespace> = namespace {.InnerN = %.loc8, .B = %B, .CallAB = %CallAB}
// CHECK:STDOUT:   %.loc8: <namespace> = namespace {.C = %C, .CallABC = %CallABC}
// CHECK:STDOUT:   %A: <function> = fn_decl @A
// CHECK:STDOUT:   %B: <function> = fn_decl @B
// CHECK:STDOUT:   %C: <function> = fn_decl @C
// CHECK:STDOUT:   %CallA: <function> = fn_decl @CallA
// CHECK:STDOUT:   %CallAB: <function> = fn_decl @CallAB
// CHECK:STDOUT:   %CallABC: <function> = fn_decl @CallABC
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallA() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%A
// CHECK:STDOUT:   %.loc15: init () = call %A.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallAB() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%A
// CHECK:STDOUT:   %.loc19: init () = call %A.ref()
// CHECK:STDOUT:   %B.ref: <function> = name_ref B, file.%B
// CHECK:STDOUT:   %.loc20: init () = call %B.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallABC() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%A
// CHECK:STDOUT:   %.loc24: init () = call %A.ref()
// CHECK:STDOUT:   %B.ref: <function> = name_ref B, file.%B
// CHECK:STDOUT:   %.loc25: init () = call %B.ref()
// CHECK:STDOUT:   %C.ref: <function> = name_ref C, file.%C
// CHECK:STDOUT:   %.loc26: init () = call %C.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
