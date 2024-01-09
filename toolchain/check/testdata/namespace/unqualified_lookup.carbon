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

// CHECK:STDOUT: --- unqualified_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc15: type = tuple_type ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.OuterN = %.loc7, .A = %A, .CallA = %CallA}
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
// CHECK:STDOUT:   %.loc15_3: <function> = reify_constant %A.ref, file.%A
// CHECK:STDOUT:   %.loc15_4: init () = call file.%A()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallAB() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%A
// CHECK:STDOUT:   %.loc19_3: <function> = reify_constant %A.ref, file.%A
// CHECK:STDOUT:   %.loc19_4: init () = call file.%A()
// CHECK:STDOUT:   %B.ref: <function> = name_ref B, file.%B
// CHECK:STDOUT:   %.loc20_3: <function> = reify_constant %B.ref, file.%B
// CHECK:STDOUT:   %.loc20_4: init () = call file.%B()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallABC() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%A
// CHECK:STDOUT:   %.loc24_3: <function> = reify_constant %A.ref, file.%A
// CHECK:STDOUT:   %.loc24_4: init () = call file.%A()
// CHECK:STDOUT:   %B.ref: <function> = name_ref B, file.%B
// CHECK:STDOUT:   %.loc25_3: <function> = reify_constant %B.ref, file.%B
// CHECK:STDOUT:   %.loc25_4: init () = call file.%B()
// CHECK:STDOUT:   %C.ref: <function> = name_ref C, file.%C
// CHECK:STDOUT:   %.loc26_3: <function> = reify_constant %C.ref, file.%C
// CHECK:STDOUT:   %.loc26_4: init () = call file.%C()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
