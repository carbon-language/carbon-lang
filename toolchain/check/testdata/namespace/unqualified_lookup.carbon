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
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.2: B = struct_value () [template]
// CHECK:STDOUT:   %C: type = fn_type @C [template]
// CHECK:STDOUT:   %struct.3: C = struct_value () [template]
// CHECK:STDOUT:   %CallA: type = fn_type @CallA [template]
// CHECK:STDOUT:   %struct.4: CallA = struct_value () [template]
// CHECK:STDOUT:   %CallAB: type = fn_type @CallAB [template]
// CHECK:STDOUT:   %struct.5: CallAB = struct_value () [template]
// CHECK:STDOUT:   %CallABC: type = fn_type @CallABC [template]
// CHECK:STDOUT:   %struct.6: CallABC = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .OuterN = %OuterN
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .CallA = %CallA.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %OuterN: <namespace> = namespace [template] {
// CHECK:STDOUT:     .InnerN = %InnerN
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .CallAB = %CallAB.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InnerN: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .CallABC = %CallABC.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct.1] {}
// CHECK:STDOUT:   %B.decl: B = fn_decl @B [template = constants.%struct.2] {}
// CHECK:STDOUT:   %C.decl: C = fn_decl @C [template = constants.%struct.3] {}
// CHECK:STDOUT:   %CallA.decl: CallA = fn_decl @CallA [template = constants.%struct.4] {}
// CHECK:STDOUT:   %CallAB.decl: CallAB = fn_decl @CallAB [template = constants.%struct.5] {}
// CHECK:STDOUT:   %CallABC.decl: CallABC = fn_decl @CallABC [template = constants.%struct.6] {}
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
// CHECK:STDOUT:   %A.ref: A = name_ref A, file.%A.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallAB() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: A = name_ref A, file.%A.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   %B.ref: B = name_ref B, file.%B.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %B.call: init () = call %B.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallABC() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: A = name_ref A, file.%A.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   %B.ref: B = name_ref B, file.%B.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %B.call: init () = call %B.ref()
// CHECK:STDOUT:   %C.ref: C = name_ref C, file.%C.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %C.call: init () = call %C.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
