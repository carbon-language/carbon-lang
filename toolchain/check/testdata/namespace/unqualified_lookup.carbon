// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/unqualified_lookup.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/unqualified_lookup.carbon

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
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT:   %CallA.type: type = fn_type @CallA [template]
// CHECK:STDOUT:   %CallA: %CallA.type = struct_value () [template]
// CHECK:STDOUT:   %CallAB.type: type = fn_type @CallAB [template]
// CHECK:STDOUT:   %CallAB: %CallAB.type = struct_value () [template]
// CHECK:STDOUT:   %CallABC.type: type = fn_type @CallABC [template]
// CHECK:STDOUT:   %CallABC: %CallABC.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .OuterN = %OuterN
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .CallA = %CallA.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %OuterN: <namespace> = namespace [template] {
// CHECK:STDOUT:     .InnerN = %InnerN
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .CallAB = %CallAB.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InnerN: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .CallABC = %CallABC.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %CallA.decl: %CallA.type = fn_decl @CallA [template = constants.%CallA] {} {}
// CHECK:STDOUT:   %CallAB.decl: %CallAB.type = fn_decl @CallAB [template = constants.%CallAB] {} {}
// CHECK:STDOUT:   %CallABC.decl: %CallABC.type = fn_decl @CallABC [template = constants.%CallABC] {} {}
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
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallAB() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %B.call: init %empty_tuple.type = call %B.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallABC() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %B.call: init %empty_tuple.type = call %B.ref()
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %C.call: init %empty_tuple.type = call %C.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
