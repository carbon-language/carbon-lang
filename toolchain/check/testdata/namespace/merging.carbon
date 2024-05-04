// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon

package Example library "a" api;

namespace NS;

fn NS.A() {}

// --- b.carbon

package Example library "b" api;

namespace NS;

fn NS.B1() {}

namespace NS;

fn NS.B2() {}

// --- c.carbon

package Example library "c" api;

import library "a";
import library "b";

namespace NS;

fn NS.C() {}

fn Run() {
  NS.A();
  NS.B1();
  NS.B2();
  NS.C();
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: A = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B1: type = fn_type @B1 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: B1 = struct_value () [template]
// CHECK:STDOUT:   %B2: type = fn_type @B2 [template]
// CHECK:STDOUT:   %struct.2: B2 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .B1 = %B1.decl
// CHECK:STDOUT:     .B2 = %B2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B1.decl: B1 = fn_decl @B1 [template = constants.%struct.1] {}
// CHECK:STDOUT:   %.loc8: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %B2.decl: B2 = fn_decl @B2 [template = constants.%struct.2] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- c.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = fn_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: C = struct_value () [template]
// CHECK:STDOUT:   %Run: type = fn_type @Run [template]
// CHECK:STDOUT:   %struct.2: Run = struct_value () [template]
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %struct.3: A = struct_value () [template]
// CHECK:STDOUT:   %B1: type = fn_type @B1 [template]
// CHECK:STDOUT:   %struct.4: B1 = struct_value () [template]
// CHECK:STDOUT:   %B2: type = fn_type @B2 [template]
// CHECK:STDOUT:   %struct.5: B2 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <namespace> = import_ref ir1, inst+2, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.1, [template] {
// CHECK:STDOUT:     .A = %import_ref.2
// CHECK:STDOUT:     .B1 = %import_ref.3
// CHECK:STDOUT:     .B2 = %import_ref.4
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: A = import_ref ir1, inst+3, loaded [template = constants.%struct.3]
// CHECK:STDOUT:   %import_ref.3: B1 = import_ref ir2, inst+3, loaded [template = constants.%struct.4]
// CHECK:STDOUT:   %import_ref.4: B2 = import_ref ir2, inst+9, loaded [template = constants.%struct.5]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc7: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: C = fn_decl @C [template = constants.%struct.1] {}
// CHECK:STDOUT:   %Run.decl: Run = fn_decl @Run [template = constants.%struct.2] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NS.ref.loc12: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %A.ref: A = name_ref A, file.%import_ref.2 [template = constants.%struct.3]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   %NS.ref.loc13: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %B1.ref: B1 = name_ref B1, file.%import_ref.3 [template = constants.%struct.4]
// CHECK:STDOUT:   %B1.call: init () = call %B1.ref()
// CHECK:STDOUT:   %NS.ref.loc14: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %B2.ref: B2 = name_ref B2, file.%import_ref.4 [template = constants.%struct.5]
// CHECK:STDOUT:   %B2.call: init () = call %B2.ref()
// CHECK:STDOUT:   %NS.ref.loc15: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %C.ref: C = name_ref C, file.%C.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %C.call: init () = call %C.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B2();
// CHECK:STDOUT:
