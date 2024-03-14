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
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .B1 = %B1
// CHECK:STDOUT:     .B2 = %B2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B1: <function> = fn_decl @B1 [template] {}
// CHECK:STDOUT:   %.loc8: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %B2: <function> = fn_decl @B2 [template] {}
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
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Run = %Run
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <namespace> = import_ref ir1, inst+1, used
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.1, [template] {
// CHECK:STDOUT:     .A = %import_ref.2
// CHECK:STDOUT:     .B1 = %import_ref.3
// CHECK:STDOUT:     .B2 = %import_ref.4
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir1, inst+2, used [template = imports.%A]
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir2, inst+2, used [template = imports.%B1]
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir2, inst+5, used [template = imports.%B2]
// CHECK:STDOUT:   %.loc7: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C: <function> = fn_decl @C [template] {}
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run [template] {}
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
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%import_ref.2 [template = imports.%A]
// CHECK:STDOUT:   %.loc12: init () = call %A.ref()
// CHECK:STDOUT:   %NS.ref.loc13: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %B1.ref: <function> = name_ref B1, file.%import_ref.3 [template = imports.%B1]
// CHECK:STDOUT:   %.loc13: init () = call %B1.ref()
// CHECK:STDOUT:   %NS.ref.loc14: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %B2.ref: <function> = name_ref B2, file.%import_ref.4 [template = imports.%B2]
// CHECK:STDOUT:   %.loc14: init () = call %B2.ref()
// CHECK:STDOUT:   %NS.ref.loc15: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %C.ref: <function> = name_ref C, file.%C [template = file.%C]
// CHECK:STDOUT:   %.loc15: init () = call %C.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B2();
// CHECK:STDOUT:
