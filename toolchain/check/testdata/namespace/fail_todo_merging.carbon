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
  // CHECK:STDERR: c.carbon:[[@LINE+3]]:3: ERROR: Value of type `<function>` is not callable.
  // CHECK:STDERR:   NS.A();
  // CHECK:STDERR:   ^~~~~
  NS.A();
  // CHECK:STDERR: c.carbon:[[@LINE+3]]:3: ERROR: Value of type `<function>` is not callable.
  // CHECK:STDERR:   NS.B1();
  // CHECK:STDERR:   ^~~~~~
  NS.B1();
  // CHECK:STDERR: c.carbon:[[@LINE+3]]:3: ERROR: Value of type `<function>` is not callable.
  // CHECK:STDERR:   NS.B2();
  // CHECK:STDERR:   ^~~~~~
  NS.B2();
  NS.C();
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.NS = %.loc4} [template]
// CHECK:STDOUT:   %.loc4: <namespace> = namespace {.A = %A} [template]
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template]
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
// CHECK:STDOUT:   package: <namespace> = namespace {.NS = %.loc4} [template]
// CHECK:STDOUT:   %.loc4: <namespace> = namespace {.B1 = %B1, .B2 = %B2} [template]
// CHECK:STDOUT:   %B1: <function> = fn_decl @B1 [template]
// CHECK:STDOUT:   %.loc8: <namespace> = namespace {} [template]
// CHECK:STDOUT:   %B2: <function> = fn_decl @B2 [template]
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
// CHECK:STDOUT:   package: <namespace> = namespace {.NS = %.loc7_13.1, .Run = %Run} [template]
// CHECK:STDOUT:   %import_ref.1: <namespace> = import_ref ir1, inst+1, used
// CHECK:STDOUT:   %.loc7_13.1: <namespace> = namespace {.A = %import_ref.2, .B1 = %import_ref.3, .B2 = %import_ref.4, .C = %C}, %import_ref.1 [template]
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir1, inst+2, used
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir2, inst+2, used
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir2, inst+5, used
// CHECK:STDOUT:   %.loc7_13.2: <namespace> = namespace {} [template]
// CHECK:STDOUT:   %C: <function> = fn_decl @C [template]
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NS.ref.loc15: <namespace> = name_ref NS, file.%.loc7_13.1 [template = file.%.loc7_13.1]
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%import_ref.2
// CHECK:STDOUT:   %NS.ref.loc19: <namespace> = name_ref NS, file.%.loc7_13.1 [template = file.%.loc7_13.1]
// CHECK:STDOUT:   %B1.ref: <function> = name_ref B1, file.%import_ref.3
// CHECK:STDOUT:   %NS.ref.loc23: <namespace> = name_ref NS, file.%.loc7_13.1 [template = file.%.loc7_13.1]
// CHECK:STDOUT:   %B2.ref: <function> = name_ref B2, file.%import_ref.4
// CHECK:STDOUT:   %NS.ref.loc24: <namespace> = name_ref NS, file.%.loc7_13.1 [template = file.%.loc7_13.1]
// CHECK:STDOUT:   %C.ref: <function> = name_ref C, file.%C [template = file.%C]
// CHECK:STDOUT:   %.loc24: init () = call %C.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
