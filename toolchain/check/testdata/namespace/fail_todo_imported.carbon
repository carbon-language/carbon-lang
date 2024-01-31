// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- implicit.carbon

package Implicit api;

namespace NS;
namespace NS.ChildNS;

fn NS.A();
fn NS.ChildNS.B();

// --- implicit.impl.carbon

package Implicit impl;

// CHECK:STDERR: implicit.impl.carbon:[[@LINE+3]]:13: ERROR: Value of type `<function>` is not callable.
// CHECK:STDERR: var a: () = NS.A();
// CHECK:STDERR:             ^~~~~
var a: () = NS.A();
// CHECK:STDERR: implicit.impl.carbon:[[@LINE+3]]:13: ERROR: Value of type `<function>` is not callable.
// CHECK:STDERR: var b: () = NS.ChildNS.B();
// CHECK:STDERR:             ^~~~~~~~~~~~~
var b: () = NS.ChildNS.B();

// CHECK:STDERR: implicit.impl.carbon:[[@LINE+3]]:21: ERROR: Value of type `<function>` is not callable.
// CHECK:STDERR: var package_a: () = package.NS.A();
// CHECK:STDERR:                     ^~~~~~~~~~~~~
var package_a: () = package.NS.A();
// CHECK:STDERR: implicit.impl.carbon:[[@LINE+3]]:21: ERROR: Value of type `<function>` is not callable.
// CHECK:STDERR: var package_b: () = package.NS.ChildNS.B();
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~~~~~~~
var package_b: () = package.NS.ChildNS.B();

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.NS = %.loc4} [template]
// CHECK:STDOUT:   %.loc4: <namespace> = namespace {.ChildNS = %.loc5, .A = %A} [template]
// CHECK:STDOUT:   %.loc5: <namespace> = namespace {.B = %B} [template]
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template]
// CHECK:STDOUT:   %B: <function> = fn_decl @B [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B();
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.NS = %.2, .a = %a, .b = %b, .package_a = %package_a, .package_b = %package_b} [template]
// CHECK:STDOUT:   %import_ref.1: <namespace> = import_ref ir1, inst+1, used
// CHECK:STDOUT:   %.2: <namespace> = namespace {.ChildNS = %.3, .A = %import_ref.3}, %import_ref.1 [template]
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir1, inst+2, used
// CHECK:STDOUT:   %.3: <namespace> = namespace {.B = %import_ref.4}, %import_ref.2 [template]
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir1, inst+3, used
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir1, inst+4, used
// CHECK:STDOUT:   %.loc7_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc7_9.2: type = converted %.loc7_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %NS.ref.loc7: <namespace> = name_ref NS, %.2 [template = %.2]
// CHECK:STDOUT:   %A.ref.loc7: <function> = name_ref A, %import_ref.3
// CHECK:STDOUT:   assign %a.var, <error>
// CHECK:STDOUT:   %.loc11_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc11_9.2: type = converted %.loc11_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %b.var: ref () = var b
// CHECK:STDOUT:   %b: ref () = bind_name b, %b.var
// CHECK:STDOUT:   %NS.ref.loc11: <namespace> = name_ref NS, %.2 [template = %.2]
// CHECK:STDOUT:   %ChildNS.ref.loc11: <namespace> = name_ref ChildNS, %.3 [template = %.3]
// CHECK:STDOUT:   %B.ref.loc11: <function> = name_ref B, %import_ref.4
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT:   %.loc16_17.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc16_17.2: type = converted %.loc16_17.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %package_a.var: ref () = var package_a
// CHECK:STDOUT:   %package_a: ref () = bind_name package_a, %package_a.var
// CHECK:STDOUT:   %package.ref.loc16: <namespace> = name_ref package, package [template = package]
// CHECK:STDOUT:   %NS.ref.loc16: <namespace> = name_ref NS, %.2 [template = %.2]
// CHECK:STDOUT:   %A.ref.loc16: <function> = name_ref A, %import_ref.3
// CHECK:STDOUT:   assign %package_a.var, <error>
// CHECK:STDOUT:   %.loc20_17.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc20_17.2: type = converted %.loc20_17.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %package_b.var: ref () = var package_b
// CHECK:STDOUT:   %package_b: ref () = bind_name package_b, %package_b.var
// CHECK:STDOUT:   %package.ref.loc20: <namespace> = name_ref package, package [template = package]
// CHECK:STDOUT:   %NS.ref.loc20: <namespace> = name_ref NS, %.2 [template = %.2]
// CHECK:STDOUT:   %ChildNS.ref.loc20: <namespace> = name_ref ChildNS, %.3 [template = %.3]
// CHECK:STDOUT:   %B.ref.loc20: <function> = name_ref B, %import_ref.4
// CHECK:STDOUT:   assign %package_b.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
