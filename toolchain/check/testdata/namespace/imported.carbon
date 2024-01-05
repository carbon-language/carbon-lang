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

var a: () = NS.A();
var b: () = NS.ChildNS.B();

var package_a: () = package.NS.A();
var package_b: () = package.NS.ChildNS.B();

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.NS = %.loc4}
// CHECK:STDOUT:   %.loc4: <namespace> = namespace NS, {.ChildNS = %.loc5, .A = %A}
// CHECK:STDOUT:   %.loc5: <namespace> = namespace ChildNS, {.B = %B}
// CHECK:STDOUT:   %A: <function> = fn_decl @A
// CHECK:STDOUT:   %B: <function> = fn_decl @B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B();
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc4: type = tuple_type ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.NS = %.2, .a = %a, .b = %b, .package_a = %package_a, .package_b = %package_b}
// CHECK:STDOUT:   %.2: <namespace> = namespace NS, {.ChildNS = %.3, .A = %.4}
// CHECK:STDOUT:   %.3: <namespace> = namespace ChildNS, {.B = %.5}
// CHECK:STDOUT:   %.4: <function> = fn_decl @.1
// CHECK:STDOUT:   %.5: <function> = fn_decl @.2
// CHECK:STDOUT:   %.loc4_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc4_9.2: type = converted %.loc4_9.1, constants.%.loc4
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %NS.ref.loc4: <namespace> = name_ref NS, %.2
// CHECK:STDOUT:   %A.ref.loc4: <function> = name_ref A, %.4
// CHECK:STDOUT:   %.loc4_17: init () = call %A.ref.loc4()
// CHECK:STDOUT:   assign %a.var, %.loc4_17
// CHECK:STDOUT:   %.loc5_9: () = tuple_literal ()
// CHECK:STDOUT:   %.loc4_9.3: type = converted %.loc5_9, constants.%.loc4
// CHECK:STDOUT:   %b.var: ref () = var b
// CHECK:STDOUT:   %b: ref () = bind_name b, %b.var
// CHECK:STDOUT:   %NS.ref.loc5: <namespace> = name_ref NS, %.2
// CHECK:STDOUT:   %ChildNS.ref.loc5: <namespace> = name_ref ChildNS, %.3
// CHECK:STDOUT:   %B.ref.loc5: <function> = name_ref B, %.5
// CHECK:STDOUT:   %.loc5_25: init () = call %B.ref.loc5()
// CHECK:STDOUT:   assign %b.var, %.loc5_25
// CHECK:STDOUT:   %.loc7_17: () = tuple_literal ()
// CHECK:STDOUT:   %.loc4_9.4: type = converted %.loc7_17, constants.%.loc4
// CHECK:STDOUT:   %package_a.var: ref () = var package_a
// CHECK:STDOUT:   %package_a: ref () = bind_name package_a, %package_a.var
// CHECK:STDOUT:   %package.ref.loc7: <namespace> = name_ref package, package
// CHECK:STDOUT:   %NS.ref.loc7: <namespace> = name_ref NS, %.2
// CHECK:STDOUT:   %A.ref.loc7: <function> = name_ref A, %.4
// CHECK:STDOUT:   %.loc7_33: init () = call %A.ref.loc7()
// CHECK:STDOUT:   assign %package_a.var, %.loc7_33
// CHECK:STDOUT:   %.loc8_17: () = tuple_literal ()
// CHECK:STDOUT:   %.loc4_9.5: type = converted %.loc8_17, constants.%.loc4
// CHECK:STDOUT:   %package_b.var: ref () = var package_b
// CHECK:STDOUT:   %package_b: ref () = bind_name package_b, %package_b.var
// CHECK:STDOUT:   %package.ref.loc8: <namespace> = name_ref package, package
// CHECK:STDOUT:   %NS.ref.loc8: <namespace> = name_ref NS, %.2
// CHECK:STDOUT:   %ChildNS.ref.loc8: <namespace> = name_ref ChildNS, %.3
// CHECK:STDOUT:   %B.ref.loc8: <function> = name_ref B, %.5
// CHECK:STDOUT:   %.loc8_41: init () = call %B.ref.loc8()
// CHECK:STDOUT:   assign %package_b.var, %.loc8_41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.2();
// CHECK:STDOUT:
