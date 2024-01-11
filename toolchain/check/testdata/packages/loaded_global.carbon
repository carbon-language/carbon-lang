// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- implicit.carbon

package Implicit api;

fn A();

// --- implicit.impl.carbon

package Implicit impl;

var a: () = A();

var package_a: () = package.A();

// --- same_package.carbon

package SamePackage api;

fn B();

// --- same_package_importer.carbon

package SamePackage library "importer" api;

import library default;

var b: () = B();

var package_b: () = package.B();

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.A = %A}
// CHECK:STDOUT:   %A: <function> = fn_decl @A, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc4: type = tuple_type (), const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.A = %.2, .a = %a, .package_a = %package_a}
// CHECK:STDOUT:   %.2: <function> = fn_decl @.1, const
// CHECK:STDOUT:   %.loc4_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc4_9.2: type = converted %.loc4_9.1, constants.%.loc4, const = constants.%.loc4
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %A.ref.loc4: <function> = name_ref A, %.2, const = %.2
// CHECK:STDOUT:   %.loc4_14: init () = call %A.ref.loc4()
// CHECK:STDOUT:   assign %a.var, %.loc4_14
// CHECK:STDOUT:   %.loc6_17: () = tuple_literal ()
// CHECK:STDOUT:   %.loc4_9.3: type = converted %.loc6_17, constants.%.loc4, const = constants.%.loc4
// CHECK:STDOUT:   %package_a.var: ref () = var package_a
// CHECK:STDOUT:   %package_a: ref () = bind_name package_a, %package_a.var
// CHECK:STDOUT:   %package.ref: <namespace> = name_ref package, package
// CHECK:STDOUT:   %A.ref.loc6: <function> = name_ref A, %.2, const = %.2
// CHECK:STDOUT:   %.loc6_30: init () = call %A.ref.loc6()
// CHECK:STDOUT:   assign %package_a.var, %.loc6_30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1();
// CHECK:STDOUT:
// CHECK:STDOUT: --- same_package.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.B = %B}
// CHECK:STDOUT:   %B: <function> = fn_decl @B, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B();
// CHECK:STDOUT:
// CHECK:STDOUT: --- same_package_importer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc6: type = tuple_type (), const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.B = %.2, .b = %b, .package_b = %package_b}
// CHECK:STDOUT:   %.2: <function> = fn_decl @.1, const
// CHECK:STDOUT:   %.loc6_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%.loc6, const = constants.%.loc6
// CHECK:STDOUT:   %b.var: ref () = var b
// CHECK:STDOUT:   %b: ref () = bind_name b, %b.var
// CHECK:STDOUT:   %B.ref.loc6: <function> = name_ref B, %.2, const = %.2
// CHECK:STDOUT:   %.loc6_14: init () = call %B.ref.loc6()
// CHECK:STDOUT:   assign %b.var, %.loc6_14
// CHECK:STDOUT:   %.loc8_17: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_9.3: type = converted %.loc8_17, constants.%.loc6, const = constants.%.loc6
// CHECK:STDOUT:   %package_b.var: ref () = var package_b
// CHECK:STDOUT:   %package_b: ref () = bind_name package_b, %package_b.var
// CHECK:STDOUT:   %package.ref: <namespace> = name_ref package, package
// CHECK:STDOUT:   %B.ref.loc8: <function> = name_ref B, %.2, const = %.2
// CHECK:STDOUT:   %.loc8_30: init () = call %B.ref.loc8()
// CHECK:STDOUT:   assign %package_b.var, %.loc8_30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1();
// CHECK:STDOUT:
