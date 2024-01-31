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

// CHECK:STDERR: implicit.impl.carbon:[[@LINE+3]]:13: ERROR: Value of type `<function>` is not callable.
// CHECK:STDERR: var a: () = A();
// CHECK:STDERR:             ^~
var a: () = A();

// CHECK:STDERR: implicit.impl.carbon:[[@LINE+3]]:21: ERROR: Value of type `<function>` is not callable.
// CHECK:STDERR: var package_a: () = package.A();
// CHECK:STDERR:                     ^~~~~~~~~~
var package_a: () = package.A();

// --- same_package.carbon

package SamePackage api;

fn B();

// --- same_package_importer.carbon

package SamePackage library "importer" api;

import library default;

// CHECK:STDERR: same_package_importer.carbon:[[@LINE+3]]:13: ERROR: Value of type `<function>` is not callable.
// CHECK:STDERR: var b: () = B();
// CHECK:STDERR:             ^~
var b: () = B();

// CHECK:STDERR: same_package_importer.carbon:[[@LINE+3]]:21: ERROR: Value of type `<function>` is not callable.
// CHECK:STDERR: var package_b: () = package.B();
// CHECK:STDERR:                     ^~~~~~~~~~
var package_b: () = package.B();

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.A = %A} [template]
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.A = %import_ref, .a = %a, .package_a = %package_a} [template]
// CHECK:STDOUT:   %import_ref: <function> = import_ref ir1, inst+1, used
// CHECK:STDOUT:   %.loc7_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc7_9.2: type = converted %.loc7_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %A.ref.loc7: <function> = name_ref A, %import_ref
// CHECK:STDOUT:   assign %a.var, <error>
// CHECK:STDOUT:   %.loc12_17.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc12_17.2: type = converted %.loc12_17.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %package_a.var: ref () = var package_a
// CHECK:STDOUT:   %package_a: ref () = bind_name package_a, %package_a.var
// CHECK:STDOUT:   %package.ref: <namespace> = name_ref package, package [template = package]
// CHECK:STDOUT:   %A.ref.loc12: <function> = name_ref A, %import_ref
// CHECK:STDOUT:   assign %package_a.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- same_package.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.B = %B} [template]
// CHECK:STDOUT:   %B: <function> = fn_decl @B [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B();
// CHECK:STDOUT:
// CHECK:STDOUT: --- same_package_importer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.B = %import_ref, .b = %b, .package_b = %package_b} [template]
// CHECK:STDOUT:   %import_ref: <function> = import_ref ir1, inst+1, used
// CHECK:STDOUT:   %.loc9_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_9.2: type = converted %.loc9_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %b.var: ref () = var b
// CHECK:STDOUT:   %b: ref () = bind_name b, %b.var
// CHECK:STDOUT:   %B.ref.loc9: <function> = name_ref B, %import_ref
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT:   %.loc14_17.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc14_17.2: type = converted %.loc14_17.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %package_b.var: ref () = var package_b
// CHECK:STDOUT:   %package_b: ref () = bind_name package_b, %package_b.var
// CHECK:STDOUT:   %package.ref: <namespace> = name_ref package, package [template = package]
// CHECK:STDOUT:   %B.ref.loc14: <function> = name_ref B, %import_ref
// CHECK:STDOUT:   assign %package_b.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
