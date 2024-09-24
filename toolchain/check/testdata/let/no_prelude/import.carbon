// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/no_prelude/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/no_prelude/import.carbon

// --- fail_implicit.carbon

package Implicit;

// CHECK:STDERR: fail_implicit.carbon:[[@LINE+4]]:5: error: `let` compile time bindings require function or interface (may be supported later)
// CHECK:STDERR: let a:! () = ();
// CHECK:STDERR:     ^~~~~~
// CHECK:STDERR:
let a:! () = ();

// --- fail_implicit.impl.carbon

impl package Implicit;

// CHECK:STDERR: fail_implicit.impl.carbon:[[@LINE+4]]:5: error: `let` compile time bindings require function or interface (may be supported later)
// CHECK:STDERR: let b:! () = a;
// CHECK:STDERR:     ^~~~~~
// CHECK:STDERR:
let b:! () = a;

// --- fail_other.carbon

package Other;

// CHECK:STDERR: fail_other.carbon:[[@LINE+4]]:5: error: `let` compile time bindings require function or interface (may be supported later)
// CHECK:STDERR: let a:! () = ();
// CHECK:STDERR:     ^~~~~~
// CHECK:STDERR:
let a:! () = ();

// --- fail_import_other.carbon

import Other;

// CHECK:STDERR: fail_import_other.carbon:[[@LINE+3]]:5: error: `let` compile time bindings require function or interface (may be supported later)
// CHECK:STDERR: let b:! () = Other.a;
// CHECK:STDERR:     ^~~~~~
let b:! () = Other.a;

// CHECK:STDOUT: --- fail_implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_10.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc8_10.2: type = converted %.loc8_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_15: %.1 = tuple_literal ()
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc8_16: %.1 = converted %.loc8_15, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %a: %.1 = bind_name a, %.loc8_16
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref: %.1 = import_ref Implicit//default, inst+4, loaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = imports.%import_ref
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %.loc8_10.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc8_10.2: type = converted %.loc8_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %.1 = name_ref a, imports.%import_ref
// CHECK:STDOUT:   %b: %.1 = bind_name b, %a.ref
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_10.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc8_10.2: type = converted %.loc8_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_15: %.1 = tuple_literal ()
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc8_16: %.1 = converted %.loc8_15, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %a: %.1 = bind_name a, %.loc8_16
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_other.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .a = %import_ref
// CHECK:STDOUT:     import Other//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %.1 = import_ref Other//default, inst+4, loaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %.loc7_10.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc7_10.2: type = converted %.loc7_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %a.ref: %.1 = name_ref a, imports.%import_ref
// CHECK:STDOUT:   %b: %.1 = bind_name b, %a.ref
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
