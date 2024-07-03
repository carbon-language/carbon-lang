// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/no_prelude/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/no_prelude/import.carbon

// --- implicit.carbon

package Implicit;

let a:! () = ();

// --- implicit.impl.carbon

impl package Implicit;

let b:! () = a;

// --- other.carbon

package Other;

let a:! () = ();

// --- import_other.carbon

import Other;

let b:! () = Other.a;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %a: %.1 = bind_symbolic_name a 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_10.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc4_10.2: type = converted %.loc4_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_15: %.1 = tuple_literal ()
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc4_16: %.1 = converted %.loc4_15, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %a: %.1 = bind_symbolic_name a 0, %.loc4_16 [symbolic = constants.%a]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %a: %.1 = bind_symbolic_name a 0 [symbolic]
// CHECK:STDOUT:   %b: %.1 = bind_symbolic_name b 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %import_ref
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %import_ref: %.1 = import_ref ir0, inst+4, loaded [symbolic = constants.%a]
// CHECK:STDOUT:   %.loc4_10.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc4_10.2: type = converted %.loc4_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %.1 = name_ref a, file.%import_ref [symbolic = constants.%a]
// CHECK:STDOUT:   %b: %.1 = bind_symbolic_name b 0, %a.ref [symbolic = constants.%b]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- other.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %a: %.1 = bind_symbolic_name a 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_10.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc4_10.2: type = converted %.loc4_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_15: %.1 = tuple_literal ()
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc4_16: %.1 = converted %.loc4_15, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %a: %.1 = bind_symbolic_name a 0, %.loc4_16 [symbolic = constants.%a]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_other.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %a: %.1 = bind_symbolic_name a 0 [symbolic]
// CHECK:STDOUT:   %b: %.1 = bind_symbolic_name b 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other: <namespace> = namespace %Other.import, [template] {}
// CHECK:STDOUT:   %.loc4_10.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc4_10.2: type = converted %.loc4_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %import_ref: %.1 = import_ref ir1, inst+4, loaded [symbolic = constants.%a]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, file.%Other [template = file.%Other]
// CHECK:STDOUT:   %a.ref: %.1 = name_ref a, file.%import_ref [symbolic = constants.%a]
// CHECK:STDOUT:   %b: %.1 = bind_symbolic_name b 0, %a.ref [symbolic = constants.%b]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
