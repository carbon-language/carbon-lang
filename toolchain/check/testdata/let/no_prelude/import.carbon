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

let a: () = ();

// --- implicit.impl.carbon

impl package Implicit;

let b: () = a;

// --- other.carbon

package Other;

let a: () = ();

// --- import_other.carbon

import Other;

let b: () = Other.a;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %empty_tuple.type = binding_pattern a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_9.1: type = splice_block %.loc4_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc4_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_9.3: type = converted %.loc4_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_14: %empty_tuple.type = converted @__global_init.%.loc4, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %a: %empty_tuple.type = bind_name a, %.loc4_14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Implicit.a: %empty_tuple.type = import_ref Implicit//default, a, loaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .a = imports.%Implicit.a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %empty_tuple.type = binding_pattern b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_9.1: type = splice_block %.loc4_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc4_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_9.3: type = converted %.loc4_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = bind_name b, @__global_init.%a.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %empty_tuple.type = name_ref a, imports.%Implicit.a
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- other.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %empty_tuple.type = binding_pattern a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_9.1: type = splice_block %.loc4_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc4_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_9.3: type = converted %.loc4_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_14: %empty_tuple.type = converted @__global_init.%.loc4, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %a: %empty_tuple.type = bind_name a, %.loc4_14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_other.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .a = %Other.a
// CHECK:STDOUT:     import Other//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.a: %empty_tuple.type = import_ref Other//default, a, loaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %empty_tuple.type = binding_pattern b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_9.1: type = splice_block %.loc4_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc4_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_9.3: type = converted %.loc4_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = bind_name b, @__global_init.%a.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:   %a.ref: %empty_tuple.type = name_ref a, imports.%Other.a
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
