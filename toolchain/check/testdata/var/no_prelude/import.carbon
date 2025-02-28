// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/no_prelude/import.carbon

// --- implicit.carbon

package Implicit;

var a_ref: () = ();

// --- implicit.impl.carbon

impl package Implicit;

var a: () = a_ref;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_ref.patt: %empty_tuple.type = binding_pattern a_ref
// CHECK:STDOUT:     %.loc4_1: %empty_tuple.type = var_pattern %a_ref.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ref.var: ref %empty_tuple.type = var a_ref
// CHECK:STDOUT:   %.loc4_13.1: type = splice_block %.loc4_13.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc4_13.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_13.3: type = converted %.loc4_13.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ref: ref %empty_tuple.type = bind_name a_ref, %a_ref.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_18.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_18.2: init %empty_tuple.type = tuple_init () to file.%a_ref.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_1: init %empty_tuple.type = converted %.loc4_18.1, %.loc4_18.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc4_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Implicit.a_ref: ref %empty_tuple.type = import_ref Implicit//default, a_ref, loaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .a_ref = imports.%Implicit.a_ref
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %empty_tuple.type = binding_pattern a
// CHECK:STDOUT:     %.loc4_1: %empty_tuple.type = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %.loc4_9.1: type = splice_block %.loc4_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc4_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_9.3: type = converted %.loc4_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %empty_tuple.type = name_ref a_ref, imports.%Implicit.a_ref
// CHECK:STDOUT:   %.loc4_13: init %empty_tuple.type = tuple_init () to file.%a.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_1: init %empty_tuple.type = converted %a_ref.ref, %.loc4_13 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc4_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
