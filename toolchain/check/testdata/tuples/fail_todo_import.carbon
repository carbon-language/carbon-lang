// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// When there are no more cases that can hit a TODO, remove this test. Until
// then, update it whenever its target is implemented.
//
// AUTOUPDATE
// CHECK:STDERR: implicit.impl.carbon: ERROR: Semantics TODO: `TODO: ResolveIfImportRefUnused for non-builtin type`.

// --- implicit.carbon

package Implicit api;

var a_ref: (i32,) = (0,);

// --- implicit.impl.carbon

package Implicit impl;

var a: (i32,) = a_ref;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.4: (i32,) = tuple_value (%.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a_ref = %a_ref} [template]
// CHECK:STDOUT:   %.loc4_17.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc4_17.2: type = converted %.loc4_17.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %a_ref.var: ref (i32,) = var a_ref
// CHECK:STDOUT:   %a_ref: ref (i32,) = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %.loc4_22: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_24.1: (i32,) = tuple_literal (%.loc4_22)
// CHECK:STDOUT:   %.loc4_24.2: init (i32,) = tuple_init (%.loc4_22) to %a_ref.var [template = constants.%.4]
// CHECK:STDOUT:   %.loc4_24.3: init (i32,) = converted %.loc4_24.1, %.loc4_24.2 [template = constants.%.4]
// CHECK:STDOUT:   assign %a_ref.var, %.loc4_24.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a_ref = %import_ref, .a = %a} [template]
// CHECK:STDOUT:   %import_ref: ref <error> = import_ref ir1, inst+6, used
// CHECK:STDOUT:   %.loc4_13.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc4_13.2: type = converted %.loc4_13.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref (i32,) = var a
// CHECK:STDOUT:   %a: ref (i32,) = bind_name a, %a.var
// CHECK:STDOUT:   %a_ref.ref: ref <error> = name_ref a_ref, %import_ref
// CHECK:STDOUT:   assign %a.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
