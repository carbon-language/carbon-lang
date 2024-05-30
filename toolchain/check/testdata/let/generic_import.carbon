// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/generic_import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/let/generic_import.carbon

// --- implicit.carbon

package Implicit;

let T:! type = i32;

// --- implicit.impl.carbon

impl package Implicit;

var a: T*;
var b: T = *a;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc4_19.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc4_19.2: type = converted %int.make_type_32, %.loc4_19.1 [template = i32]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0, %.loc4_19.2 [symbolic = constants.%T]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .T = %import_ref
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: type = import_ref ir0, inst+2, loaded [symbolic = constants.%T]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %T.ref.loc4: type = name_ref T, %import_ref [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc4: type = ptr_type T [symbolic = constants.%.1]
// CHECK:STDOUT:   %a.var: ref T* = var a
// CHECK:STDOUT:   %a: ref T* = bind_name a, %a.var
// CHECK:STDOUT:   %T.ref.loc5: type = name_ref T, %import_ref [symbolic = constants.%T]
// CHECK:STDOUT:   %b.var: ref T = var b
// CHECK:STDOUT:   %b: ref T = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: ref T* = name_ref a, file.%a
// CHECK:STDOUT:   %.loc5_13: T* = bind_value %a.ref
// CHECK:STDOUT:   %.loc5_12.1: ref T = deref %.loc5_13
// CHECK:STDOUT:   %.loc5_12.2: T = bind_value %.loc5_12.1
// CHECK:STDOUT:   assign file.%b.var, %.loc5_12.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
