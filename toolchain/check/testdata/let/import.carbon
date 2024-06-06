// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/import.carbon

// --- implicit.carbon

package Implicit;

let a:! bool = true;

// --- implicit.impl.carbon

impl package Implicit;

let b:! bool = a;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Bool: <type of Bool> = struct_value () [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT:   %a: bool = bind_symbolic_name a 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref: <type of Bool> = import_ref ir7, inst+2, loaded [template = constants.%Bool]
// CHECK:STDOUT:   %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc4_9.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:   %.loc4_9.2: type = converted %bool.make_type, %.loc4_9.1 [template = bool]
// CHECK:STDOUT:   %.loc4_16: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   %a: bool = bind_symbolic_name a 0, %.loc4_16 [symbolic = constants.%a]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Bool: <type of Bool> = struct_value () [template]
// CHECK:STDOUT:   %a: bool = bind_symbolic_name a 0 [symbolic]
// CHECK:STDOUT:   %b: bool = bind_symbolic_name b 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %import_ref.1
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: bool = import_ref ir0, inst+11, loaded [symbolic = constants.%a]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.2: <type of Bool> = import_ref ir7, inst+2, loaded [template = constants.%Bool]
// CHECK:STDOUT:   %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc4_9.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:   %.loc4_9.2: type = converted %bool.make_type, %.loc4_9.1 [template = bool]
// CHECK:STDOUT:   %a.ref: bool = name_ref a, %import_ref.1 [symbolic = constants.%a]
// CHECK:STDOUT:   %b: bool = bind_symbolic_name b 0, %a.ref [symbolic = constants.%b]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
