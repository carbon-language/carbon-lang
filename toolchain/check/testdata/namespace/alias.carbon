// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/alias.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/namespace/alias.carbon

namespace NS;

alias ns = NS;

fn NS.A() -> i32 { return 0; }

fn B() -> i32 { return ns.A(); }

alias C = NS.A;

fn D() -> i32 { return C(); }

// CHECK:STDOUT: --- alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %struct.2: A = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.3: B = struct_value () [template]
// CHECK:STDOUT:   %D: type = fn_type @D [template]
// CHECK:STDOUT:   %struct.4: D = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .ns = %ns
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS.ref.loc13: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:   %ns: <namespace> = bind_alias ns, %NS [template = %NS]
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc15: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc15_14.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:     %.loc15_14.2: type = converted %int.make_type_32.loc15, %.loc15_14.1 [template = i32]
// CHECK:STDOUT:     @A.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %B.decl: B = fn_decl @B [template = constants.%struct.3] {
// CHECK:STDOUT:     %int.make_type_32.loc17: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc17_11.1: type = value_of_initializer %int.make_type_32.loc17 [template = i32]
// CHECK:STDOUT:     %.loc17_11.2: type = converted %int.make_type_32.loc17, %.loc17_11.1 [template = i32]
// CHECK:STDOUT:     @B.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS.ref.loc19: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:   %A.ref: A = name_ref A, %A.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %C: A = bind_alias C, %A.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %D.decl: D = fn_decl @D [template = constants.%struct.4] {
// CHECK:STDOUT:     %int.make_type_32.loc21: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc21_11.1: type = value_of_initializer %int.make_type_32.loc21 [template = i32]
// CHECK:STDOUT:     %.loc21_11.2: type = converted %int.make_type_32.loc21, %.loc21_11.1 [template = i32]
// CHECK:STDOUT:     @D.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   return %.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ns.ref: <namespace> = name_ref ns, file.%ns [template = file.%NS]
// CHECK:STDOUT:   %A.ref: A = name_ref A, file.%A.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %A.call: init i32 = call %A.ref()
// CHECK:STDOUT:   %.loc17_30.1: i32 = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc17_30.2: i32 = converted %A.call, %.loc17_30.1
// CHECK:STDOUT:   return %.loc17_30.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: A = name_ref C, file.%C [template = constants.%struct.2]
// CHECK:STDOUT:   %A.call: init i32 = call %C.ref()
// CHECK:STDOUT:   %.loc21_27.1: i32 = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc21_27.2: i32 = converted %A.call, %.loc21_27.1
// CHECK:STDOUT:   return %.loc21_27.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
