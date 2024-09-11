// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/alias.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/alias.carbon

namespace NS;

alias ns = NS;

fn NS.A() -> i32 { return 0; }

fn B() -> i32 { return ns.A(); }

alias C = NS.A;

fn D() -> i32 { return C(); }

// CHECK:STDOUT: --- alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: %D.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .ns = %ns
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS.ref.loc13: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:   %ns: <namespace> = bind_alias ns, %NS [template = %NS]
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {
// CHECK:STDOUT:     %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_14.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:     %.loc15_14.2: type = converted %int.make_type_32.loc15, %.loc15_14.1 [template = i32]
// CHECK:STDOUT:     @A.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %int.make_type_32.loc17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc17_11.1: type = value_of_initializer %int.make_type_32.loc17 [template = i32]
// CHECK:STDOUT:     %.loc17_11.2: type = converted %int.make_type_32.loc17, %.loc17_11.1 [template = i32]
// CHECK:STDOUT:     @B.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS.ref.loc19: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:   %C: %A.type = bind_alias C, %A.decl [template = constants.%A]
// CHECK:STDOUT:   %D.decl: %D.type = fn_decl @D [template = constants.%D] {
// CHECK:STDOUT:     %int.make_type_32.loc21: init type = call constants.%Int32() [template = i32]
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
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %A.call: init i32 = call %A.ref()
// CHECK:STDOUT:   %.loc17_30.1: i32 = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc17_30.2: i32 = converted %A.call, %.loc17_30.1
// CHECK:STDOUT:   return %.loc17_30.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: %A.type = name_ref C, file.%C [template = constants.%A]
// CHECK:STDOUT:   %A.call: init i32 = call %C.ref()
// CHECK:STDOUT:   %.loc21_27.1: i32 = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc21_27.2: i32 = converted %A.call, %.loc21_27.1
// CHECK:STDOUT:   return %.loc21_27.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
