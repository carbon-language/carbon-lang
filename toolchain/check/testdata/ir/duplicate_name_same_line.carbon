// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/ir/duplicate_name_same_line.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/ir/duplicate_name_same_line.carbon

fn A() { if (true) { var n: i32 = 1; } if (true) { var n: i32 = 2; } }

// CHECK:STDOUT: --- duplicate_name_same_line.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 2 [template]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_14: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc11_14 br !if.then.loc11_18 else br !if.else.loc11_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc11_18:
// CHECK:STDOUT:   %int.make_type_32.loc11_29: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_29.1: type = value_of_initializer %int.make_type_32.loc11_29 [template = i32]
// CHECK:STDOUT:   %.loc11_29.2: type = converted %int.make_type_32.loc11_29, %.loc11_29.1 [template = i32]
// CHECK:STDOUT:   %n.var.loc11_26: ref i32 = var n
// CHECK:STDOUT:   %n.loc11_26: ref i32 = bind_name n, %n.var.loc11_26
// CHECK:STDOUT:   %.loc11_35: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   assign %n.var.loc11_26, %.loc11_35
// CHECK:STDOUT:   br !if.else.loc11_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11_18:
// CHECK:STDOUT:   %.loc11_44: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc11_44 br !if.then.loc11_48 else br !if.else.loc11_48
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc11_48:
// CHECK:STDOUT:   %int.make_type_32.loc11_59: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_59.1: type = value_of_initializer %int.make_type_32.loc11_59 [template = i32]
// CHECK:STDOUT:   %.loc11_59.2: type = converted %int.make_type_32.loc11_59, %.loc11_59.1 [template = i32]
// CHECK:STDOUT:   %n.var.loc11_56: ref i32 = var n
// CHECK:STDOUT:   %n.loc11_56: ref i32 = bind_name n, %n.var.loc11_56
// CHECK:STDOUT:   %.loc11_65: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   assign %n.var.loc11_56, %.loc11_65
// CHECK:STDOUT:   br !if.else.loc11_48
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11_48:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
