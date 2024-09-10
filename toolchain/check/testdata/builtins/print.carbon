// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/print.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/print.carbon

fn Print(a: i32) = "print.int";

fn Main() {
  Print(1);

  Core.Print(2);
}

// CHECK:STDOUT: --- print.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Print.type.1: type = fn_type @Print.1 [template]
// CHECK:STDOUT:   %Print.1: %Print.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %Print.type.2: type = fn_type @Print.2 [template]
// CHECK:STDOUT:   %Print.2: %Print.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .Print = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %Print.type.2 = import_ref Core//prelude, inst+45, loaded [template = constants.%Print.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Print = %Print.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Print.decl: %Print.type.1 = fn_decl @Print.1 [template = constants.%Print.1] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_13.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc11_13.2: type = converted %int.make_type_32, %.loc11_13.1 [template = i32]
// CHECK:STDOUT:     %a.loc11_10.1: i32 = param a
// CHECK:STDOUT:     @Print.1.%a: i32 = bind_name a, %a.loc11_10.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Print.1(%a: i32) = "print.int";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Print.ref.loc14: %Print.type.1 = name_ref Print, file.%Print.decl [template = constants.%Print.1]
// CHECK:STDOUT:   %.loc14: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %print.int.loc14: init %.1 = call %Print.ref.loc14(%.loc14)
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:   %Print.ref.loc16: %Print.type.2 = name_ref Print, imports.%import_ref.2 [template = constants.%Print.2]
// CHECK:STDOUT:   %.loc16: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %print.int.loc16: init %.1 = call %Print.ref.loc16(%.loc16)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Print.2(%x: i32) = "print.int";
// CHECK:STDOUT:
