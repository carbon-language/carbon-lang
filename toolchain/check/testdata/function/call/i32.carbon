// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/i32.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/i32.carbon

fn Echo(a: i32) -> i32 {
  return a;
}

fn Main() {
  var b: i32 = Echo(1);
}

// CHECK:STDOUT: --- i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Echo.type: type = fn_type @Echo [template]
// CHECK:STDOUT:   %Echo: %Echo.type = struct_value () [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
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
// CHECK:STDOUT:     .Echo = %Echo.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Echo.decl: %Echo.type = fn_decl @Echo [template = constants.%Echo] {
// CHECK:STDOUT:     %int.make_type_32.loc11_12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_12.1: type = value_of_initializer %int.make_type_32.loc11_12 [template = i32]
// CHECK:STDOUT:     %.loc11_12.2: type = converted %int.make_type_32.loc11_12, %.loc11_12.1 [template = i32]
// CHECK:STDOUT:     %a.loc11_9.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     @Echo.%a: i32 = bind_name a, %a.loc11_9.1
// CHECK:STDOUT:     %int.make_type_32.loc11_20: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_20.1: type = value_of_initializer %int.make_type_32.loc11_20 [template = i32]
// CHECK:STDOUT:     %.loc11_20.2: type = converted %int.make_type_32.loc11_20, %.loc11_20.1 [template = i32]
// CHECK:STDOUT:     @Echo.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Echo(%a: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   return %a.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc16_10.2: type = converted %int.make_type_32, %.loc16_10.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %Echo.ref: %Echo.type = name_ref Echo, file.%Echo.decl [template = constants.%Echo]
// CHECK:STDOUT:   %.loc16_21: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %Echo.call: init i32 = call %Echo.ref(%.loc16_21)
// CHECK:STDOUT:   assign %b.var, %Echo.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
