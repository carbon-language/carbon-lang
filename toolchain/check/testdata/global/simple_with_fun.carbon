// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/global/simple_with_fun.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/global/simple_with_fun.carbon

fn test_a() -> i32 {
  return 0;
}

var a: i32 = test_a();

// CHECK:STDOUT: --- simple_with_fun.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %test_a.type: type = fn_type @test_a [template]
// CHECK:STDOUT:   %test_a: %test_a.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
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
// CHECK:STDOUT:     .test_a = %test_a.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %test_a.decl: %test_a.type = fn_decl @test_a [template = constants.%test_a] {
// CHECK:STDOUT:     %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_16.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:     %.loc11_16.2: type = converted %int.make_type_32.loc11, %.loc11_16.1 [template = i32]
// CHECK:STDOUT:     @test_a.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_8.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_8.2: type = converted %int.make_type_32.loc15, %.loc15_8.1 [template = i32]
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @test_a() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   return %.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %test_a.ref: %test_a.type = name_ref test_a, file.%test_a.decl [template = constants.%test_a]
// CHECK:STDOUT:   %test_a.call: init i32 = call %test_a.ref()
// CHECK:STDOUT:   assign file.%a.var, %test_a.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
