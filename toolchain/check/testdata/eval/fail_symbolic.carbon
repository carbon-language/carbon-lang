// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// TODO: This should work.
fn G(N:! i32) {
  // CHECK:STDERR: fail_symbolic.carbon:[[@LINE+3]]:16: ERROR: Semantics TODO: `symbolic array bound`.
  // CHECK:STDERR:   var k: [i32; N];
  // CHECK:STDERR:                ^
  var k: [i32; N];
}

// CHECK:STDOUT: --- fail_symbolic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N 0 [symbolic]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.2: G = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc8_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc8_10.2: type = converted %int.make_type_32, %.loc8_10.1 [template = i32]
// CHECK:STDOUT:     %N.loc8_6.1: i32 = param N
// CHECK:STDOUT:     @G.%N: i32 = bind_symbolic_name N 0, %N.loc8_6.1 [symbolic = constants.%N]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%N: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %N.ref: i32 = name_ref N, %N [symbolic = constants.%N]
// CHECK:STDOUT:   %.loc12_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc12_11.2: type = converted %int.make_type_32, %.loc12_11.1 [template = i32]
// CHECK:STDOUT:   %.loc12_17: type = array_type %N.ref, i32 [template = <error>]
// CHECK:STDOUT:   %k.var: ref <error> = var k
// CHECK:STDOUT:   %k: ref <error> = bind_name k, %k.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
