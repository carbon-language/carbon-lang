// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/reorder_fields.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/reorder_fields.carbon

fn MakeI32() -> i32;
fn MakeF64() -> f64;

fn F() -> {.a: i32, .b: f64} {
  let x: {.a: i32, .b: f64} = {.b = MakeF64(), .a = MakeI32()};
  let y: {.b: f64, .a: i32} = x;
  return y;
}

// CHECK:STDOUT: --- reorder_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %MakeI32.type: type = fn_type @MakeI32 [template]
// CHECK:STDOUT:   %MakeI32: %MakeI32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %MakeF64.type: type = fn_type @MakeF64 [template]
// CHECK:STDOUT:   %MakeF64: %MakeF64.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32, .b: f64} [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.b: f64, .a: i32} [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .Float = %import_ref.2
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
// CHECK:STDOUT:   %import_ref.2: %Float.type = import_ref Core//prelude/types, inst+32, loaded [template = constants.%Float]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .MakeI32 = %MakeI32.decl
// CHECK:STDOUT:     .MakeF64 = %MakeF64.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %MakeI32.decl: %MakeI32.type = fn_decl @MakeI32 [template = constants.%MakeI32] {
// CHECK:STDOUT:     %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_17.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:     %.loc11_17.2: type = converted %int.make_type_32.loc11, %.loc11_17.1 [template = i32]
// CHECK:STDOUT:     @MakeI32.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeF64.decl: %MakeF64.type = fn_decl @MakeF64 [template = constants.%MakeF64] {
// CHECK:STDOUT:     %.loc12_17.1: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:     %float.make_type.loc12: init type = call constants.%Float(%.loc12_17.1) [template = f64]
// CHECK:STDOUT:     %.loc12_17.2: type = value_of_initializer %float.make_type.loc12 [template = f64]
// CHECK:STDOUT:     %.loc12_17.3: type = converted %float.make_type.loc12, %.loc12_17.2 [template = f64]
// CHECK:STDOUT:     @MakeF64.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc14_16.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:     %.loc14_16.2: type = converted %int.make_type_32.loc14, %.loc14_16.1 [template = i32]
// CHECK:STDOUT:     %.loc14_25.1: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:     %float.make_type.loc14: init type = call constants.%Float(%.loc14_25.1) [template = f64]
// CHECK:STDOUT:     %.loc14_25.2: type = value_of_initializer %float.make_type.loc14 [template = f64]
// CHECK:STDOUT:     %.loc14_25.3: type = converted %float.make_type.loc14, %.loc14_25.2 [template = f64]
// CHECK:STDOUT:     %.loc14_28: type = struct_type {.a: i32, .b: f64} [template = constants.%.3]
// CHECK:STDOUT:     @F.%return: ref %.3 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeI32() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeF64() -> f64;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: %.3 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_15.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_15.2: type = converted %int.make_type_32.loc15, %.loc15_15.1 [template = i32]
// CHECK:STDOUT:   %.loc15_24.1: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:   %float.make_type.loc15: init type = call constants.%Float(%.loc15_24.1) [template = f64]
// CHECK:STDOUT:   %.loc15_24.2: type = value_of_initializer %float.make_type.loc15 [template = f64]
// CHECK:STDOUT:   %.loc15_24.3: type = converted %float.make_type.loc15, %.loc15_24.2 [template = f64]
// CHECK:STDOUT:   %.loc15_27: type = struct_type {.a: i32, .b: f64} [template = constants.%.3]
// CHECK:STDOUT:   %MakeF64.ref: %MakeF64.type = name_ref MakeF64, file.%MakeF64.decl [template = constants.%MakeF64]
// CHECK:STDOUT:   %MakeF64.call: init f64 = call %MakeF64.ref()
// CHECK:STDOUT:   %MakeI32.ref: %MakeI32.type = name_ref MakeI32, file.%MakeI32.decl [template = constants.%MakeI32]
// CHECK:STDOUT:   %MakeI32.call: init i32 = call %MakeI32.ref()
// CHECK:STDOUT:   %.loc15_62.1: %.5 = struct_literal (%MakeF64.call, %MakeI32.call)
// CHECK:STDOUT:   %.loc15_62.2: i32 = value_of_initializer %MakeI32.call
// CHECK:STDOUT:   %.loc15_62.3: i32 = converted %MakeI32.call, %.loc15_62.2
// CHECK:STDOUT:   %.loc15_62.4: f64 = value_of_initializer %MakeF64.call
// CHECK:STDOUT:   %.loc15_62.5: f64 = converted %MakeF64.call, %.loc15_62.4
// CHECK:STDOUT:   %struct.loc15: %.3 = struct_value (%.loc15_62.3, %.loc15_62.5)
// CHECK:STDOUT:   %.loc15_63: %.3 = converted %.loc15_62.1, %struct.loc15
// CHECK:STDOUT:   %x: %.3 = bind_name x, %.loc15_63
// CHECK:STDOUT:   %.loc16_15.1: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:   %float.make_type.loc16: init type = call constants.%Float(%.loc16_15.1) [template = f64]
// CHECK:STDOUT:   %.loc16_15.2: type = value_of_initializer %float.make_type.loc16 [template = f64]
// CHECK:STDOUT:   %.loc16_15.3: type = converted %float.make_type.loc16, %.loc16_15.2 [template = f64]
// CHECK:STDOUT:   %int.make_type_32.loc16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_24.1: type = value_of_initializer %int.make_type_32.loc16 [template = i32]
// CHECK:STDOUT:   %.loc16_24.2: type = converted %int.make_type_32.loc16, %.loc16_24.1 [template = i32]
// CHECK:STDOUT:   %.loc16_27: type = struct_type {.b: f64, .a: i32} [template = constants.%.5]
// CHECK:STDOUT:   %x.ref: %.3 = name_ref x, %x
// CHECK:STDOUT:   %.loc16_31.1: f64 = struct_access %x.ref, element1
// CHECK:STDOUT:   %.loc16_31.2: i32 = struct_access %x.ref, element0
// CHECK:STDOUT:   %struct.loc16: %.5 = struct_value (%.loc16_31.1, %.loc16_31.2)
// CHECK:STDOUT:   %.loc16_32: %.5 = converted %x.ref, %struct.loc16
// CHECK:STDOUT:   %y: %.5 = bind_name y, %.loc16_32
// CHECK:STDOUT:   %y.ref: %.5 = name_ref y, %y
// CHECK:STDOUT:   %.loc17_10.1: i32 = struct_access %y.ref, element1
// CHECK:STDOUT:   %.loc17_10.2: ref i32 = struct_access %return, element1
// CHECK:STDOUT:   %.loc17_10.3: init i32 = initialize_from %.loc17_10.1 to %.loc17_10.2
// CHECK:STDOUT:   %.loc17_10.4: f64 = struct_access %y.ref, element0
// CHECK:STDOUT:   %.loc17_10.5: ref f64 = struct_access %return, element0
// CHECK:STDOUT:   %.loc17_10.6: init f64 = initialize_from %.loc17_10.4 to %.loc17_10.5
// CHECK:STDOUT:   %.loc17_10.7: init %.3 = struct_init (%.loc17_10.3, %.loc17_10.6) to %return
// CHECK:STDOUT:   %.loc17_11: init %.3 = converted %y.ref, %.loc17_10.7
// CHECK:STDOUT:   return %.loc17_11 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
