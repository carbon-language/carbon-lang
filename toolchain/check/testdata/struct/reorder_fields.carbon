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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %MakeI32.type: type = fn_type @MakeI32 [template]
// CHECK:STDOUT:   %MakeI32: %MakeI32.type = struct_value () [template]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %MakeF64.type: type = fn_type @MakeF64 [template]
// CHECK:STDOUT:   %MakeF64: %MakeF64.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: f64} [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.b.a: type = struct_type {.b: f64, .a: %i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .Float = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_17.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_17.2: type = converted %int.make_type_signed, %.loc11_17.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeF64.decl: %MakeF64.type = fn_decl @MakeF64 [template = constants.%MakeF64] {
// CHECK:STDOUT:     %return.patt: f64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: f64 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type: init type = call constants.%Float(%int_64) [template = f64]
// CHECK:STDOUT:     %.loc12_17.1: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:     %.loc12_17.2: type = converted %float.make_type, %.loc12_17.1 [template = f64]
// CHECK:STDOUT:     %return.param: ref f64 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref f64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %struct_type.a.b = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %struct_type.a.b = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc14: init type = call constants.%Int(%int_32.loc14) [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_16.1: type = value_of_initializer %int.make_type_signed.loc14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_16.2: type = converted %int.make_type_signed.loc14, %.loc14_16.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_64.loc14: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc14: init type = call constants.%Float(%int_64.loc14) [template = f64]
// CHECK:STDOUT:     %.loc14_25.1: type = value_of_initializer %float.make_type.loc14 [template = f64]
// CHECK:STDOUT:     %.loc14_25.2: type = converted %float.make_type.loc14, %.loc14_25.1 [template = f64]
// CHECK:STDOUT:     %struct_type.a.b.loc14: type = struct_type {.a: %i32, .b: f64} [template = constants.%struct_type.a.b]
// CHECK:STDOUT:     %return.param: ref %struct_type.a.b = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %struct_type.a.b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeI32() -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeF64() -> f64;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return.param_patt: %struct_type.a.b {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32.loc15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%int_32.loc15) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_15.1: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_15.2: type = converted %int.make_type_signed.loc15, %.loc15_15.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_64.loc15: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %float.make_type.loc15: init type = call constants.%Float(%int_64.loc15) [template = f64]
// CHECK:STDOUT:   %.loc15_24.1: type = value_of_initializer %float.make_type.loc15 [template = f64]
// CHECK:STDOUT:   %.loc15_24.2: type = converted %float.make_type.loc15, %.loc15_24.1 [template = f64]
// CHECK:STDOUT:   %struct_type.a.b.loc15: type = struct_type {.a: %i32, .b: f64} [template = constants.%struct_type.a.b]
// CHECK:STDOUT:   %MakeF64.ref: %MakeF64.type = name_ref MakeF64, file.%MakeF64.decl [template = constants.%MakeF64]
// CHECK:STDOUT:   %MakeF64.call: init f64 = call %MakeF64.ref()
// CHECK:STDOUT:   %MakeI32.ref: %MakeI32.type = name_ref MakeI32, file.%MakeI32.decl [template = constants.%MakeI32]
// CHECK:STDOUT:   %MakeI32.call: init %i32 = call %MakeI32.ref()
// CHECK:STDOUT:   %.loc15_62.1: %struct_type.b.a = struct_literal (%MakeF64.call, %MakeI32.call)
// CHECK:STDOUT:   %.loc15_62.2: %i32 = value_of_initializer %MakeI32.call
// CHECK:STDOUT:   %.loc15_62.3: %i32 = converted %MakeI32.call, %.loc15_62.2
// CHECK:STDOUT:   %.loc15_62.4: f64 = value_of_initializer %MakeF64.call
// CHECK:STDOUT:   %.loc15_62.5: f64 = converted %MakeF64.call, %.loc15_62.4
// CHECK:STDOUT:   %struct.loc15: %struct_type.a.b = struct_value (%.loc15_62.3, %.loc15_62.5)
// CHECK:STDOUT:   %.loc15_63: %struct_type.a.b = converted %.loc15_62.1, %struct.loc15
// CHECK:STDOUT:   %x: %struct_type.a.b = bind_name x, %.loc15_63
// CHECK:STDOUT:   %int_64.loc16: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %float.make_type.loc16: init type = call constants.%Float(%int_64.loc16) [template = f64]
// CHECK:STDOUT:   %.loc16_15.1: type = value_of_initializer %float.make_type.loc16 [template = f64]
// CHECK:STDOUT:   %.loc16_15.2: type = converted %float.make_type.loc16, %.loc16_15.1 [template = f64]
// CHECK:STDOUT:   %int_32.loc16: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc16: init type = call constants.%Int(%int_32.loc16) [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_24.1: type = value_of_initializer %int.make_type_signed.loc16 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_24.2: type = converted %int.make_type_signed.loc16, %.loc16_24.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.b.a: type = struct_type {.b: f64, .a: %i32} [template = constants.%struct_type.b.a]
// CHECK:STDOUT:   %x.ref: %struct_type.a.b = name_ref x, %x
// CHECK:STDOUT:   %.loc16_31.1: f64 = struct_access %x.ref, element1
// CHECK:STDOUT:   %.loc16_31.2: %i32 = struct_access %x.ref, element0
// CHECK:STDOUT:   %struct.loc16: %struct_type.b.a = struct_value (%.loc16_31.1, %.loc16_31.2)
// CHECK:STDOUT:   %.loc16_32: %struct_type.b.a = converted %x.ref, %struct.loc16
// CHECK:STDOUT:   %y: %struct_type.b.a = bind_name y, %.loc16_32
// CHECK:STDOUT:   %y.ref: %struct_type.b.a = name_ref y, %y
// CHECK:STDOUT:   %.loc17_10.1: %i32 = struct_access %y.ref, element1
// CHECK:STDOUT:   %.loc17_10.2: ref %i32 = struct_access %return, element1
// CHECK:STDOUT:   %.loc17_10.3: init %i32 = initialize_from %.loc17_10.1 to %.loc17_10.2
// CHECK:STDOUT:   %.loc17_10.4: f64 = struct_access %y.ref, element0
// CHECK:STDOUT:   %.loc17_10.5: ref f64 = struct_access %return, element0
// CHECK:STDOUT:   %.loc17_10.6: init f64 = initialize_from %.loc17_10.4 to %.loc17_10.5
// CHECK:STDOUT:   %.loc17_10.7: init %struct_type.a.b = struct_init (%.loc17_10.3, %.loc17_10.6) to %return
// CHECK:STDOUT:   %.loc17_11: init %struct_type.a.b = converted %y.ref, %.loc17_10.7
// CHECK:STDOUT:   return %.loc17_11 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
