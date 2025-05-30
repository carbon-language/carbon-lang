// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/literal_member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/literal_member_access.carbon

fn G() -> {.x: i32, .y: i32, .z: i32};

fn F() -> i32 {
  return {.a = 1, .b = G(), .c = 3}.b.y;
}

// CHECK:STDOUT: --- literal_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.x.y.z: type = struct_type {.x: %i32, .y: %i32, .z: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %struct_type.x.y.z [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c.4ca: type = struct_type {.a: Core.IntLiteral, .b: %struct_type.x.y.z, .c: Core.IntLiteral} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc14_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc14_25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc14_34: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_34: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.x.y.z: type = struct_type {.x: %i32, .y: %i32, .z: %i32} [concrete = constants.%struct_type.x.y.z]
// CHECK:STDOUT:     %return.param: ref %struct_type.x.y.z = out_param call_param0
// CHECK:STDOUT:     %return: ref %struct_type.x.y.z = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return.param: %struct_type.x.y.z;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc17_26.1: ref %struct_type.x.y.z = temporary_storage
// CHECK:STDOUT:   %G.call: init %struct_type.x.y.z = call %G.ref() to %.loc17_26.1
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc17_35.1: %struct_type.a.b.c.4ca = struct_literal (%int_1, %G.call, %int_3)
// CHECK:STDOUT:   %.loc17_26.2: ref %struct_type.x.y.z = temporary %.loc17_26.1, %G.call
// CHECK:STDOUT:   %.loc17_26.3: ref %i32 = struct_access %.loc17_26.2, element0
// CHECK:STDOUT:   %.loc17_26.4: %i32 = bind_value %.loc17_26.3
// CHECK:STDOUT:   %.loc17_26.5: ref %i32 = struct_access %.loc17_26.2, element1
// CHECK:STDOUT:   %.loc17_26.6: %i32 = bind_value %.loc17_26.5
// CHECK:STDOUT:   %.loc17_26.7: ref %i32 = struct_access %.loc17_26.2, element2
// CHECK:STDOUT:   %.loc17_26.8: %i32 = bind_value %.loc17_26.7
// CHECK:STDOUT:   %struct.loc17_26: %struct_type.x.y.z = struct_value (%.loc17_26.4, %.loc17_26.6, %.loc17_26.8)
// CHECK:STDOUT:   %.loc17_35.2: %struct_type.x.y.z = converted %G.call, %struct.loc17_26
// CHECK:STDOUT:   %struct.loc17_35: %struct_type.a.b.c.4ca = struct_value (%int_1, %.loc17_35.2, %int_3)
// CHECK:STDOUT:   %.loc17_35.3: %struct_type.a.b.c.4ca = converted %.loc17_35.1, %struct.loc17_35
// CHECK:STDOUT:   %.loc17_36: %struct_type.x.y.z = struct_access %.loc17_35.3, element1
// CHECK:STDOUT:   %.loc17_38: %i32 = struct_access %.loc17_36, element1
// CHECK:STDOUT:   return %.loc17_38
// CHECK:STDOUT: }
// CHECK:STDOUT:
