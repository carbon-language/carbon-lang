// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
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
// CHECK:STDOUT:   %ptr.6eb: type = ptr_type %struct_type.x.y.z [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c.4ca: type = struct_type {.a: Core.IntLiteral, .b: %struct_type.x.y.z, .c: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.f60: type = fn_type @Op.1, @Destroy.impl(%struct_type.x.y.z) [concrete]
// CHECK:STDOUT:   %Op.1a4: %Op.type.f60 = struct_value () [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.1a4, @Op.1(%struct_type.x.y.z) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %int_32.loc15_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc15_25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc15_34: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_34: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
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
// CHECK:STDOUT:   %.loc18_26.1: ref %struct_type.x.y.z = temporary_storage
// CHECK:STDOUT:   %G.call: init %struct_type.x.y.z = call %G.ref() to %.loc18_26.1
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc18_35.1: %struct_type.a.b.c.4ca = struct_literal (%int_1, %G.call, %int_3)
// CHECK:STDOUT:   %.loc18_26.2: ref %struct_type.x.y.z = temporary %.loc18_26.1, %G.call
// CHECK:STDOUT:   %.loc18_26.3: ref %i32 = struct_access %.loc18_26.2, element0
// CHECK:STDOUT:   %.loc18_26.4: %i32 = bind_value %.loc18_26.3
// CHECK:STDOUT:   %.loc18_26.5: ref %i32 = struct_access %.loc18_26.2, element1
// CHECK:STDOUT:   %.loc18_26.6: %i32 = bind_value %.loc18_26.5
// CHECK:STDOUT:   %.loc18_26.7: ref %i32 = struct_access %.loc18_26.2, element2
// CHECK:STDOUT:   %.loc18_26.8: %i32 = bind_value %.loc18_26.7
// CHECK:STDOUT:   %struct.loc18_26: %struct_type.x.y.z = struct_value (%.loc18_26.4, %.loc18_26.6, %.loc18_26.8)
// CHECK:STDOUT:   %.loc18_35.2: %struct_type.x.y.z = converted %G.call, %struct.loc18_26
// CHECK:STDOUT:   %struct.loc18_35: %struct_type.a.b.c.4ca = struct_value (%int_1, %.loc18_35.2, %int_3)
// CHECK:STDOUT:   %.loc18_35.3: %struct_type.a.b.c.4ca = converted %.loc18_35.1, %struct.loc18_35
// CHECK:STDOUT:   %.loc18_36: %struct_type.x.y.z = struct_access %.loc18_35.3, element1
// CHECK:STDOUT:   %.loc18_38: %i32 = struct_access %.loc18_36, element1
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc18_26.1, constants.%Op.1a4
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.1a4, @Op.1(constants.%struct_type.x.y.z) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc18_26.1, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.6eb = addr_of %.loc18_26.1
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %.loc18_38
// CHECK:STDOUT: }
// CHECK:STDOUT:
