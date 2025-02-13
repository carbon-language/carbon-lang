// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/no_prelude/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/no_prelude/import.carbon

// --- core.carbon

package Core library "[[@TEST_NAME]]";


fn IntLiteral() -> type = "int_literal.make_type";
fn Int(N: IntLiteral()) -> type = "int.make_type_signed";

fn AsI32(a: IntLiteral()) -> i32 = "int.convert_checked";
fn AsIntLiteral(a: i32) -> IntLiteral() = "int.convert_checked";

fn TestAdd(a: i32, b: i32) -> i32 = "int.sadd";

// --- use.carbon

import Core library "core";

var arr: [i32; Core.AsIntLiteral(Core.TestAdd(Core.AsI32(1), Core.AsI32(2)))] = (Core.AsI32(1), Core.AsI32(2), Core.AsI32(3));

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %AsI32.type: type = fn_type @AsI32 [concrete]
// CHECK:STDOUT:   %AsI32: %AsI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %AsIntLiteral.type: type = fn_type @AsIntLiteral [concrete]
// CHECK:STDOUT:   %AsIntLiteral: %AsIntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %TestAdd.type: type = fn_type @TestAdd [concrete]
// CHECK:STDOUT:   %TestAdd: %TestAdd.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .AsI32 = %AsI32.decl
// CHECK:STDOUT:     .AsIntLiteral = %AsIntLiteral.decl
// CHECK:STDOUT:     .TestAdd = %TestAdd.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [concrete = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [concrete = constants.%Int] {
// CHECK:STDOUT:     %N.patt: Core.IntLiteral = binding_pattern N
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc6_22.1: type = splice_block %.loc6_22.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_22.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_22.3: type = converted %int_literal.make_type, %.loc6_22.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N: Core.IntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AsI32.decl: %AsI32.type = fn_decl @AsI32 [concrete = constants.%AsI32] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32.builtin = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32.builtin = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc8_30.1: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc8_30.2: type = converted %int.make_type_signed, %.loc8_30.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc8_24.1: type = splice_block %.loc8_24.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc8_24.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc8_24.3: type = converted %int_literal.make_type, %.loc8_24.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AsIntLiteral.decl: %AsIntLiteral.type = fn_decl @AsIntLiteral [concrete = constants.%AsIntLiteral] {
// CHECK:STDOUT:     %a.patt: %i32.builtin = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32.builtin = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc9_39.1: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc9_39.2: type = converted %int_literal.make_type, %.loc9_39.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: %i32.builtin = value_param runtime_param0
// CHECK:STDOUT:     %.loc9_20.1: type = splice_block %.loc9_20.3 [concrete = constants.%i32.builtin] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc9_20.2: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc9_20.3: type = converted %int.make_type_signed, %.loc9_20.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32.builtin = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAdd.decl: %TestAdd.type = fn_decl @TestAdd [concrete = constants.%TestAdd] {
// CHECK:STDOUT:     %a.patt: %i32.builtin = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32.builtin = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32.builtin = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32.builtin = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32.builtin = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32.builtin = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_31: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_31: init type = call constants.%Int(%int_32.loc11_31) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc11_31.1: type = value_of_initializer %int.make_type_signed.loc11_31 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc11_31.2: type = converted %int.make_type_signed.loc11_31, %.loc11_31.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %a.param: %i32.builtin = value_param runtime_param0
// CHECK:STDOUT:     %.loc11_15.1: type = splice_block %.loc11_15.3 [concrete = constants.%i32.builtin] {
// CHECK:STDOUT:       %int_32.loc11_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %int.make_type_signed.loc11_15: init type = call constants.%Int(%int_32.loc11_15) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc11_15.2: type = value_of_initializer %int.make_type_signed.loc11_15 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc11_15.3: type = converted %int.make_type_signed.loc11_15, %.loc11_15.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32.builtin = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32.builtin = value_param runtime_param1
// CHECK:STDOUT:     %.loc11_23.1: type = splice_block %.loc11_23.3 [concrete = constants.%i32.builtin] {
// CHECK:STDOUT:       %int_32.loc11_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %int.make_type_signed.loc11_23: init type = call constants.%Int(%int_32.loc11_23) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc11_23.2: type = value_of_initializer %int.make_type_signed.loc11_23 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc11_23.3: type = converted %int.make_type_signed.loc11_23, %.loc11_23.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32.builtin = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AsI32(%a.param_patt: Core.IntLiteral) -> %i32.builtin = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AsIntLiteral(%a.param_patt: %i32.builtin) -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAdd(%a.param_patt: %i32.builtin, %b.param_patt: %i32.builtin) -> %i32.builtin = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %AsIntLiteral.type: type = fn_type @AsIntLiteral [concrete]
// CHECK:STDOUT:   %AsIntLiteral: %AsIntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %TestAdd.type: type = fn_type @TestAdd [concrete]
// CHECK:STDOUT:   %TestAdd: %TestAdd.type = struct_value () [concrete]
// CHECK:STDOUT:   %AsI32.type: type = fn_type @AsI32 [concrete]
// CHECK:STDOUT:   %AsI32: %AsI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_1.f38: %i32.builtin = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_2.5a1: %i32.builtin = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.a0f: %i32.builtin = int_value 3 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32.builtin [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%i32.builtin, %i32.builtin, %i32.builtin) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.f38, %int_2.5a1, %int_3.a0f) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .AsIntLiteral = %Core.AsIntLiteral
// CHECK:STDOUT:     .TestAdd = %Core.TestAdd
// CHECK:STDOUT:     .AsI32 = %Core.AsI32
// CHECK:STDOUT:     import Core//core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.AsIntLiteral: %AsIntLiteral.type = import_ref Core//core, AsIntLiteral, loaded [concrete = constants.%AsIntLiteral]
// CHECK:STDOUT:   %Core.TestAdd: %TestAdd.type = import_ref Core//core, TestAdd, loaded [concrete = constants.%TestAdd]
// CHECK:STDOUT:   %Core.AsI32: %AsI32.type = import_ref Core//core, AsI32, loaded [concrete = constants.%AsI32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %arr.patt: %array_type = binding_pattern arr
// CHECK:STDOUT:     %.loc4_1: %array_type = var_pattern %arr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr.var: ref %array_type = var arr
// CHECK:STDOUT:   %.loc4_77: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %Core.ref.loc4_16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %AsIntLiteral.ref: %AsIntLiteral.type = name_ref AsIntLiteral, imports.%Core.AsIntLiteral [concrete = constants.%AsIntLiteral]
// CHECK:STDOUT:     %Core.ref.loc4_34: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %TestAdd.ref: %TestAdd.type = name_ref TestAdd, imports.%Core.TestAdd [concrete = constants.%TestAdd]
// CHECK:STDOUT:     %Core.ref.loc4_47: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %AsI32.ref.loc4_51: %AsI32.type = name_ref AsI32, imports.%Core.AsI32 [concrete = constants.%AsI32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int.convert_checked.loc4_59: init %i32.builtin = call %AsI32.ref.loc4_51(%int_1) [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %Core.ref.loc4_62: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %AsI32.ref.loc4_66: %AsI32.type = name_ref AsI32, imports.%Core.AsI32 [concrete = constants.%AsI32]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %int.convert_checked.loc4_74: init %i32.builtin = call %AsI32.ref.loc4_66(%int_2) [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc4_59.1: %i32.builtin = value_of_initializer %int.convert_checked.loc4_59 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc4_59.2: %i32.builtin = converted %int.convert_checked.loc4_59, %.loc4_59.1 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc4_74.1: %i32.builtin = value_of_initializer %int.convert_checked.loc4_74 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc4_74.2: %i32.builtin = converted %int.convert_checked.loc4_74, %.loc4_74.1 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %int.sadd: init %i32.builtin = call %TestAdd.ref(%.loc4_59.2, %.loc4_74.2) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %.loc4_75.1: %i32.builtin = value_of_initializer %int.sadd [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %.loc4_75.2: %i32.builtin = converted %int.sadd, %.loc4_75.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %int.convert_checked.loc4_76: init Core.IntLiteral = call %AsIntLiteral.ref(%.loc4_75.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc4_11.1: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc4_11.2: type = converted %int.make_type_signed, %.loc4_11.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc4_76.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc4_76 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc4_76.2: Core.IntLiteral = converted %int.convert_checked.loc4_76, %.loc4_76.1 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %.loc4_76.2, %i32.builtin [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr: ref %array_type = bind_name arr, %arr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AsIntLiteral(%a.param_patt: %i32.builtin) -> Core.IntLiteral = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAdd(%a.param_patt: %i32.builtin, %b.param_patt: %i32.builtin) -> %i32.builtin = "int.sadd" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AsI32(%a.param_patt: Core.IntLiteral) -> %i32.builtin = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Core.ref.loc4_82: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %AsI32.ref.loc4_86: %AsI32.type = name_ref AsI32, imports.%Core.AsI32 [concrete = constants.%AsI32]
// CHECK:STDOUT:   %int_1.loc4_93: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int.convert_checked.loc4_94: init %i32.builtin = call %AsI32.ref.loc4_86(%int_1.loc4_93) [concrete = constants.%int_1.f38]
// CHECK:STDOUT:   %Core.ref.loc4_97: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %AsI32.ref.loc4_101: %AsI32.type = name_ref AsI32, imports.%Core.AsI32 [concrete = constants.%AsI32]
// CHECK:STDOUT:   %int_2.loc4_108: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int.convert_checked.loc4_109: init %i32.builtin = call %AsI32.ref.loc4_101(%int_2.loc4_108) [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:   %Core.ref.loc4_112: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %AsI32.ref.loc4_116: %AsI32.type = name_ref AsI32, imports.%Core.AsI32 [concrete = constants.%AsI32]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int.convert_checked.loc4_124: init %i32.builtin = call %AsI32.ref.loc4_116(%int_3) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_125.1: %tuple.type = tuple_literal (%int.convert_checked.loc4_94, %int.convert_checked.loc4_109, %int.convert_checked.loc4_124)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc4_125.2: ref %i32.builtin = array_index file.%arr.var, %int_0
// CHECK:STDOUT:   %.loc4_125.3: init %i32.builtin = initialize_from %int.convert_checked.loc4_94 to %.loc4_125.2 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:   %int_1.loc4_125: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc4_125.4: ref %i32.builtin = array_index file.%arr.var, %int_1.loc4_125
// CHECK:STDOUT:   %.loc4_125.5: init %i32.builtin = initialize_from %int.convert_checked.loc4_109 to %.loc4_125.4 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:   %int_2.loc4_125: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc4_125.6: ref %i32.builtin = array_index file.%arr.var, %int_2.loc4_125
// CHECK:STDOUT:   %.loc4_125.7: init %i32.builtin = initialize_from %int.convert_checked.loc4_124 to %.loc4_125.6 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_125.8: init %array_type = array_init (%.loc4_125.3, %.loc4_125.5, %.loc4_125.7) to file.%arr.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc4_1: init %array_type = converted %.loc4_125.1, %.loc4_125.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%arr.var, %.loc4_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
