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

fn Int32() -> type = "int.make_type_32";
fn TestAdd(a: i32, b: i32) -> i32 = "int.sadd";

fn IntLiteral() -> type = "int_literal.make_type";
fn AsIntLiteral(a: i32) -> IntLiteral() = "int.convert_checked";

// --- use.carbon

import Core library "core";

var arr: [i32; Core.AsIntLiteral(Core.TestAdd(1, 2))] = (1, 2, 3);

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %TestAdd.type: type = fn_type @TestAdd [template]
// CHECK:STDOUT:   %TestAdd: %TestAdd.type = struct_value () [template]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %AsIntLiteral.type: type = fn_type @AsIntLiteral [template]
// CHECK:STDOUT:   %AsIntLiteral: %AsIntLiteral.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int32 = %Int32.decl
// CHECK:STDOUT:     .TestAdd = %TestAdd.decl
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .AsIntLiteral = %AsIntLiteral.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32.decl: %Int32.type = fn_decl @Int32 [template = constants.%Int32] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAdd.decl: %TestAdd.type = fn_decl @TestAdd [template = constants.%TestAdd] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc5_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_15.1: type = value_of_initializer %int.make_type_32.loc5_15 [template = i32]
// CHECK:STDOUT:     %.loc5_15.2: type = converted %int.make_type_32.loc5_15, %.loc5_15.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc5_23: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_23.1: type = value_of_initializer %int.make_type_32.loc5_23 [template = i32]
// CHECK:STDOUT:     %.loc5_23.2: type = converted %int.make_type_32.loc5_23, %.loc5_23.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc5_31: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_31.1: type = value_of_initializer %int.make_type_32.loc5_31 [template = i32]
// CHECK:STDOUT:     %.loc5_31.2: type = converted %int.make_type_32.loc5_31, %.loc5_31.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [template = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AsIntLiteral.decl: %AsIntLiteral.type = fn_decl @AsIntLiteral [template = constants.%AsIntLiteral] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc8_20.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc8_20.2: type = converted %int.make_type_32, %.loc8_20.1 [template = i32]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc8_39.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc8_39.2: type = converted %int_literal.make_type, %.loc8_39.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAdd(%a.param_patt: i32, %b.param_patt: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AsIntLiteral(%a.param_patt: i32) -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %AsIntLiteral.type: type = fn_type @AsIntLiteral [template]
// CHECK:STDOUT:   %AsIntLiteral: %AsIntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %TestAdd.type: type = fn_type @TestAdd [template]
// CHECK:STDOUT:   %TestAdd: %TestAdd.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.3: i32 = int_value 3 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.5: type = array_type %.4, i32 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.7: i32 = int_value 0 [template]
// CHECK:STDOUT:   %array: %.5 = tuple_value (%.1, %.2, %.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .AsIntLiteral = %import_ref.2
// CHECK:STDOUT:     .TestAdd = %import_ref.3
// CHECK:STDOUT:     import Core//core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: %AsIntLiteral.type = import_ref Core//core, inst+55, loaded [template = constants.%AsIntLiteral]
// CHECK:STDOUT:   %import_ref.3: %TestAdd.type = import_ref Core//core, inst+30, loaded [template = constants.%TestAdd]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %Core.ref.loc4_16: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:   %AsIntLiteral.ref: %AsIntLiteral.type = name_ref AsIntLiteral, imports.%import_ref.2 [template = constants.%AsIntLiteral]
// CHECK:STDOUT:   %Core.ref.loc4_34: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:   %TestAdd.ref: %TestAdd.type = name_ref TestAdd, imports.%import_ref.3 [template = constants.%TestAdd]
// CHECK:STDOUT:   %.loc4_47: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc4_50: i32 = int_value 2 [template = constants.%.2]
// CHECK:STDOUT:   %int.sadd: init i32 = call %TestAdd.ref(%.loc4_47, %.loc4_50) [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_46.1: i32 = value_of_initializer %int.sadd [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_46.2: i32 = converted %int.sadd, %.loc4_46.1 [template = constants.%.3]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %AsIntLiteral.ref(%.loc4_46.2) [template = constants.%.4]
// CHECK:STDOUT:   %.loc4_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc4_11.2: type = converted %int.make_type_32, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:   %.loc4_33.1: Core.IntLiteral = value_of_initializer %int.convert_checked [template = constants.%.4]
// CHECK:STDOUT:   %.loc4_33.2: Core.IntLiteral = converted %int.convert_checked, %.loc4_33.1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc4_53: type = array_type %.loc4_33.2, i32 [template = constants.%.5]
// CHECK:STDOUT:   %arr.var: ref %.5 = var arr
// CHECK:STDOUT:   %arr: ref %.5 = bind_name arr, %arr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AsIntLiteral(%a.param_patt: i32) -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAdd(%a.param_patt: i32, %b.param_patt: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_58: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc4_61: i32 = int_value 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_64: i32 = int_value 3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_65.1: %tuple.type = tuple_literal (%.loc4_58, %.loc4_61, %.loc4_64)
// CHECK:STDOUT:   %.loc4_65.2: i32 = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc4_65.3: ref i32 = array_index file.%arr.var, %.loc4_65.2
// CHECK:STDOUT:   %.loc4_65.4: init i32 = initialize_from %.loc4_58 to %.loc4_65.3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc4_65.5: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc4_65.6: ref i32 = array_index file.%arr.var, %.loc4_65.5
// CHECK:STDOUT:   %.loc4_65.7: init i32 = initialize_from %.loc4_61 to %.loc4_65.6 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_65.8: i32 = int_value 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_65.9: ref i32 = array_index file.%arr.var, %.loc4_65.8
// CHECK:STDOUT:   %.loc4_65.10: init i32 = initialize_from %.loc4_64 to %.loc4_65.9 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_65.11: init %.5 = array_init (%.loc4_65.4, %.loc4_65.7, %.loc4_65.10) to file.%arr.var [template = constants.%array]
// CHECK:STDOUT:   %.loc4_66: init %.5 = converted %.loc4_65.1, %.loc4_65.11 [template = constants.%array]
// CHECK:STDOUT:   assign file.%arr.var, %.loc4_66
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
