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
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %AsI32.type: type = fn_type @AsI32 [template]
// CHECK:STDOUT:   %AsI32: %AsI32.type = struct_value () [template]
// CHECK:STDOUT:   %AsIntLiteral.type: type = fn_type @AsIntLiteral [template]
// CHECK:STDOUT:   %AsIntLiteral: %AsIntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %TestAdd.type: type = fn_type @TestAdd [template]
// CHECK:STDOUT:   %TestAdd: %TestAdd.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .AsI32 = %AsI32.decl
// CHECK:STDOUT:     .AsIntLiteral = %AsIntLiteral.decl
// CHECK:STDOUT:     .TestAdd = %TestAdd.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [template = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [template = constants.%Int] {
// CHECK:STDOUT:     %N.patt: Core.IntLiteral = binding_pattern N
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_22.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_22.2: type = converted %int_literal.make_type, %.loc6_22.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %N: Core.IntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AsI32.decl: %AsI32.type = fn_decl @AsI32 [template = constants.%AsI32] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc8_24.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc8_24.2: type = converted %int_literal.make_type, %.loc8_24.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_30.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_30.2: type = converted %int.make_type_signed, %.loc8_30.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AsIntLiteral.decl: %AsIntLiteral.type = fn_decl @AsIntLiteral [template = constants.%AsIntLiteral] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc9_20.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc9_20.2: type = converted %int.make_type_signed, %.loc9_20.1 [template = constants.%i32]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc9_39.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc9_39.2: type = converted %int_literal.make_type, %.loc9_39.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAdd.decl: %TestAdd.type = fn_decl @TestAdd [template = constants.%TestAdd] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_15: init type = call constants.%Int(%int_32.loc11_15) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_15.1: type = value_of_initializer %int.make_type_signed.loc11_15 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_15.2: type = converted %int.make_type_signed.loc11_15, %.loc11_15.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_23: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_23: init type = call constants.%Int(%int_32.loc11_23) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_23.1: type = value_of_initializer %int.make_type_signed.loc11_23 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_23.2: type = converted %int.make_type_signed.loc11_23, %.loc11_23.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_31: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_31: init type = call constants.%Int(%int_32.loc11_31) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_31.1: type = value_of_initializer %int.make_type_signed.loc11_31 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_31.2: type = converted %int.make_type_signed.loc11_31, %.loc11_31.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AsI32(%a.param_patt: Core.IntLiteral) -> %i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AsIntLiteral(%a.param_patt: %i32) -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAdd(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %AsIntLiteral.type: type = fn_type @AsIntLiteral [template]
// CHECK:STDOUT:   %AsIntLiteral: %AsIntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %TestAdd.type: type = fn_type @TestAdd [template]
// CHECK:STDOUT:   %TestAdd: %TestAdd.type = struct_value () [template]
// CHECK:STDOUT:   %AsI32.type: type = fn_type @AsI32 [template]
// CHECK:STDOUT:   %AsI32: %AsI32.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %int_3.1: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %int_3.2: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.2, %i32 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%i32, %i32, %i32) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.2, %int_2.2, %int_3.1) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .AsIntLiteral = %import_ref.2
// CHECK:STDOUT:     .TestAdd = %import_ref.3
// CHECK:STDOUT:     .AsI32 = %import_ref.4
// CHECK:STDOUT:     import Core//core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: %AsIntLiteral.type = import_ref Core//core, inst+61, loaded [template = constants.%AsIntLiteral]
// CHECK:STDOUT:   %import_ref.3: %TestAdd.type = import_ref Core//core, inst+88, loaded [template = constants.%TestAdd]
// CHECK:STDOUT:   %import_ref.4: %AsI32.type = import_ref Core//core, inst+42, loaded [template = constants.%AsI32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %Core.ref.loc4_16: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:   %AsIntLiteral.ref: %AsIntLiteral.type = name_ref AsIntLiteral, imports.%import_ref.2 [template = constants.%AsIntLiteral]
// CHECK:STDOUT:   %Core.ref.loc4_34: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:   %TestAdd.ref: %TestAdd.type = name_ref TestAdd, imports.%import_ref.3 [template = constants.%TestAdd]
// CHECK:STDOUT:   %Core.ref.loc4_47: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:   %AsI32.ref.loc4_51: %AsI32.type = name_ref AsI32, imports.%import_ref.4 [template = constants.%AsI32]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int.convert_checked.loc4_59: init %i32 = call %AsI32.ref.loc4_51(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %Core.ref.loc4_62: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:   %AsI32.ref.loc4_66: %AsI32.type = name_ref AsI32, imports.%import_ref.4 [template = constants.%AsI32]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %int.convert_checked.loc4_74: init %i32 = call %AsI32.ref.loc4_66(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc4_59.1: %i32 = value_of_initializer %int.convert_checked.loc4_59 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc4_59.2: %i32 = converted %int.convert_checked.loc4_59, %.loc4_59.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc4_74.1: %i32 = value_of_initializer %int.convert_checked.loc4_74 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc4_74.2: %i32 = converted %int.convert_checked.loc4_74, %.loc4_74.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int.sadd: init %i32 = call %TestAdd.ref(%.loc4_59.2, %.loc4_74.2) [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc4_75.1: %i32 = value_of_initializer %int.sadd [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc4_75.2: %i32 = converted %int.sadd, %.loc4_75.1 [template = constants.%int_3.1]
// CHECK:STDOUT:   %int.convert_checked.loc4_76: init Core.IntLiteral = call %AsIntLiteral.ref(%.loc4_75.2) [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc4_11.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_11.2: type = converted %int.make_type_signed, %.loc4_11.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_76.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc4_76 [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc4_76.2: Core.IntLiteral = converted %int.convert_checked.loc4_76, %.loc4_76.1 [template = constants.%int_3.2]
// CHECK:STDOUT:   %array_type: type = array_type %.loc4_76.2, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %arr.var: ref %array_type = var arr
// CHECK:STDOUT:   %arr: ref %array_type = bind_name arr, %arr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AsIntLiteral(%a.param_patt: %i32) -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAdd(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AsI32(%a.param_patt: Core.IntLiteral) -> %i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Core.ref.loc4_82: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:   %AsI32.ref.loc4_86: %AsI32.type = name_ref AsI32, imports.%import_ref.4 [template = constants.%AsI32]
// CHECK:STDOUT:   %int_1.loc4_93: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int.convert_checked.loc4_94: init %i32 = call %AsI32.ref.loc4_86(%int_1.loc4_93) [template = constants.%int_1.2]
// CHECK:STDOUT:   %Core.ref.loc4_97: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:   %AsI32.ref.loc4_101: %AsI32.type = name_ref AsI32, imports.%import_ref.4 [template = constants.%AsI32]
// CHECK:STDOUT:   %int_2.loc4_108: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %int.convert_checked.loc4_109: init %i32 = call %AsI32.ref.loc4_101(%int_2.loc4_108) [template = constants.%int_2.2]
// CHECK:STDOUT:   %Core.ref.loc4_112: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:   %AsI32.ref.loc4_116: %AsI32.type = name_ref AsI32, imports.%import_ref.4 [template = constants.%AsI32]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.2]
// CHECK:STDOUT:   %int.convert_checked.loc4_124: init %i32 = call %AsI32.ref.loc4_116(%int_3) [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc4_125.1: %tuple.type = tuple_literal (%int.convert_checked.loc4_94, %int.convert_checked.loc4_109, %int.convert_checked.loc4_124)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc4_125.2: ref %i32 = array_index file.%arr.var, %int_0
// CHECK:STDOUT:   %.loc4_125.3: init %i32 = initialize_from %int.convert_checked.loc4_94 to %.loc4_125.2 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int_1.loc4_125: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc4_125.4: ref %i32 = array_index file.%arr.var, %int_1.loc4_125
// CHECK:STDOUT:   %.loc4_125.5: init %i32 = initialize_from %int.convert_checked.loc4_109 to %.loc4_125.4 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int_2.loc4_125: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc4_125.6: ref %i32 = array_index file.%arr.var, %int_2.loc4_125
// CHECK:STDOUT:   %.loc4_125.7: init %i32 = initialize_from %int.convert_checked.loc4_124 to %.loc4_125.6 [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc4_125.8: init %array_type = array_init (%.loc4_125.3, %.loc4_125.5, %.loc4_125.7) to file.%arr.var [template = constants.%array]
// CHECK:STDOUT:   %.loc4_126: init %array_type = converted %.loc4_125.1, %.loc4_125.8 [template = constants.%array]
// CHECK:STDOUT:   assign file.%arr.var, %.loc4_126
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
