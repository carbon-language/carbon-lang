// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/no_prelude/adapted_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/no_prelude/adapted_type.carbon

// --- adapt.carbon

library "[[@TEST_NAME]]";

fn IntLiteral() -> type = "int_literal.make_type";

class MyIntLiteral {
  adapt IntLiteral();
}

fn Int(N: MyIntLiteral) -> type = "int.make_type_signed";

class MyInt32 {
  adapt Int(32 as MyIntLiteral);

  fn Make(a: MyIntLiteral) -> MyInt32;
}

fn MyInt32.Make(a: MyIntLiteral) -> MyInt32 = "int.convert_checked";

fn MyAdd(a: MyInt32, b: MyInt32) -> MyInt32 = "int.sadd";

var v: MyInt32 = MyAdd(MyInt32.Make(1 as MyIntLiteral), MyInt32.Make(2 as MyIntLiteral));

// --- fail_bad_adapt.carbon

library "[[@TEST_NAME]]";

class MyIntLiteral {
  adapt {};
}

// CHECK:STDERR: fail_bad_adapt.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.make_type_signed" [InvalidBuiltinSignature]
// CHECK:STDERR: fn Int(N: MyIntLiteral) -> type = "int.make_type_signed";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Int(N: MyIntLiteral) -> type = "int.make_type_signed";

// CHECK:STDOUT: --- adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %MyIntLiteral: type = class_type @MyIntLiteral [concrete]
// CHECK:STDOUT:   %complete_type.972: <witness> = complete_type_witness Core.IntLiteral [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %MyInt32: type = class_type @MyInt32 [concrete]
// CHECK:STDOUT:   %int_32.be0: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %int_32.4da: %MyIntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32.4da [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.ec2: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %MyAdd.type: type = fn_type @MyAdd [concrete]
// CHECK:STDOUT:   %MyAdd: %MyAdd.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_1.383: %MyIntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_1.d74: %MyInt32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_2.a27: %MyIntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_2.eda: %MyInt32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3: %MyInt32 = int_value 3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .MyIntLiteral = %MyIntLiteral.decl
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .MyInt32 = %MyInt32.decl
// CHECK:STDOUT:     .MyAdd = %MyAdd.decl
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [concrete = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyIntLiteral.decl: type = class_decl @MyIntLiteral [concrete = constants.%MyIntLiteral] {} {}
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [concrete = constants.%Int] {
// CHECK:STDOUT:     %N.patt: %MyIntLiteral = binding_pattern N
// CHECK:STDOUT:     %N.param_patt: %MyIntLiteral = value_param_pattern %N.patt, call_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: %MyIntLiteral = value_param call_param0
// CHECK:STDOUT:     %MyIntLiteral.ref: type = name_ref MyIntLiteral, file.%MyIntLiteral.decl [concrete = constants.%MyIntLiteral]
// CHECK:STDOUT:     %N: %MyIntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyInt32.decl: type = class_decl @MyInt32 [concrete = constants.%MyInt32] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %a.patt: %MyIntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %MyIntLiteral = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: %MyInt32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %MyInt32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %MyInt32.ref.loc18: type = name_ref MyInt32, file.%MyInt32.decl [concrete = constants.%MyInt32]
// CHECK:STDOUT:     %a.param.loc18: %MyIntLiteral = value_param call_param0
// CHECK:STDOUT:     %MyIntLiteral.ref.loc18: type = name_ref MyIntLiteral, file.%MyIntLiteral.decl [concrete = constants.%MyIntLiteral]
// CHECK:STDOUT:     %a.loc18: %MyIntLiteral = bind_name a, %a.param.loc18
// CHECK:STDOUT:     %return.param.loc18: ref %MyInt32 = out_param call_param1
// CHECK:STDOUT:     %return.loc18: ref %MyInt32 = return_slot %return.param.loc18
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyAdd.decl: %MyAdd.type = fn_decl @MyAdd [concrete = constants.%MyAdd] {
// CHECK:STDOUT:     %a.patt: %MyInt32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %MyInt32 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: %MyInt32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %MyInt32 = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: %MyInt32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %MyInt32 = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %MyInt32.ref.loc20_37: type = name_ref MyInt32, file.%MyInt32.decl [concrete = constants.%MyInt32]
// CHECK:STDOUT:     %a.param: %MyInt32 = value_param call_param0
// CHECK:STDOUT:     %MyInt32.ref.loc20_13: type = name_ref MyInt32, file.%MyInt32.decl [concrete = constants.%MyInt32]
// CHECK:STDOUT:     %a: %MyInt32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %MyInt32 = value_param call_param1
// CHECK:STDOUT:     %MyInt32.ref.loc20_25: type = name_ref MyInt32, file.%MyInt32.decl [concrete = constants.%MyInt32]
// CHECK:STDOUT:     %b: %MyInt32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %MyInt32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %MyInt32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %MyInt32 = binding_pattern v
// CHECK:STDOUT:     %.loc22: %MyInt32 = var_pattern %v.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %MyInt32 = var v
// CHECK:STDOUT:   %MyInt32.ref: type = name_ref MyInt32, %MyInt32.decl [concrete = constants.%MyInt32]
// CHECK:STDOUT:   %v: ref %MyInt32 = bind_name v, %v.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @MyIntLiteral {
// CHECK:STDOUT:   %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %.loc7_21.1: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %.loc7_21.2: type = converted %int_literal.make_type, %.loc7_21.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:   adapt_decl %.loc7_21.2 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness Core.IntLiteral [concrete = constants.%complete_type.972]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%MyIntLiteral
// CHECK:STDOUT:   .IntLiteral = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @MyInt32 {
// CHECK:STDOUT:   %Int.ref: %Int.type = name_ref Int, file.%Int.decl [concrete = constants.%Int]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32.be0]
// CHECK:STDOUT:   %MyIntLiteral.ref: type = name_ref MyIntLiteral, file.%MyIntLiteral.decl [concrete = constants.%MyIntLiteral]
// CHECK:STDOUT:   %.loc13_16.1: %MyIntLiteral = as_compatible %int_32 [concrete = constants.%int_32.4da]
// CHECK:STDOUT:   %.loc13_16.2: %MyIntLiteral = converted %int_32, %.loc13_16.1 [concrete = constants.%int_32.4da]
// CHECK:STDOUT:   %int.make_type_signed: init type = call %Int.ref(%.loc13_16.2) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc13_32.1: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc13_32.2: type = converted %int.make_type_signed, %.loc13_32.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   adapt_decl %.loc13_32.2 [concrete]
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %a.patt: %MyIntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %MyIntLiteral = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: %MyInt32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %MyInt32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %MyInt32.ref.loc15: type = name_ref MyInt32, file.%MyInt32.decl [concrete = constants.%MyInt32]
// CHECK:STDOUT:     %a.param.loc15: %MyIntLiteral = value_param call_param0
// CHECK:STDOUT:     %MyIntLiteral.ref.loc15: type = name_ref MyIntLiteral, file.%MyIntLiteral.decl [concrete = constants.%MyIntLiteral]
// CHECK:STDOUT:     %a.loc15: %MyIntLiteral = bind_name a, %a.param.loc15
// CHECK:STDOUT:     %return.param.loc15: ref %MyInt32 = out_param call_param1
// CHECK:STDOUT:     %return.loc15: ref %MyInt32 = return_slot %return.param.loc15
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %i32.builtin [concrete = constants.%complete_type.ec2]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%MyInt32
// CHECK:STDOUT:   .Int = <poisoned>
// CHECK:STDOUT:   .MyIntLiteral = <poisoned>
// CHECK:STDOUT:   .MyInt32 = <poisoned>
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: %MyIntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make(%a.param_patt: %MyIntLiteral) -> %MyInt32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyAdd(%a.param_patt: %MyInt32, %b.param_patt: %MyInt32) -> %MyInt32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MyAdd.ref: %MyAdd.type = name_ref MyAdd, file.%MyAdd.decl [concrete = constants.%MyAdd]
// CHECK:STDOUT:   %MyInt32.ref.loc22_24: type = name_ref MyInt32, file.%MyInt32.decl [concrete = constants.%MyInt32]
// CHECK:STDOUT:   %Make.ref.loc22_31: %Make.type = name_ref Make, @MyInt32.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %MyIntLiteral.ref.loc22_42: type = name_ref MyIntLiteral, file.%MyIntLiteral.decl [concrete = constants.%MyIntLiteral]
// CHECK:STDOUT:   %.loc22_39.1: %MyIntLiteral = as_compatible %int_1 [concrete = constants.%int_1.383]
// CHECK:STDOUT:   %.loc22_39.2: %MyIntLiteral = converted %int_1, %.loc22_39.1 [concrete = constants.%int_1.383]
// CHECK:STDOUT:   %int.convert_checked.loc22_54: init %MyInt32 = call %Make.ref.loc22_31(%.loc22_39.2) [concrete = constants.%int_1.d74]
// CHECK:STDOUT:   %MyInt32.ref.loc22_57: type = name_ref MyInt32, file.%MyInt32.decl [concrete = constants.%MyInt32]
// CHECK:STDOUT:   %Make.ref.loc22_64: %Make.type = name_ref Make, @MyInt32.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %MyIntLiteral.ref.loc22_75: type = name_ref MyIntLiteral, file.%MyIntLiteral.decl [concrete = constants.%MyIntLiteral]
// CHECK:STDOUT:   %.loc22_72.1: %MyIntLiteral = as_compatible %int_2 [concrete = constants.%int_2.a27]
// CHECK:STDOUT:   %.loc22_72.2: %MyIntLiteral = converted %int_2, %.loc22_72.1 [concrete = constants.%int_2.a27]
// CHECK:STDOUT:   %int.convert_checked.loc22_87: init %MyInt32 = call %Make.ref.loc22_64(%.loc22_72.2) [concrete = constants.%int_2.eda]
// CHECK:STDOUT:   %.loc22_54.1: %MyInt32 = value_of_initializer %int.convert_checked.loc22_54 [concrete = constants.%int_1.d74]
// CHECK:STDOUT:   %.loc22_54.2: %MyInt32 = converted %int.convert_checked.loc22_54, %.loc22_54.1 [concrete = constants.%int_1.d74]
// CHECK:STDOUT:   %.loc22_87.1: %MyInt32 = value_of_initializer %int.convert_checked.loc22_87 [concrete = constants.%int_2.eda]
// CHECK:STDOUT:   %.loc22_87.2: %MyInt32 = converted %int.convert_checked.loc22_87, %.loc22_87.1 [concrete = constants.%int_2.eda]
// CHECK:STDOUT:   %int.sadd: init %MyInt32 = call %MyAdd.ref(%.loc22_54.2, %.loc22_87.2) [concrete = constants.%int_3]
// CHECK:STDOUT:   assign file.%v.var, %int.sadd
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MyIntLiteral: type = class_type @MyIntLiteral [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .MyIntLiteral = %MyIntLiteral.decl
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyIntLiteral.decl: type = class_decl @MyIntLiteral [concrete = constants.%MyIntLiteral] {} {}
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [concrete = constants.%Int] {
// CHECK:STDOUT:     %N.patt: %MyIntLiteral = binding_pattern N
// CHECK:STDOUT:     %N.param_patt: %MyIntLiteral = value_param_pattern %N.patt, call_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: %MyIntLiteral = value_param call_param0
// CHECK:STDOUT:     %MyIntLiteral.ref: type = name_ref MyIntLiteral, file.%MyIntLiteral.decl [concrete = constants.%MyIntLiteral]
// CHECK:STDOUT:     %N: %MyIntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @MyIntLiteral {
// CHECK:STDOUT:   %.loc5_10: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc5_11: type = converted %.loc5_10, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   adapt_decl %.loc5_11 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%MyIntLiteral
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: %MyIntLiteral) -> type;
// CHECK:STDOUT:
