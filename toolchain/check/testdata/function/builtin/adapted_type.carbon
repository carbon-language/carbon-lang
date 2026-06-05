// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/adapted_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/adapted_type.carbon

// --- adapt.carbon

library "[[@TEST_NAME]]";

fn MakeIntLiteral() -> type = "int_literal.make_type";
alias IntLiteral = MakeIntLiteral();

class MyIntLiteral {
  adapt IntLiteral;
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
// CHECK:STDOUT:   %.805: Core.Form = init_form type [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %MakeIntLiteral.type: type = fn_type @MakeIntLiteral [concrete]
// CHECK:STDOUT:   %MakeIntLiteral: %MakeIntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %MyIntLiteral: type = class_type @MyIntLiteral [concrete]
// CHECK:STDOUT:   %complete_type.972: <witness> = complete_type_witness Core.IntLiteral [concrete]
// CHECK:STDOUT:   %pattern_type.129: type = pattern_type %MyIntLiteral [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %MyInt32: type = class_type @MyInt32 [concrete]
// CHECK:STDOUT:   %int_32.be0: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %int_32.80c: %MyIntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32.80c [concrete]
// CHECK:STDOUT:   %.6a2: Core.Form = init_form %MyInt32 [concrete]
// CHECK:STDOUT:   %pattern_type.af4: type = pattern_type %MyInt32 [concrete]
// CHECK:STDOUT:   %MyInt32.Make.type: type = fn_type @MyInt32.Make [concrete]
// CHECK:STDOUT:   %MyInt32.Make: %MyInt32.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.113: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %MyAdd.type: type = fn_type @MyAdd [concrete]
// CHECK:STDOUT:   %MyAdd: %MyAdd.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_1.004: %MyIntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_1.adf: %MyInt32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_2.6b8: %MyIntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_2.596: %MyInt32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3: %MyInt32 = int_value 3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .MakeIntLiteral = %MakeIntLiteral.decl
// CHECK:STDOUT:     .IntLiteral = %IntLiteral
// CHECK:STDOUT:     .MyIntLiteral = %MyIntLiteral.decl
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .MyInt32 = %MyInt32.decl
// CHECK:STDOUT:     .MyAdd = %MyAdd.decl
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeIntLiteral.decl: %MakeIntLiteral.type = fn_decl @MakeIntLiteral [concrete = constants.%MakeIntLiteral] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern %return.param_patt, %.loc4_24.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_24.1: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %.loc4_24.2: Core.Form = init_form %.loc4_24.1 [concrete = constants.%.805]
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeIntLiteral.ref: %MakeIntLiteral.type = name_ref MakeIntLiteral, %MakeIntLiteral.decl [concrete = constants.%MakeIntLiteral]
// CHECK:STDOUT:   %MakeIntLiteral.call: init type = call %MakeIntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %.loc5_35.1: type = value_of_initializer %MakeIntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %.loc5_35.2: type = converted %MakeIntLiteral.call, %.loc5_35.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %IntLiteral: type = alias_binding IntLiteral, %.loc5_35.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %MyIntLiteral.decl: type = class_decl @MyIntLiteral [concrete = constants.%MyIntLiteral] {} {}
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [concrete = constants.%Int] {
// CHECK:STDOUT:     %N.param_patt: %pattern_type.129 = value_param_pattern [concrete]
// CHECK:STDOUT:     %N.patt: %pattern_type.129 = at_binding_pattern N, %N.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern %return.param_patt, %.loc11_28.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_28.1: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %.loc11_28.2: Core.Form = init_form %.loc11_28.1 [concrete = constants.%.805]
// CHECK:STDOUT:     %N.param: %MyIntLiteral = value_param call_param0
// CHECK:STDOUT:     %MyIntLiteral.ref: type = name_ref MyIntLiteral, file.%MyIntLiteral.decl [concrete = constants.%MyIntLiteral]
// CHECK:STDOUT:     %N: %MyIntLiteral = value_binding N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyInt32.decl: type = class_decl @MyInt32 [concrete = constants.%MyInt32] {} {}
// CHECK:STDOUT:   %MyInt32.Make.decl: %MyInt32.Make.type = fn_decl @MyInt32.Make [concrete = constants.%MyInt32.Make] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.129 = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.129 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.af4 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.af4 = return_slot_pattern %return.param_patt, %MyInt32.ref.loc19 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %MyInt32.ref.loc19: type = name_ref MyInt32, file.%MyInt32.decl [concrete = constants.%MyInt32]
// CHECK:STDOUT:     %.loc19: Core.Form = init_form %MyInt32.ref.loc19 [concrete = constants.%.6a2]
// CHECK:STDOUT:     %a.param.loc19: %MyIntLiteral = value_param call_param0
// CHECK:STDOUT:     %MyIntLiteral.ref.loc19: type = name_ref MyIntLiteral, file.%MyIntLiteral.decl [concrete = constants.%MyIntLiteral]
// CHECK:STDOUT:     %a.loc19: %MyIntLiteral = value_binding a, %a.param.loc19
// CHECK:STDOUT:     %return.param.loc19: ref %MyInt32 = out_param call_param1
// CHECK:STDOUT:     %return.loc19: ref %MyInt32 = return_slot %return.param.loc19
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyAdd.decl: %MyAdd.type = fn_decl @MyAdd [concrete = constants.%MyAdd] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.af4 = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.af4 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.af4 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.af4 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.af4 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.af4 = return_slot_pattern %return.param_patt, %MyInt32.ref.loc21_37 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %MyInt32.ref.loc21_37: type = name_ref MyInt32, file.%MyInt32.decl [concrete = constants.%MyInt32]
// CHECK:STDOUT:     %.loc21: Core.Form = init_form %MyInt32.ref.loc21_37 [concrete = constants.%.6a2]
// CHECK:STDOUT:     %a.param: %MyInt32 = value_param call_param0
// CHECK:STDOUT:     %MyInt32.ref.loc21_13: type = name_ref MyInt32, file.%MyInt32.decl [concrete = constants.%MyInt32]
// CHECK:STDOUT:     %a: %MyInt32 = value_binding a, %a.param
// CHECK:STDOUT:     %b.param: %MyInt32 = value_param call_param1
// CHECK:STDOUT:     %MyInt32.ref.loc21_25: type = name_ref MyInt32, file.%MyInt32.decl [concrete = constants.%MyInt32]
// CHECK:STDOUT:     %b: %MyInt32 = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref %MyInt32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %MyInt32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.af4 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.af4 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %MyInt32 = var %v.var_patt [concrete]
// CHECK:STDOUT:   %MyInt32.ref: type = name_ref MyInt32, %MyInt32.decl [concrete = constants.%MyInt32]
// CHECK:STDOUT:   %v: ref %MyInt32 = ref_binding v, %v.var [concrete = %v.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @MyIntLiteral {
// CHECK:STDOUT:   %IntLiteral.ref: type = name_ref IntLiteral, file.%IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:   adapt_decl %IntLiteral.ref [concrete]
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
// CHECK:STDOUT:   %.loc14_16.1: %MyIntLiteral = as_compatible %int_32 [concrete = constants.%int_32.80c]
// CHECK:STDOUT:   %.loc14_16.2: %MyIntLiteral = converted %int_32, %.loc14_16.1 [concrete = constants.%int_32.80c]
// CHECK:STDOUT:   %Int.call: init type = call %Int.ref(%.loc14_16.2) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc14_32.1: type = value_of_initializer %Int.call [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc14_32.2: type = converted %Int.call, %.loc14_32.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   adapt_decl %.loc14_32.2 [concrete]
// CHECK:STDOUT:   %MyInt32.Make.decl: %MyInt32.Make.type = fn_decl @MyInt32.Make [concrete = constants.%MyInt32.Make] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.129 = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.129 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.af4 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.af4 = return_slot_pattern %return.param_patt, %MyInt32.ref.loc19 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %MyInt32.ref.loc16: type = name_ref MyInt32, file.%MyInt32.decl [concrete = constants.%MyInt32]
// CHECK:STDOUT:     %.loc16: Core.Form = init_form %MyInt32.ref.loc16 [concrete = constants.%.6a2]
// CHECK:STDOUT:     %a.param.loc16: %MyIntLiteral = value_param call_param0
// CHECK:STDOUT:     %MyIntLiteral.ref.loc16: type = name_ref MyIntLiteral, file.%MyIntLiteral.decl [concrete = constants.%MyIntLiteral]
// CHECK:STDOUT:     %a.loc16: %MyIntLiteral = value_binding a, %a.param.loc16
// CHECK:STDOUT:     %return.param.loc16: ref %MyInt32 = out_param call_param1
// CHECK:STDOUT:     %return.loc16: ref %MyInt32 = return_slot %return.param.loc16
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%i32.builtin [concrete = constants.%complete_type.113]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%MyInt32
// CHECK:STDOUT:   .Int = <poisoned>
// CHECK:STDOUT:   .MyIntLiteral = <poisoned>
// CHECK:STDOUT:   .MyInt32 = <poisoned>
// CHECK:STDOUT:   .Make = %MyInt32.Make.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeIntLiteral() -> out %return.param: type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param: %MyIntLiteral) -> out %return.param: type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyInt32.Make(%a.param.loc19: %MyIntLiteral) -> out %return.param.loc19: %MyInt32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyAdd(%a.param: %MyInt32, %b.param: %MyInt32) -> out %return.param: %MyInt32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MyAdd.ref: %MyAdd.type = name_ref MyAdd, file.%MyAdd.decl [concrete = constants.%MyAdd]
// CHECK:STDOUT:   %MyInt32.ref.loc23_24: type = name_ref MyInt32, file.%MyInt32.decl [concrete = constants.%MyInt32]
// CHECK:STDOUT:   %Make.ref.loc23_31: %MyInt32.Make.type = name_ref Make, @MyInt32.%MyInt32.Make.decl [concrete = constants.%MyInt32.Make]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %MyIntLiteral.ref.loc23_42: type = name_ref MyIntLiteral, file.%MyIntLiteral.decl [concrete = constants.%MyIntLiteral]
// CHECK:STDOUT:   %.loc23_39.1: %MyIntLiteral = as_compatible %int_1 [concrete = constants.%int_1.004]
// CHECK:STDOUT:   %.loc23_39.2: %MyIntLiteral = converted %int_1, %.loc23_39.1 [concrete = constants.%int_1.004]
// CHECK:STDOUT:   %MyInt32.Make.call.loc23_54: init %MyInt32 = call %Make.ref.loc23_31(%.loc23_39.2) [concrete = constants.%int_1.adf]
// CHECK:STDOUT:   %MyInt32.ref.loc23_57: type = name_ref MyInt32, file.%MyInt32.decl [concrete = constants.%MyInt32]
// CHECK:STDOUT:   %Make.ref.loc23_64: %MyInt32.Make.type = name_ref Make, @MyInt32.%MyInt32.Make.decl [concrete = constants.%MyInt32.Make]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %MyIntLiteral.ref.loc23_75: type = name_ref MyIntLiteral, file.%MyIntLiteral.decl [concrete = constants.%MyIntLiteral]
// CHECK:STDOUT:   %.loc23_72.1: %MyIntLiteral = as_compatible %int_2 [concrete = constants.%int_2.6b8]
// CHECK:STDOUT:   %.loc23_72.2: %MyIntLiteral = converted %int_2, %.loc23_72.1 [concrete = constants.%int_2.6b8]
// CHECK:STDOUT:   %MyInt32.Make.call.loc23_87: init %MyInt32 = call %Make.ref.loc23_64(%.loc23_72.2) [concrete = constants.%int_2.596]
// CHECK:STDOUT:   %.loc23_54.1: %MyInt32 = value_of_initializer %MyInt32.Make.call.loc23_54 [concrete = constants.%int_1.adf]
// CHECK:STDOUT:   %.loc23_54.2: %MyInt32 = converted %MyInt32.Make.call.loc23_54, %.loc23_54.1 [concrete = constants.%int_1.adf]
// CHECK:STDOUT:   %.loc23_87.1: %MyInt32 = value_of_initializer %MyInt32.Make.call.loc23_87 [concrete = constants.%int_2.596]
// CHECK:STDOUT:   %.loc23_87.2: %MyInt32 = converted %MyInt32.Make.call.loc23_87, %.loc23_87.1 [concrete = constants.%int_2.596]
// CHECK:STDOUT:   %MyAdd.call: init %MyInt32 = call %MyAdd.ref(%.loc23_54.2, %.loc23_87.2) [concrete = constants.%int_3]
// CHECK:STDOUT:   assign file.%v.var, %MyAdd.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MyIntLiteral: type = class_type @MyIntLiteral [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.129: type = pattern_type %MyIntLiteral [concrete]
// CHECK:STDOUT:   %.805: Core.Form = init_form type [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
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
// CHECK:STDOUT:     %N.param_patt: %pattern_type.129 = value_param_pattern [concrete]
// CHECK:STDOUT:     %N.patt: %pattern_type.129 = at_binding_pattern N, %N.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern %return.param_patt, %.loc12_28.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_28.1: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %.loc12_28.2: Core.Form = init_form %.loc12_28.1 [concrete = constants.%.805]
// CHECK:STDOUT:     %N.param: %MyIntLiteral = value_param call_param0
// CHECK:STDOUT:     %MyIntLiteral.ref: type = name_ref MyIntLiteral, file.%MyIntLiteral.decl [concrete = constants.%MyIntLiteral]
// CHECK:STDOUT:     %N: %MyIntLiteral = value_binding N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @MyIntLiteral {
// CHECK:STDOUT:   %.loc5_10: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_11: type = converted %.loc5_10, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   adapt_decl %.loc5_11 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%MyIntLiteral
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param: %MyIntLiteral) -> out %return.param: type;
// CHECK:STDOUT:
