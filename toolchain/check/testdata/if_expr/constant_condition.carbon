// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/if_expr/constant_condition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/if_expr/constant_condition.carbon

fn A() -> i32 { return 1; }
fn B() -> i32 { return 2; }

fn F() -> i32 {
  return if true then A() else B();
}

fn G() -> i32 {
  return if false then A() else B();
}

fn Constant() -> i32 {
  var v: if true then i32 else i32* = 1;
  var w: if false then i32 else i32* = &v;
  return *w;
}

fn PartiallyConstant(t: type) -> i32 {
  var v: if true then i32 else t = 1;
  var w: if false then t else i32* = &v;
  return *w;
}

// CHECK:STDOUT: --- constant_condition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %.30: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.30, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.34: bool = bool_literal true [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.35: bool = bool_literal false [template]
// CHECK:STDOUT:   %Constant.type: type = fn_type @Constant [template]
// CHECK:STDOUT:   %Constant: %Constant.type = struct_value () [template]
// CHECK:STDOUT:   %.36: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %PartiallyConstant.type: type = fn_type @PartiallyConstant [template]
// CHECK:STDOUT:   %PartiallyConstant: %PartiallyConstant.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .Constant = %Constant.decl
// CHECK:STDOUT:     .PartiallyConstant = %PartiallyConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc11_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_11.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_11.3: type = converted %int.make_type_signed, %.loc11_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc12_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_11.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_11.3: type = converted %int.make_type_signed, %.loc12_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc14_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_11.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_11.3: type = converted %int.make_type_signed, %.loc14_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc18_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_11.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_11.3: type = converted %int.make_type_signed, %.loc18_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Constant.decl: %Constant.type = fn_decl @Constant [template = constants.%Constant] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc22_18.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc22: init type = call constants.%Int(%.loc22_18.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc22_18.2: type = value_of_initializer %int.make_type_signed.loc22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc22_18.3: type = converted %int.make_type_signed.loc22, %.loc22_18.2 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PartiallyConstant.decl: %PartiallyConstant.type = fn_decl @PartiallyConstant [template = constants.%PartiallyConstant] {
// CHECK:STDOUT:     %t.patt: type = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: type = value_param_pattern %t.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc28_34.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc28: init type = call constants.%Int(%.loc28_34.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc28_34.2: type = value_of_initializer %int.make_type_signed.loc28 [template = constants.%i32]
// CHECK:STDOUT:     %.loc28_34.3: type = converted %int.make_type_signed.loc28, %.loc28_34.2 [template = constants.%i32]
// CHECK:STDOUT:     %t.param: type = value_param runtime_param0
// CHECK:STDOUT:     %t: type = bind_name t, %t.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_24: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_25.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_25.2: <bound method> = bound_method %.loc11_24, %.loc11_25.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc11_25.3: <specific function> = specific_function %.loc11_25.2, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc11_25.3(%.loc11_24) [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_25.4: %i32 = value_of_initializer %int.convert_checked [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_25.5: %i32 = converted %.loc11_24, %.loc11_25.4 [template = constants.%.29]
// CHECK:STDOUT:   return %.loc11_25.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_24: Core.IntLiteral = int_value 2 [template = constants.%.30]
// CHECK:STDOUT:   %.loc12_25.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_25.2: <bound method> = bound_method %.loc12_24, %.loc12_25.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc12_25.3: <specific function> = specific_function %.loc12_25.2, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc12_25.3(%.loc12_24) [template = constants.%.33]
// CHECK:STDOUT:   %.loc12_25.4: %i32 = value_of_initializer %int.convert_checked [template = constants.%.33]
// CHECK:STDOUT:   %.loc12_25.5: %i32 = converted %.loc12_24, %.loc12_25.4 [template = constants.%.33]
// CHECK:STDOUT:   return %.loc12_25.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_13: bool = bool_literal true [template = constants.%.34]
// CHECK:STDOUT:   if %.loc15_13 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %i32 = call %A.ref()
// CHECK:STDOUT:   %.loc15_24.1: %i32 = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc15_24.2: %i32 = converted %A.call, %.loc15_24.1
// CHECK:STDOUT:   br !if.expr.result(%.loc15_24.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref()
// CHECK:STDOUT:   %.loc15_27.1: %i32 = value_of_initializer %B.call
// CHECK:STDOUT:   %.loc15_27.2: %i32 = converted %B.call, %.loc15_27.1
// CHECK:STDOUT:   br !if.expr.result(%.loc15_27.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc15_10: %i32 = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc15_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_13: bool = bool_literal false [template = constants.%.35]
// CHECK:STDOUT:   if %.loc19_13 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %i32 = call %A.ref()
// CHECK:STDOUT:   %.loc19_25.1: %i32 = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc19_25.2: %i32 = converted %A.call, %.loc19_25.1
// CHECK:STDOUT:   br !if.expr.result(%.loc19_25.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref()
// CHECK:STDOUT:   %.loc19_28.1: %i32 = value_of_initializer %B.call
// CHECK:STDOUT:   %.loc19_28.2: %i32 = converted %B.call, %.loc19_28.1
// CHECK:STDOUT:   br !if.expr.result(%.loc19_28.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc19_10: %i32 = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc19_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Constant() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc23_13: bool = bool_literal true [template = constants.%.34]
// CHECK:STDOUT:   if %.loc23_13 br !if.expr.then.loc23 else br !if.expr.else.loc23
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc23:
// CHECK:STDOUT:   %.loc23_23.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc23_23: init type = call constants.%Int(%.loc23_23.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_23.2: type = value_of_initializer %int.make_type_signed.loc23_23 [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_23.3: type = converted %int.make_type_signed.loc23_23, %.loc23_23.2 [template = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result.loc23(%.loc23_23.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc23:
// CHECK:STDOUT:   %.loc23_32: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc23_32: init type = call constants.%Int(%.loc23_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_35.1: type = value_of_initializer %int.make_type_signed.loc23_32 [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_35.2: type = converted %int.make_type_signed.loc23_32, %.loc23_35.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_35.3: type = ptr_type %i32 [template = constants.%.36]
// CHECK:STDOUT:   br !if.expr.result.loc23(%.loc23_35.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc23:
// CHECK:STDOUT:   %.loc23_10: type = block_arg !if.expr.result.loc23 [template = constants.%i32]
// CHECK:STDOUT:   %v.var: ref %i32 = var v
// CHECK:STDOUT:   %v: ref %i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc23_39: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc23_40.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc23_40.2: <bound method> = bound_method %.loc23_39, %.loc23_40.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc23_40.3: <specific function> = specific_function %.loc23_40.2, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc23_40.3(%.loc23_39) [template = constants.%.29]
// CHECK:STDOUT:   %.loc23_40.4: init %i32 = converted %.loc23_39, %int.convert_checked [template = constants.%.29]
// CHECK:STDOUT:   assign %v.var, %.loc23_40.4
// CHECK:STDOUT:   %.loc24_13: bool = bool_literal false [template = constants.%.35]
// CHECK:STDOUT:   if %.loc24_13 br !if.expr.then.loc24 else br !if.expr.else.loc24
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc24:
// CHECK:STDOUT:   %.loc24_24.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc24_24: init type = call constants.%Int(%.loc24_24.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc24_24.2: type = value_of_initializer %int.make_type_signed.loc24_24 [template = constants.%i32]
// CHECK:STDOUT:   %.loc24_24.3: type = converted %int.make_type_signed.loc24_24, %.loc24_24.2 [template = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result.loc24(%.loc24_24.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc24:
// CHECK:STDOUT:   %.loc24_33: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc24_33: init type = call constants.%Int(%.loc24_33) [template = constants.%i32]
// CHECK:STDOUT:   %.loc24_36.1: type = value_of_initializer %int.make_type_signed.loc24_33 [template = constants.%i32]
// CHECK:STDOUT:   %.loc24_36.2: type = converted %int.make_type_signed.loc24_33, %.loc24_36.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc24_36.3: type = ptr_type %i32 [template = constants.%.36]
// CHECK:STDOUT:   br !if.expr.result.loc24(%.loc24_36.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc24:
// CHECK:STDOUT:   %.loc24_10: type = block_arg !if.expr.result.loc24 [template = constants.%.36]
// CHECK:STDOUT:   %w.var: ref %.36 = var w
// CHECK:STDOUT:   %w: ref %.36 = bind_name w, %w.var
// CHECK:STDOUT:   %v.ref: ref %i32 = name_ref v, %v
// CHECK:STDOUT:   %.loc24_40: %.36 = addr_of %v.ref
// CHECK:STDOUT:   assign %w.var, %.loc24_40
// CHECK:STDOUT:   %w.ref: ref %.36 = name_ref w, %w
// CHECK:STDOUT:   %.loc25_11: %.36 = bind_value %w.ref
// CHECK:STDOUT:   %.loc25_10.1: ref %i32 = deref %.loc25_11
// CHECK:STDOUT:   %.loc25_10.2: %i32 = bind_value %.loc25_10.1
// CHECK:STDOUT:   return %.loc25_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartiallyConstant(%t.param_patt: type) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc29_13: bool = bool_literal true [template = constants.%.34]
// CHECK:STDOUT:   if %.loc29_13 br !if.expr.then.loc29 else br !if.expr.else.loc29
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc29:
// CHECK:STDOUT:   %.loc29_23.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc29: init type = call constants.%Int(%.loc29_23.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc29_23.2: type = value_of_initializer %int.make_type_signed.loc29 [template = constants.%i32]
// CHECK:STDOUT:   %.loc29_23.3: type = converted %int.make_type_signed.loc29, %.loc29_23.2 [template = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result.loc29(%.loc29_23.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc29:
// CHECK:STDOUT:   %t.ref.loc29: type = name_ref t, %t
// CHECK:STDOUT:   br !if.expr.result.loc29(%t.ref.loc29)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc29:
// CHECK:STDOUT:   %.loc29_10: type = block_arg !if.expr.result.loc29 [template = constants.%i32]
// CHECK:STDOUT:   %v.var: ref %i32 = var v
// CHECK:STDOUT:   %v: ref %i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc29_36: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc29_37.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc29_37.2: <bound method> = bound_method %.loc29_36, %.loc29_37.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc29_37.3: <specific function> = specific_function %.loc29_37.2, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc29_37.3(%.loc29_36) [template = constants.%.29]
// CHECK:STDOUT:   %.loc29_37.4: init %i32 = converted %.loc29_36, %int.convert_checked [template = constants.%.29]
// CHECK:STDOUT:   assign %v.var, %.loc29_37.4
// CHECK:STDOUT:   %.loc30_13: bool = bool_literal false [template = constants.%.35]
// CHECK:STDOUT:   if %.loc30_13 br !if.expr.then.loc30 else br !if.expr.else.loc30
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc30:
// CHECK:STDOUT:   %t.ref.loc30: type = name_ref t, %t
// CHECK:STDOUT:   br !if.expr.result.loc30(%t.ref.loc30)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc30:
// CHECK:STDOUT:   %.loc30_31: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc30: init type = call constants.%Int(%.loc30_31) [template = constants.%i32]
// CHECK:STDOUT:   %.loc30_34.1: type = value_of_initializer %int.make_type_signed.loc30 [template = constants.%i32]
// CHECK:STDOUT:   %.loc30_34.2: type = converted %int.make_type_signed.loc30, %.loc30_34.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc30_34.3: type = ptr_type %i32 [template = constants.%.36]
// CHECK:STDOUT:   br !if.expr.result.loc30(%.loc30_34.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc30:
// CHECK:STDOUT:   %.loc30_10: type = block_arg !if.expr.result.loc30 [template = constants.%.36]
// CHECK:STDOUT:   %w.var: ref %.36 = var w
// CHECK:STDOUT:   %w: ref %.36 = bind_name w, %w.var
// CHECK:STDOUT:   %v.ref: ref %i32 = name_ref v, %v
// CHECK:STDOUT:   %.loc30_38: %.36 = addr_of %v.ref
// CHECK:STDOUT:   assign %w.var, %.loc30_38
// CHECK:STDOUT:   %w.ref: ref %.36 = name_ref w, %w
// CHECK:STDOUT:   %.loc31_11: %.36 = bind_value %w.ref
// CHECK:STDOUT:   %.loc31_10.1: ref %i32 = deref %.loc31_11
// CHECK:STDOUT:   %.loc31_10.2: %i32 = bind_value %.loc31_10.1
// CHECK:STDOUT:   return %.loc31_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
