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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %false: bool = bool_literal false [template]
// CHECK:STDOUT:   %Constant.type: type = fn_type @Constant [template]
// CHECK:STDOUT:   %Constant: %Constant.type = struct_value () [template]
// CHECK:STDOUT:   %ptr: type = ptr_type %i32 [template]
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
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %int.make_type_signed, %.loc11_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_11.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_11.2: type = converted %int.make_type_signed, %.loc12_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_11.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_11.2: type = converted %int.make_type_signed, %.loc14_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_11.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_11.2: type = converted %int.make_type_signed, %.loc18_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Constant.decl: %Constant.type = fn_decl @Constant [template = constants.%Constant] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc22: init type = call constants.%Int(%int_32.loc22) [template = constants.%i32]
// CHECK:STDOUT:     %.loc22_18.1: type = value_of_initializer %int.make_type_signed.loc22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc22_18.2: type = converted %int.make_type_signed.loc22, %.loc22_18.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PartiallyConstant.decl: %PartiallyConstant.type = fn_decl @PartiallyConstant [template = constants.%PartiallyConstant] {
// CHECK:STDOUT:     %t.patt: type = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: type = value_param_pattern %t.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc28: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc28: init type = call constants.%Int(%int_32.loc28) [template = constants.%i32]
// CHECK:STDOUT:     %.loc28_34.1: type = value_of_initializer %int.make_type_signed.loc28 [template = constants.%i32]
// CHECK:STDOUT:     %.loc28_34.2: type = converted %int.make_type_signed.loc28, %.loc28_34.1 [template = constants.%i32]
// CHECK:STDOUT:     %t.param: type = value_param runtime_param0
// CHECK:STDOUT:     %t: type = bind_name t, %t.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1, %impl.elem0 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_25.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_25.2: %i32 = converted %int_1, %.loc11_25.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   return %.loc11_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_2, %impl.elem0 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc12_25.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc12_25.2: %i32 = converted %int_2, %.loc12_25.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   return %.loc12_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   if %true br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %i32 = call %A.ref()
// CHECK:STDOUT:   %.loc15_25.1: %i32 = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc15_25.2: %i32 = converted %A.call, %.loc15_25.1
// CHECK:STDOUT:   br !if.expr.result(%.loc15_25.2)
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
// CHECK:STDOUT:   %false: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   if %false br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %i32 = call %A.ref()
// CHECK:STDOUT:   %.loc19_26.1: %i32 = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc19_26.2: %i32 = converted %A.call, %.loc19_26.1
// CHECK:STDOUT:   br !if.expr.result(%.loc19_26.2)
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
// CHECK:STDOUT:   %true: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   if %true br !if.expr.then.loc23 else br !if.expr.else.loc23
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc23:
// CHECK:STDOUT:   %int_32.loc23_23: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc23_23: init type = call constants.%Int(%int_32.loc23_23) [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_23.1: type = value_of_initializer %int.make_type_signed.loc23_23 [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_23.2: type = converted %int.make_type_signed.loc23_23, %.loc23_23.1 [template = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result.loc23(%.loc23_23.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc23:
// CHECK:STDOUT:   %int_32.loc23_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc23_32: init type = call constants.%Int(%int_32.loc23_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_35.1: type = value_of_initializer %int.make_type_signed.loc23_32 [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_35.2: type = converted %int.make_type_signed.loc23_32, %.loc23_35.1 [template = constants.%i32]
// CHECK:STDOUT:   %ptr.loc23: type = ptr_type %i32 [template = constants.%ptr]
// CHECK:STDOUT:   br !if.expr.result.loc23(%ptr.loc23)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc23:
// CHECK:STDOUT:   %.loc23_10: type = block_arg !if.expr.result.loc23 [template = constants.%i32]
// CHECK:STDOUT:   %v.var: ref %i32 = var v
// CHECK:STDOUT:   %v: ref %i32 = bind_name v, %v.var
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1, %impl.elem0 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc23_40: init %i32 = converted %int_1, %int.convert_checked [template = constants.%int_1.2]
// CHECK:STDOUT:   assign %v.var, %.loc23_40
// CHECK:STDOUT:   %false: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   if %false br !if.expr.then.loc24 else br !if.expr.else.loc24
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc24:
// CHECK:STDOUT:   %int_32.loc24_24: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc24_24: init type = call constants.%Int(%int_32.loc24_24) [template = constants.%i32]
// CHECK:STDOUT:   %.loc24_24.1: type = value_of_initializer %int.make_type_signed.loc24_24 [template = constants.%i32]
// CHECK:STDOUT:   %.loc24_24.2: type = converted %int.make_type_signed.loc24_24, %.loc24_24.1 [template = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result.loc24(%.loc24_24.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc24:
// CHECK:STDOUT:   %int_32.loc24_33: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc24_33: init type = call constants.%Int(%int_32.loc24_33) [template = constants.%i32]
// CHECK:STDOUT:   %.loc24_36.1: type = value_of_initializer %int.make_type_signed.loc24_33 [template = constants.%i32]
// CHECK:STDOUT:   %.loc24_36.2: type = converted %int.make_type_signed.loc24_33, %.loc24_36.1 [template = constants.%i32]
// CHECK:STDOUT:   %ptr.loc24: type = ptr_type %i32 [template = constants.%ptr]
// CHECK:STDOUT:   br !if.expr.result.loc24(%ptr.loc24)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc24:
// CHECK:STDOUT:   %.loc24_10: type = block_arg !if.expr.result.loc24 [template = constants.%ptr]
// CHECK:STDOUT:   %w.var: ref %ptr = var w
// CHECK:STDOUT:   %w: ref %ptr = bind_name w, %w.var
// CHECK:STDOUT:   %v.ref: ref %i32 = name_ref v, %v
// CHECK:STDOUT:   %addr: %ptr = addr_of %v.ref
// CHECK:STDOUT:   assign %w.var, %addr
// CHECK:STDOUT:   %w.ref: ref %ptr = name_ref w, %w
// CHECK:STDOUT:   %.loc25_11: %ptr = bind_value %w.ref
// CHECK:STDOUT:   %.loc25_10.1: ref %i32 = deref %.loc25_11
// CHECK:STDOUT:   %.loc25_10.2: %i32 = bind_value %.loc25_10.1
// CHECK:STDOUT:   return %.loc25_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartiallyConstant(%t.param_patt: type) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   if %true br !if.expr.then.loc29 else br !if.expr.else.loc29
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc29:
// CHECK:STDOUT:   %int_32.loc29: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc29: init type = call constants.%Int(%int_32.loc29) [template = constants.%i32]
// CHECK:STDOUT:   %.loc29_23.1: type = value_of_initializer %int.make_type_signed.loc29 [template = constants.%i32]
// CHECK:STDOUT:   %.loc29_23.2: type = converted %int.make_type_signed.loc29, %.loc29_23.1 [template = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result.loc29(%.loc29_23.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc29:
// CHECK:STDOUT:   %t.ref.loc29: type = name_ref t, %t
// CHECK:STDOUT:   br !if.expr.result.loc29(%t.ref.loc29)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc29:
// CHECK:STDOUT:   %.loc29_10: type = block_arg !if.expr.result.loc29 [template = constants.%i32]
// CHECK:STDOUT:   %v.var: ref %i32 = var v
// CHECK:STDOUT:   %v: ref %i32 = bind_name v, %v.var
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1, %impl.elem0 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc29_37: init %i32 = converted %int_1, %int.convert_checked [template = constants.%int_1.2]
// CHECK:STDOUT:   assign %v.var, %.loc29_37
// CHECK:STDOUT:   %false: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   if %false br !if.expr.then.loc30 else br !if.expr.else.loc30
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc30:
// CHECK:STDOUT:   %t.ref.loc30: type = name_ref t, %t
// CHECK:STDOUT:   br !if.expr.result.loc30(%t.ref.loc30)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc30:
// CHECK:STDOUT:   %int_32.loc30: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc30: init type = call constants.%Int(%int_32.loc30) [template = constants.%i32]
// CHECK:STDOUT:   %.loc30_34.1: type = value_of_initializer %int.make_type_signed.loc30 [template = constants.%i32]
// CHECK:STDOUT:   %.loc30_34.2: type = converted %int.make_type_signed.loc30, %.loc30_34.1 [template = constants.%i32]
// CHECK:STDOUT:   %ptr: type = ptr_type %i32 [template = constants.%ptr]
// CHECK:STDOUT:   br !if.expr.result.loc30(%ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc30:
// CHECK:STDOUT:   %.loc30_10: type = block_arg !if.expr.result.loc30 [template = constants.%ptr]
// CHECK:STDOUT:   %w.var: ref %ptr = var w
// CHECK:STDOUT:   %w: ref %ptr = bind_name w, %w.var
// CHECK:STDOUT:   %v.ref: ref %i32 = name_ref v, %v
// CHECK:STDOUT:   %addr: %ptr = addr_of %v.ref
// CHECK:STDOUT:   assign %w.var, %addr
// CHECK:STDOUT:   %w.ref: ref %ptr = name_ref w, %w
// CHECK:STDOUT:   %.loc31_11: %ptr = bind_value %w.ref
// CHECK:STDOUT:   %.loc31_10.1: ref %i32 = deref %.loc31_11
// CHECK:STDOUT:   %.loc31_10.2: %i32 = bind_value %.loc31_10.1
// CHECK:STDOUT:   return %.loc31_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
