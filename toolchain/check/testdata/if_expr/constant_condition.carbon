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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.2, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.1, @ImplicitAs.impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.1, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %Constant.type: type = fn_type @Constant [concrete]
// CHECK:STDOUT:   %Constant: %Constant.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %ptr.235 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.ba2: type = fn_type @Op.2, @Destroy.impl(%ptr.235) [concrete]
// CHECK:STDOUT:   %Op.649: %Op.type.ba2 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5d5: type = ptr_type %ptr.235 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.829: <specific function> = specific_function %Op.649, @Op.2(%ptr.235) [concrete]
// CHECK:STDOUT:   %Op.type.a17: type = fn_type @Op.2, @Destroy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Op.e6a: %Op.type.a17 = struct_value () [concrete]
// CHECK:STDOUT:   %Op.specific_fn.014: <specific function> = specific_function %Op.e6a, @Op.2(%i32) [concrete]
// CHECK:STDOUT:   %PartiallyConstant.type: type = fn_type @PartiallyConstant [concrete]
// CHECK:STDOUT:   %PartiallyConstant: %PartiallyConstant.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @ImplicitAs.impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @ImplicitAs.impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @ImplicitAs.impl.4f9 [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .Constant = %Constant.decl
// CHECK:STDOUT:     .PartiallyConstant = %PartiallyConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
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
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Constant.decl: %Constant.type = fn_decl @Constant [concrete = constants.%Constant] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc26: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc26: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PartiallyConstant.decl: %PartiallyConstant.type = fn_decl @PartiallyConstant [concrete = constants.%PartiallyConstant] {
// CHECK:STDOUT:     %t.patt: %pattern_type.98f = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: %pattern_type.98f = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %t.param: type = value_param call_param0
// CHECK:STDOUT:     %t: type = bind_name t, %t.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc15_25.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_25.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc15_25.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc15_25.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc15_25.2: %i32 = converted %int_1, %.loc15_25.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   return %.loc15_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc16_25.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_25.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc16_25.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc16_25.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc16_25.2: %i32 = converted %int_2, %.loc16_25.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   return %.loc16_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %i32 = call %A.ref()
// CHECK:STDOUT:   %.loc19_25.1: %i32 = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc19_25.2: %i32 = converted %A.call, %.loc19_25.1
// CHECK:STDOUT:   br !if.expr.result(%.loc19_25.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref()
// CHECK:STDOUT:   %.loc19_27.1: %i32 = value_of_initializer %B.call
// CHECK:STDOUT:   %.loc19_27.2: %i32 = converted %B.call, %.loc19_27.1
// CHECK:STDOUT:   br !if.expr.result(%.loc19_27.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc19_10: %i32 = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc19_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %i32 = call %A.ref()
// CHECK:STDOUT:   %.loc23_26.1: %i32 = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc23_26.2: %i32 = converted %A.call, %.loc23_26.1
// CHECK:STDOUT:   br !if.expr.result(%.loc23_26.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref()
// CHECK:STDOUT:   %.loc23_28.1: %i32 = value_of_initializer %B.call
// CHECK:STDOUT:   %.loc23_28.2: %i32 = converted %B.call, %.loc23_28.1
// CHECK:STDOUT:   br !if.expr.result(%.loc23_28.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc23_10: %i32 = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc23_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Constant() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.7ce = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.7ce = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %i32 = var %v.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc27_3.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_3.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc27_3.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc27_3: init %i32 = converted %int_1, %int.convert_checked [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   assign %v.var, %.loc27_3
// CHECK:STDOUT:   br !.loc27_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc27_13:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.expr.then.loc27 else br !if.expr.else.loc27
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc27:
// CHECK:STDOUT:   %int_32.loc27_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc27_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result.loc27(%i32.loc27_23)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc27:
// CHECK:STDOUT:   %int_32.loc27_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc27_32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %ptr.loc27: type = ptr_type %i32.loc27_32 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   br !if.expr.result.loc27(%ptr.loc27)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc27:
// CHECK:STDOUT:   %.loc27_10: type = block_arg !if.expr.result.loc27 [concrete = constants.%i32]
// CHECK:STDOUT:   br !.loc27_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc27_7:
// CHECK:STDOUT:   %v: ref %i32 = bind_name v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type.fe8 = binding_pattern w [concrete]
// CHECK:STDOUT:     %w.var_patt: %pattern_type.fe8 = var_pattern %w.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.var: ref %ptr.235 = var %w.var_patt
// CHECK:STDOUT:   %v.ref: ref %i32 = name_ref v, %v
// CHECK:STDOUT:   %addr.loc28_40: %ptr.235 = addr_of %v.ref
// CHECK:STDOUT:   assign %w.var, %addr.loc28_40
// CHECK:STDOUT:   br !.loc28_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc28_13:
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false br !if.expr.then.loc28 else br !if.expr.else.loc28
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc28:
// CHECK:STDOUT:   %int_32.loc28_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc28_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result.loc28(%i32.loc28_24)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc28:
// CHECK:STDOUT:   %int_32.loc28_33: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc28_33: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %ptr.loc28: type = ptr_type %i32.loc28_33 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   br !if.expr.result.loc28(%ptr.loc28)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc28:
// CHECK:STDOUT:   %.loc28: type = block_arg !if.expr.result.loc28 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   br !.loc28_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc28_7:
// CHECK:STDOUT:   %w: ref %ptr.235 = bind_name w, %w.var
// CHECK:STDOUT:   %w.ref: ref %ptr.235 = name_ref w, %w
// CHECK:STDOUT:   %.loc29_11: %ptr.235 = bind_value %w.ref
// CHECK:STDOUT:   %.loc29_10.1: ref %i32 = deref %.loc29_11
// CHECK:STDOUT:   %.loc29_10.2: %i32 = bind_value %.loc29_10.1
// CHECK:STDOUT:   %Op.bound.loc28: <bound method> = bound_method %w.var, constants.%Op.649
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.649, @Op.2(constants.%ptr.235) [concrete = constants.%Op.specific_fn.829]
// CHECK:STDOUT:   %bound_method.loc28: <bound method> = bound_method %w.var, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc28_3: %ptr.5d5 = addr_of %w.var
// CHECK:STDOUT:   %no_op.loc28: init %empty_tuple.type = call %bound_method.loc28(%addr.loc28_3)
// CHECK:STDOUT:   %Op.bound.loc27: <bound method> = bound_method %v.var, constants.%Op.e6a
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.e6a, @Op.2(constants.%i32) [concrete = constants.%Op.specific_fn.014]
// CHECK:STDOUT:   %bound_method.loc27_3.3: <bound method> = bound_method %v.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc27: %ptr.235 = addr_of %v.var
// CHECK:STDOUT:   %no_op.loc27: init %empty_tuple.type = call %bound_method.loc27_3.3(%addr.loc27)
// CHECK:STDOUT:   return %.loc29_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartiallyConstant(%t.param: type) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.7ce = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.7ce = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %i32 = var %v.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc33_3.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc33_3.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc33_3.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc33_3: init %i32 = converted %int_1, %int.convert_checked [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   assign %v.var, %.loc33_3
// CHECK:STDOUT:   br !.loc33_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc33_13:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.expr.then.loc33 else br !if.expr.else.loc33
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc33:
// CHECK:STDOUT:   %int_32.loc33: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc33: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result.loc33(%i32.loc33)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc33:
// CHECK:STDOUT:   %t.ref.loc33: type = name_ref t, %t
// CHECK:STDOUT:   br !if.expr.result.loc33(%t.ref.loc33)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc33:
// CHECK:STDOUT:   %.loc33_10: type = block_arg !if.expr.result.loc33 [concrete = constants.%i32]
// CHECK:STDOUT:   br !.loc33_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc33_7:
// CHECK:STDOUT:   %v: ref %i32 = bind_name v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type.fe8 = binding_pattern w [concrete]
// CHECK:STDOUT:     %w.var_patt: %pattern_type.fe8 = var_pattern %w.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.var: ref %ptr.235 = var %w.var_patt
// CHECK:STDOUT:   %v.ref: ref %i32 = name_ref v, %v
// CHECK:STDOUT:   %addr.loc34_38: %ptr.235 = addr_of %v.ref
// CHECK:STDOUT:   assign %w.var, %addr.loc34_38
// CHECK:STDOUT:   br !.loc34_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc34_13:
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false br !if.expr.then.loc34 else br !if.expr.else.loc34
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc34:
// CHECK:STDOUT:   %t.ref.loc34: type = name_ref t, %t
// CHECK:STDOUT:   br !if.expr.result.loc34(%t.ref.loc34)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc34:
// CHECK:STDOUT:   %int_32.loc34: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc34: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %ptr: type = ptr_type %i32.loc34 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   br !if.expr.result.loc34(%ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc34:
// CHECK:STDOUT:   %.loc34: type = block_arg !if.expr.result.loc34 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   br !.loc34_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc34_7:
// CHECK:STDOUT:   %w: ref %ptr.235 = bind_name w, %w.var
// CHECK:STDOUT:   %w.ref: ref %ptr.235 = name_ref w, %w
// CHECK:STDOUT:   %.loc35_11: %ptr.235 = bind_value %w.ref
// CHECK:STDOUT:   %.loc35_10.1: ref %i32 = deref %.loc35_11
// CHECK:STDOUT:   %.loc35_10.2: %i32 = bind_value %.loc35_10.1
// CHECK:STDOUT:   %Op.bound.loc34: <bound method> = bound_method %w.var, constants.%Op.649
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.649, @Op.2(constants.%ptr.235) [concrete = constants.%Op.specific_fn.829]
// CHECK:STDOUT:   %bound_method.loc34: <bound method> = bound_method %w.var, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc34_3: %ptr.5d5 = addr_of %w.var
// CHECK:STDOUT:   %no_op.loc34: init %empty_tuple.type = call %bound_method.loc34(%addr.loc34_3)
// CHECK:STDOUT:   %Op.bound.loc33: <bound method> = bound_method %v.var, constants.%Op.e6a
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.e6a, @Op.2(constants.%i32) [concrete = constants.%Op.specific_fn.014]
// CHECK:STDOUT:   %bound_method.loc33_3.3: <bound method> = bound_method %v.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc33: %ptr.235 = addr_of %v.var
// CHECK:STDOUT:   %no_op.loc33: init %empty_tuple.type = call %bound_method.loc33_3.3(%addr.loc33)
// CHECK:STDOUT:   return %.loc35_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
