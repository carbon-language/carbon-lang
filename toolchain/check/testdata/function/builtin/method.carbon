// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/method.carbon

interface I {
  fn F[self: Self](other: Self) -> Self;
}

impl i32 as I {
  fn F[self: i32](other: i32) -> i32 = "int.sadd";
}

var arr: [i32; (1 as i32).(I.F)(2)];

// CHECK:STDOUT: --- method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.1: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %I.type, %F.type.1 [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = int_type signed, %.3 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @As(%.4) [template]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.2, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.7, @impl.6(%.3) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.28: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.6, %Convert.14 [template]
// CHECK:STDOUT:   %.30: <specific function> = specific_function %.29, @Convert.7(%.3) [template]
// CHECK:STDOUT:   %.31: %.4 = int_value 1 [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.31, %F.2 [template]
// CHECK:STDOUT:   %.33: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.2, @ImplicitAs(%.4) [template]
// CHECK:STDOUT:   %Convert.type.16: type = fn_type @Convert.3, @impl.2(%.3) [template]
// CHECK:STDOUT:   %Convert.16: %Convert.type.16 = struct_value () [template]
// CHECK:STDOUT:   %.37: <witness> = interface_witness (%Convert.16) [template]
// CHECK:STDOUT:   %.38: <bound method> = bound_method %.33, %Convert.16 [template]
// CHECK:STDOUT:   %.39: <specific function> = specific_function %.38, @Convert.3(%.3) [template]
// CHECK:STDOUT:   %.40: %.4 = int_value 2 [template]
// CHECK:STDOUT:   %.41: %.4 = int_value 3 [template]
// CHECK:STDOUT:   %Convert.type.17: type = fn_type @Convert.5, @impl.4(%.3) [template]
// CHECK:STDOUT:   %Convert.17: %Convert.type.17 = struct_value () [template]
// CHECK:STDOUT:   %.42: <witness> = interface_witness (%Convert.17) [template]
// CHECK:STDOUT:   %.43: <bound method> = bound_method %.41, %Convert.17 [template]
// CHECK:STDOUT:   %.44: <specific function> = specific_function %.43, @Convert.5(%.3) [template]
// CHECK:STDOUT:   %.45: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.46: type = array_type %.45, %.4 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .As = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.38
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %.loc15_6.1: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc15_6.1) [template = constants.%.4]
// CHECK:STDOUT:     %.loc15_6.2: type = value_of_initializer %int.make_type_signed [template = constants.%.4]
// CHECK:STDOUT:     %.loc15_6.3: type = converted %int.make_type_signed, %.loc15_6.2 [template = constants.%.4]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_11.1: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_signed.loc19_11: init type = call constants.%Int(%.loc19_11.1) [template = constants.%.4]
// CHECK:STDOUT:   %.loc19_17: Core.IntLiteral = int_value 1 [template = constants.%.6]
// CHECK:STDOUT:   %.loc19_22.1: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_signed.loc19_22: init type = call constants.%Int(%.loc19_22.1) [template = constants.%.4]
// CHECK:STDOUT:   %.loc19_22.2: type = value_of_initializer %int.make_type_signed.loc19_22 [template = constants.%.4]
// CHECK:STDOUT:   %.loc19_22.3: type = converted %int.make_type_signed.loc19_22, %.loc19_22.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc19_19.1: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc19_19.2: <bound method> = bound_method %.loc19_17, %.loc19_19.1 [template = constants.%.29]
// CHECK:STDOUT:   %.loc19_19.3: <specific function> = specific_function %.loc19_19.2, @Convert.7(constants.%.3) [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc19_19: init %.4 = call %.loc19_19.3(%.loc19_17) [template = constants.%.31]
// CHECK:STDOUT:   %.loc19_19.4: %.4 = value_of_initializer %int.convert_checked.loc19_19 [template = constants.%.31]
// CHECK:STDOUT:   %.loc19_19.5: %.4 = converted %.loc19_17, %.loc19_19.4 [template = constants.%.31]
// CHECK:STDOUT:   %I.ref: type = name_ref I, %I.decl [template = constants.%I.type]
// CHECK:STDOUT:   %F.ref: %.1 = name_ref F, @I.%.loc12 [template = constants.%.2]
// CHECK:STDOUT:   %.loc19_26.1: %F.type.1 = interface_witness_access constants.%.5, element0 [template = constants.%F.2]
// CHECK:STDOUT:   %.loc19_26.2: <bound method> = bound_method %.loc19_19.5, %.loc19_26.1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc19_33.1: Core.IntLiteral = int_value 2 [template = constants.%.33]
// CHECK:STDOUT:   %.loc19_33.2: %Convert.type.15 = interface_witness_access constants.%.37, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %.loc19_33.3: <bound method> = bound_method %.loc19_33.1, %.loc19_33.2 [template = constants.%.38]
// CHECK:STDOUT:   %.loc19_33.4: <specific function> = specific_function %.loc19_33.3, @Convert.3(constants.%.3) [template = constants.%.39]
// CHECK:STDOUT:   %int.convert_checked.loc19_33: init %.4 = call %.loc19_33.4(%.loc19_33.1) [template = constants.%.40]
// CHECK:STDOUT:   %.loc19_33.5: %.4 = value_of_initializer %int.convert_checked.loc19_33 [template = constants.%.40]
// CHECK:STDOUT:   %.loc19_33.6: %.4 = converted %.loc19_33.1, %.loc19_33.5 [template = constants.%.40]
// CHECK:STDOUT:   %int.sadd: init %.4 = call %.loc19_26.2(%.loc19_19.5, %.loc19_33.6) [template = constants.%.41]
// CHECK:STDOUT:   %.loc19_11.2: type = value_of_initializer %int.make_type_signed.loc19_11 [template = constants.%.4]
// CHECK:STDOUT:   %.loc19_11.3: type = converted %int.make_type_signed.loc19_11, %.loc19_11.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc19_32.1: %Convert.type.7 = interface_witness_access constants.%.42, element0 [template = constants.%Convert.17]
// CHECK:STDOUT:   %.loc19_32.2: <bound method> = bound_method %int.sadd, %.loc19_32.1 [template = constants.%.43]
// CHECK:STDOUT:   %.loc19_32.3: <specific function> = specific_function %.loc19_32.2, @Convert.5(constants.%.3) [template = constants.%.44]
// CHECK:STDOUT:   %.loc19_32.4: %.4 = value_of_initializer %int.sadd [template = constants.%.41]
// CHECK:STDOUT:   %.loc19_32.5: %.4 = converted %int.sadd, %.loc19_32.4 [template = constants.%.41]
// CHECK:STDOUT:   %int.convert_checked.loc19_32: init Core.IntLiteral = call %.loc19_32.3(%.loc19_32.5) [template = constants.%.45]
// CHECK:STDOUT:   %.loc19_32.6: Core.IntLiteral = value_of_initializer %int.convert_checked.loc19_32 [template = constants.%.45]
// CHECK:STDOUT:   %.loc19_32.7: Core.IntLiteral = converted %int.sadd, %.loc19_32.6 [template = constants.%.45]
// CHECK:STDOUT:   %.loc19_35: type = array_type %.loc19_32.7, %.4 [template = constants.%.46]
// CHECK:STDOUT:   %arr.var: ref %.46 = var arr
// CHECK:STDOUT:   %arr: ref %.46 = bind_name arr, %arr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self (%Self.1) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self (%Self.1) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: @F.1.%Self (%Self.1) = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: @F.1.%Self (%Self.1) = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: @F.1.%Self (%Self.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%Self (%Self.1) = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc12_14: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_14.1: type = facet_type_access %Self.ref.loc12_14 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_14.2: type = converted %Self.ref.loc12_14, %.loc12_14.1 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %Self.ref.loc12_27: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_27.1: type = facet_type_access %Self.ref.loc12_27 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_27.2: type = converted %Self.ref.loc12_27, %.loc12_27.1 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %Self.ref.loc12_36: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_36.1: type = facet_type_access %Self.ref.loc12_36 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc12_36.2: type = converted %Self.ref.loc12_36, %.loc12_36.1 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %self.param: @F.1.%Self (%Self.1) = value_param runtime_param0
// CHECK:STDOUT:     %self: @F.1.%Self (%Self.1) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @F.1.%Self (%Self.1) = value_param runtime_param1
// CHECK:STDOUT:     %other: @F.1.%Self (%Self.1) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @F.1.%Self (%Self.1) = out_param runtime_param2
// CHECK:STDOUT:     %return: ref @F.1.%Self (%Self.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: %.1 = assoc_entity element0, %F.decl [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc12
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %.loc15_6.3 as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %self.patt: %.4 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.4 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: %.4 = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %.4 = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %.4 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.4 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_14.1: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed.loc16_14: init type = call constants.%Int(%.loc16_14.1) [template = constants.%.4]
// CHECK:STDOUT:     %.loc16_14.2: type = value_of_initializer %int.make_type_signed.loc16_14 [template = constants.%.4]
// CHECK:STDOUT:     %.loc16_14.3: type = converted %int.make_type_signed.loc16_14, %.loc16_14.2 [template = constants.%.4]
// CHECK:STDOUT:     %.loc16_26.1: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed.loc16_26: init type = call constants.%Int(%.loc16_26.1) [template = constants.%.4]
// CHECK:STDOUT:     %.loc16_26.2: type = value_of_initializer %int.make_type_signed.loc16_26 [template = constants.%.4]
// CHECK:STDOUT:     %.loc16_26.3: type = converted %int.make_type_signed.loc16_26, %.loc16_26.2 [template = constants.%.4]
// CHECK:STDOUT:     %.loc16_34.1: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed.loc16_34: init type = call constants.%Int(%.loc16_34.1) [template = constants.%.4]
// CHECK:STDOUT:     %.loc16_34.2: type = value_of_initializer %int.make_type_signed.loc16_34 [template = constants.%.4]
// CHECK:STDOUT:     %.loc16_34.3: type = converted %int.make_type_signed.loc16_34, %.loc16_34.2 [template = constants.%.4]
// CHECK:STDOUT:     %self.param: %.4 = value_param runtime_param0
// CHECK:STDOUT:     %self: %.4 = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %.4 = value_param runtime_param1
// CHECK:STDOUT:     %other: %.4 = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref %.4 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %.4 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_15: <witness> = interface_witness (%F.decl) [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc15_15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self (%Self.1)](%other.param_patt: @F.1.%Self (%Self.1)) -> @F.1.%Self (%Self.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %.4](%other.param_patt: %.4) -> %.4 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.1) {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%.4) {
// CHECK:STDOUT:   %Self => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
