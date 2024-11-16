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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.3: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @As(i32) [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.2, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.2, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.28: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.4, %Convert.15 [template]
// CHECK:STDOUT:   %.30: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.30, %F.2 [template]
// CHECK:STDOUT:   %.32: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.16: type = fn_type @Convert.12 [template]
// CHECK:STDOUT:   %Convert.16: %Convert.type.16 = struct_value () [template]
// CHECK:STDOUT:   %.34: <witness> = interface_witness (%Convert.16) [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %.32, %Convert.16 [template]
// CHECK:STDOUT:   %.36: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.37: i32 = int_value 3 [template]
// CHECK:STDOUT:   %Convert.type.17: type = fn_type @Convert.13 [template]
// CHECK:STDOUT:   %Convert.17: %Convert.type.17 = struct_value () [template]
// CHECK:STDOUT:   %.38: <witness> = interface_witness (%Convert.17) [template]
// CHECK:STDOUT:   %.39: <bound method> = bound_method %.37, %Convert.17 [template]
// CHECK:STDOUT:   %.40: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.41: type = array_type %.40, i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .As = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.51
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
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_6.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc15_6.2: type = converted %int.make_type_32, %.loc15_6.1 [template = i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc19_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc19_17: Core.IntLiteral = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %int.make_type_32.loc19_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc19_22.1: type = value_of_initializer %int.make_type_32.loc19_22 [template = i32]
// CHECK:STDOUT:   %.loc19_22.2: type = converted %int.make_type_32.loc19_22, %.loc19_22.1 [template = i32]
// CHECK:STDOUT:   %.loc19_19.1: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc19_19.2: <bound method> = bound_method %.loc19_17, %.loc19_19.1 [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc19_19: init i32 = call %.loc19_19.2(%.loc19_17) [template = constants.%.30]
// CHECK:STDOUT:   %.loc19_19.3: i32 = value_of_initializer %int.convert_checked.loc19_19 [template = constants.%.30]
// CHECK:STDOUT:   %.loc19_19.4: i32 = converted %.loc19_17, %.loc19_19.3 [template = constants.%.30]
// CHECK:STDOUT:   %I.ref: type = name_ref I, %I.decl [template = constants.%I.type]
// CHECK:STDOUT:   %F.ref: %.1 = name_ref F, @I.%.loc12 [template = constants.%.2]
// CHECK:STDOUT:   %.loc19_26.1: %F.type.1 = interface_witness_access constants.%.3, element0 [template = constants.%F.2]
// CHECK:STDOUT:   %.loc19_26.2: <bound method> = bound_method %.loc19_19.4, %.loc19_26.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc19_33.1: Core.IntLiteral = int_value 2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc19_33.2: %Convert.type.4 = interface_witness_access constants.%.34, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %.loc19_33.3: <bound method> = bound_method %.loc19_33.1, %.loc19_33.2 [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc19_33: init i32 = call %.loc19_33.3(%.loc19_33.1) [template = constants.%.36]
// CHECK:STDOUT:   %.loc19_33.4: i32 = value_of_initializer %int.convert_checked.loc19_33 [template = constants.%.36]
// CHECK:STDOUT:   %.loc19_33.5: i32 = converted %.loc19_33.1, %.loc19_33.4 [template = constants.%.36]
// CHECK:STDOUT:   %int.sadd: init i32 = call %.loc19_26.2(%.loc19_19.4, %.loc19_33.5) [template = constants.%.37]
// CHECK:STDOUT:   %.loc19_11.1: type = value_of_initializer %int.make_type_32.loc19_11 [template = i32]
// CHECK:STDOUT:   %.loc19_11.2: type = converted %int.make_type_32.loc19_11, %.loc19_11.1 [template = i32]
// CHECK:STDOUT:   %.loc19_32.1: %Convert.type.7 = interface_witness_access constants.%.38, element0 [template = constants.%Convert.17]
// CHECK:STDOUT:   %.loc19_32.2: <bound method> = bound_method %int.sadd, %.loc19_32.1 [template = constants.%.39]
// CHECK:STDOUT:   %.loc19_32.3: i32 = value_of_initializer %int.sadd [template = constants.%.37]
// CHECK:STDOUT:   %.loc19_32.4: i32 = converted %int.sadd, %.loc19_32.3 [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc19_32: init Core.IntLiteral = call %.loc19_32.2(%.loc19_32.4) [template = constants.%.40]
// CHECK:STDOUT:   %.loc19_32.5: Core.IntLiteral = value_of_initializer %int.convert_checked.loc19_32 [template = constants.%.40]
// CHECK:STDOUT:   %.loc19_32.6: Core.IntLiteral = converted %int.sadd, %.loc19_32.5 [template = constants.%.40]
// CHECK:STDOUT:   %.loc19_35: type = array_type %.loc19_32.6, i32 [template = constants.%.41]
// CHECK:STDOUT:   %arr.var: ref %.41 = var arr
// CHECK:STDOUT:   %arr: ref %.41 = bind_name arr, %arr.var
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
// CHECK:STDOUT: impl @impl.1: %.loc15_6.2 as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %self.patt: i32 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: i32 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: i32 = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: i32 = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc16_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_14.1: type = value_of_initializer %int.make_type_32.loc16_14 [template = i32]
// CHECK:STDOUT:     %.loc16_14.2: type = converted %int.make_type_32.loc16_14, %.loc16_14.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc16_26: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_26.1: type = value_of_initializer %int.make_type_32.loc16_26 [template = i32]
// CHECK:STDOUT:     %.loc16_26.2: type = converted %int.make_type_32.loc16_26, %.loc16_26.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc16_34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_34.1: type = value_of_initializer %int.make_type_32.loc16_34 [template = i32]
// CHECK:STDOUT:     %.loc16_34.2: type = converted %int.make_type_32.loc16_34, %.loc16_34.1 [template = i32]
// CHECK:STDOUT:     %self.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %self: i32 = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %other: i32 = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_15: <witness> = interface_witness (%F.decl) [template = constants.%.3]
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
// CHECK:STDOUT: fn @F.2[%self.param_patt: i32](%other.param_patt: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.1) {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(i32) {
// CHECK:STDOUT:   %Self => i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
