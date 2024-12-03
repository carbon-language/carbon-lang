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
// CHECK:STDOUT:   %Self.as_type.1: type = facet_access_type %Self.1 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %I.type, %F.type.1 [template]
// CHECK:STDOUT:   %assoc0.1: %F.assoc_type = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %i32, %i32 [template]
// CHECK:STDOUT:   %interface.1: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @As(%i32) [template]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.2, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.7, @impl.6(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.10: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.7(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %int_1.2, %F.2 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.2, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.16: type = fn_type @Convert.3, @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.16: %Convert.type.16 = struct_value () [template]
// CHECK:STDOUT:   %interface.11: <witness> = interface_witness (%Convert.16) [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.16 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.3(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %int_3.1: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %Convert.type.17: type = fn_type @Convert.5, @impl.4(%int_32) [template]
// CHECK:STDOUT:   %Convert.17: %Convert.type.17 = struct_value () [template]
// CHECK:STDOUT:   %interface.12: <witness> = interface_witness (%Convert.17) [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_3.1, %Convert.17 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.5(%int_32) [template]
// CHECK:STDOUT:   %int_3.2: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.2, %i32 [template]
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
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_6.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_6.2: type = converted %int.make_type_signed, %.loc15_6.1 [template = constants.%i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc19_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc19_11: init type = call constants.%Int(%int_32.loc19_11) [template = constants.%i32]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_32.loc19_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc19_22: init type = call constants.%Int(%int_32.loc19_22) [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_22.1: type = value_of_initializer %int.make_type_signed.loc19_22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_22.2: type = converted %int.make_type_signed.loc19_22, %.loc19_22.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc19_19: %Convert.type.2 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc19_19: <bound method> = bound_method %int_1, %impl.elem0.loc19_19 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc19_19: <specific function> = specific_function %Convert.bound.loc19_19, @Convert.7(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc19_19: init %i32 = call %Convert.specific_fn.loc19_19(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc19_19.1: %i32 = value_of_initializer %int.convert_checked.loc19_19 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc19_19.2: %i32 = converted %int_1, %.loc19_19.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %I.ref: type = name_ref I, %I.decl [template = constants.%I.type]
// CHECK:STDOUT:   %F.ref: %F.assoc_type = name_ref F, @I.%assoc0 [template = constants.%assoc0.1]
// CHECK:STDOUT:   %impl.elem0.loc19_26: %F.type.1 = interface_witness_access constants.%interface.1, element0 [template = constants.%F.2]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %.loc19_19.2, %impl.elem0.loc19_26 [template = constants.%F.bound]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc19_33: %Convert.type.15 = interface_witness_access constants.%interface.11, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %Convert.bound.loc19_33: <bound method> = bound_method %int_2, %impl.elem0.loc19_33 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc19_33: <specific function> = specific_function %Convert.bound.loc19_33, @Convert.3(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc19_33: init %i32 = call %Convert.specific_fn.loc19_33(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc19_33.1: %i32 = value_of_initializer %int.convert_checked.loc19_33 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc19_33.2: %i32 = converted %int_2, %.loc19_33.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int.sadd: init %i32 = call %F.bound(%.loc19_19.2, %.loc19_33.2) [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc19_11.1: type = value_of_initializer %int.make_type_signed.loc19_11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_11.2: type = converted %int.make_type_signed.loc19_11, %.loc19_11.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc19_34: %Convert.type.7 = interface_witness_access constants.%interface.12, element0 [template = constants.%Convert.17]
// CHECK:STDOUT:   %Convert.bound.loc19_34: <bound method> = bound_method %int.sadd, %impl.elem0.loc19_34 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc19_34: <specific function> = specific_function %Convert.bound.loc19_34, @Convert.5(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %.loc19_34.1: %i32 = value_of_initializer %int.sadd [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc19_34.2: %i32 = converted %int.sadd, %.loc19_34.1 [template = constants.%int_3.1]
// CHECK:STDOUT:   %int.convert_checked.loc19_34: init Core.IntLiteral = call %Convert.specific_fn.loc19_34(%.loc19_34.2) [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc19_34.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc19_34 [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc19_34.4: Core.IntLiteral = converted %int.sadd, %.loc19_34.3 [template = constants.%int_3.2]
// CHECK:STDOUT:   %array_type: type = array_type %.loc19_34.4, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %arr.var: ref %array_type = var arr
// CHECK:STDOUT:   %arr: ref %array_type = bind_name arr, %arr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc12_14: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %Self.as_type.loc12_14.2: type = facet_access_type %Self.ref.loc12_14 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:     %.loc12_14: type = converted %Self.ref.loc12_14, %Self.as_type.loc12_14.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:     %Self.ref.loc12_27: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %Self.as_type.loc12_27: type = facet_access_type %Self.ref.loc12_27 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:     %.loc12_27: type = converted %Self.ref.loc12_27, %Self.as_type.loc12_27 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:     %Self.ref.loc12_36: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %Self.as_type.loc12_36: type = facet_access_type %Self.ref.loc12_36 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:     %.loc12_36: type = converted %Self.ref.loc12_36, %Self.as_type.loc12_36 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = value_param runtime_param0
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = value_param runtime_param1
// CHECK:STDOUT:     %other: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = out_param runtime_param2
// CHECK:STDOUT:     %return: ref @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, %F.decl [template = constants.%assoc0.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %.loc15_6.2 as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %self.patt: %i32 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %i32 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: %i32 = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %i32 = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc16_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc16_14: init type = call constants.%Int(%int_32.loc16_14) [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_14.1: type = value_of_initializer %int.make_type_signed.loc16_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_14.2: type = converted %int.make_type_signed.loc16_14, %.loc16_14.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc16_26: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc16_26: init type = call constants.%Int(%int_32.loc16_26) [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_26.1: type = value_of_initializer %int.make_type_signed.loc16_26 [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_26.2: type = converted %int.make_type_signed.loc16_26, %.loc16_26.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc16_34: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc16_34: init type = call constants.%Int(%int_32.loc16_34) [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_34.1: type = value_of_initializer %int.make_type_signed.loc16_34 [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_34.2: type = converted %int.make_type_signed.loc16_34, %.loc16_34.1 [template = constants.%i32]
// CHECK:STDOUT:     %self.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %self: %i32 = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %other: %i32 = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%F.decl) [template = constants.%interface.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:   %Self.as_type.loc12_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1)](%other.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1)) -> @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %i32](%other.param_patt: %i32) -> %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.1) {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%Self.as_type.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
