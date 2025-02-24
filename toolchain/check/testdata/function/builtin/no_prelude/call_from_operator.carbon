// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/no_prelude/call_from_operator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/no_prelude/call_from_operator.carbon

// --- core.carbon

package Core;

fn IntLiteral() -> type = "int_literal.make_type";
fn Int(N: IntLiteral()) -> type = "int.make_type_signed";

interface Add {
  fn Op[self: Self](other: Self) -> Self;
}

interface As(T:! type) {
  fn Convert[self: Self]() -> T;
}

interface ImplicitAs(T:! type) {
  fn Convert[self: Self]() -> T;
}

impl i32 as Add {
  fn Op[self: Self](other: Self) -> Self = "int.sadd";
}

impl IntLiteral() as As(i32) {
  fn Convert[self: Self]() -> i32 = "int.convert_checked";
}

impl IntLiteral() as ImplicitAs(i32) {
  fn Convert[self: Self]() -> i32 = "int.convert_checked";
}

impl i32 as ImplicitAs(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// --- user.carbon

import Core;

var arr: array(i32, (1 as i32) + (2 as i32)) = (3, 4, (3 as i32) + (4 as i32));

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [concrete]
// CHECK:STDOUT:   %Self.b3d: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.f73: type = facet_access_type %Self.b3d [symbolic]
// CHECK:STDOUT:   %Op.type.31b: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.d59: %Op.type.31b = struct_value () [concrete]
// CHECK:STDOUT:   %Add.assoc_type: type = assoc_entity_type %Add.type [concrete]
// CHECK:STDOUT:   %assoc0.245: %Add.assoc_type = assoc_entity element0, @Add.%Op.decl [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %As.type.b51: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.b51 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.8ba: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %Self.b4e: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.7f0: type = facet_access_type %Self.b4e [symbolic]
// CHECK:STDOUT:   %Convert.type.ad1: type = fn_type @Convert.1, @As(%T) [symbolic]
// CHECK:STDOUT:   %Convert.0ed: %Convert.type.ad1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type.600: type = assoc_entity_type %As.type.8ba [symbolic]
// CHECK:STDOUT:   %assoc0.ac5: %As.assoc_type.600 = assoc_entity element0, @As.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.0f3: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.419: type = facet_access_type %Self.0f3 [symbolic]
// CHECK:STDOUT:   %Convert.type.4cf: type = fn_type @Convert.2, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.147: %Convert.type.4cf = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.94e: type = assoc_entity_type %ImplicitAs.type.07f [symbolic]
// CHECK:STDOUT:   %assoc0.a50: %ImplicitAs.assoc_type.94e = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %impl_witness.8b6: <witness> = impl_witness (@impl.1.%Op.decl) [concrete]
// CHECK:STDOUT:   %Op.type.c2a: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.4e3: %Op.type.c2a = struct_value () [concrete]
// CHECK:STDOUT:   %Add.facet: %Add.type = facet_value %i32.builtin, %impl_witness.8b6 [concrete]
// CHECK:STDOUT:   %As.type.a09: type = facet_type <@As, @As(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Convert.type.c0d: type = fn_type @Convert.1, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %Convert.713: %Convert.type.c0d = struct_value () [concrete]
// CHECK:STDOUT:   %As.assoc_type.567: type = assoc_entity_type %As.type.a09 [concrete]
// CHECK:STDOUT:   %assoc0.3bd: %As.assoc_type.567 = assoc_entity element0, @As.%Convert.decl [concrete]
// CHECK:STDOUT:   %impl_witness.213: <witness> = impl_witness (@impl.2.%Convert.decl) [concrete]
// CHECK:STDOUT:   %Convert.type.fc9: type = fn_type @Convert.3 [concrete]
// CHECK:STDOUT:   %Convert.33c: %Convert.type.fc9 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.a09 = facet_value Core.IntLiteral, %impl_witness.213 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.11a: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Convert.type.752: type = fn_type @Convert.2, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %Convert.fcc: %Convert.type.752 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.dd3: type = assoc_entity_type %ImplicitAs.type.11a [concrete]
// CHECK:STDOUT:   %assoc0.7cc: %ImplicitAs.assoc_type.dd3 = assoc_entity element0, @ImplicitAs.%Convert.decl [concrete]
// CHECK:STDOUT:   %impl_witness.48c: <witness> = impl_witness (@impl.3.%Convert.decl) [concrete]
// CHECK:STDOUT:   %Convert.type.c2a: type = fn_type @Convert.4 [concrete]
// CHECK:STDOUT:   %Convert.40d: %Convert.type.c2a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.1ff: %ImplicitAs.type.11a = facet_value Core.IntLiteral, %impl_witness.48c [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.9fc: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Convert.type.60e: type = fn_type @Convert.2, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %Convert.c73: %Convert.type.60e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.81c: type = assoc_entity_type %ImplicitAs.type.9fc [concrete]
// CHECK:STDOUT:   %assoc0.80e: %ImplicitAs.assoc_type.81c = assoc_entity element0, @ImplicitAs.%Convert.decl [concrete]
// CHECK:STDOUT:   %impl_witness.caf: <witness> = impl_witness (@impl.4.%Convert.decl) [concrete]
// CHECK:STDOUT:   %Convert.type.295: type = fn_type @Convert.5 [concrete]
// CHECK:STDOUT:   %Convert.2bf: %Convert.type.295 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.72d: %ImplicitAs.type.9fc = facet_value %i32.builtin, %impl_witness.caf [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [concrete = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [concrete = constants.%Int] {
// CHECK:STDOUT:     %N.patt: Core.IntLiteral = binding_pattern N
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc5_22.1: type = splice_block %.loc5_22.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_22.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_22.3: type = converted %int_literal.make_type, %.loc5_22.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N: Core.IntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Add.decl: type = interface_decl @Add [concrete = constants.%Add.type] {} {}
// CHECK:STDOUT:   %As.decl: %As.type.b51 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %T.patt.loc11_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_14.1, runtime_param<none> [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc11_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %T.patt.loc15_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_22.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc15_22.1, runtime_param<none> [symbolic = %T.patt.loc15_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc15_22.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc15_22.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.1 [concrete] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc19_6.1: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc19_6.2: type = converted %int.make_type_signed, %.loc19_6.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, file.%Add.decl [concrete = constants.%Add.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc19: <witness> = impl_witness (@impl.1.%Op.decl) [concrete = constants.%impl_witness.8b6]
// CHECK:STDOUT:   impl_decl @impl.2 [concrete] {} {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc23_17.1: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc23_17.2: type = converted %int_literal.make_type, %.loc23_17.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %As.ref: %As.type.b51 = name_ref As, file.%As.decl [concrete = constants.%As.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc23_28.1: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc23_28.2: type = converted %int.make_type_signed, %.loc23_28.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %As.type: type = facet_type <@As, @As(constants.%i32.builtin)> [concrete = constants.%As.type.a09]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc23: <witness> = impl_witness (@impl.2.%Convert.decl) [concrete = constants.%impl_witness.213]
// CHECK:STDOUT:   impl_decl @impl.3 [concrete] {} {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc27_17.1: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc27_17.2: type = converted %int_literal.make_type, %.loc27_17.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.96f = name_ref ImplicitAs, file.%ImplicitAs.decl [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc27_36.1: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc27_36.2: type = converted %int.make_type_signed, %.loc27_36.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32.builtin)> [concrete = constants.%ImplicitAs.type.11a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc27: <witness> = impl_witness (@impl.3.%Convert.decl) [concrete = constants.%impl_witness.48c]
// CHECK:STDOUT:   impl_decl @impl.4 [concrete] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc31_6.1: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc31_6.2: type = converted %int.make_type_signed, %.loc31_6.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.96f = name_ref ImplicitAs, file.%ImplicitAs.decl [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc31_36.1: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc31_36.2: type = converted %int_literal.make_type, %.loc31_36.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete = constants.%ImplicitAs.type.9fc]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc31: <witness> = impl_witness (@impl.4.%Convert.decl) [concrete = constants.%impl_witness.caf]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.b3d]
// CHECK:STDOUT:   %Op.decl: %Op.type.31b = fn_decl @Op.1 [concrete = constants.%Op.d59] {
// CHECK:STDOUT:     %self.patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.f73) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.f73) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.f73) = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.f73) = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.f73) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.f73) = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc8_37: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self.b3d)]
// CHECK:STDOUT:     %Self.as_type.loc8_37: type = facet_access_type %Self.ref.loc8_37 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.f73)]
// CHECK:STDOUT:     %.loc8_37: type = converted %Self.ref.loc8_37, %Self.as_type.loc8_37 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.f73)]
// CHECK:STDOUT:     %self.param: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.f73) = value_param runtime_param0
// CHECK:STDOUT:     %.loc8_15.1: type = splice_block %.loc8_15.2 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.f73)] {
// CHECK:STDOUT:       %Self.ref.loc8_15: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self.b3d)]
// CHECK:STDOUT:       %Self.as_type.loc8_15.2: type = facet_access_type %Self.ref.loc8_15 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.f73)]
// CHECK:STDOUT:       %.loc8_15.2: type = converted %Self.ref.loc8_15, %Self.as_type.loc8_15.2 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.f73)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.f73) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.f73) = value_param runtime_param1
// CHECK:STDOUT:     %.loc8_28.1: type = splice_block %.loc8_28.2 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.f73)] {
// CHECK:STDOUT:       %Self.ref.loc8_28: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self.b3d)]
// CHECK:STDOUT:       %Self.as_type.loc8_28: type = facet_access_type %Self.ref.loc8_28 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.f73)]
// CHECK:STDOUT:       %.loc8_28.2: type = converted %Self.ref.loc8_28, %Self.as_type.loc8_28 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.f73)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.f73) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.f73) = out_param runtime_param2
// CHECK:STDOUT:     %return: ref @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.f73) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Add.assoc_type = assoc_entity element0, %Op.decl [concrete = constants.%assoc0.245]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%T.loc11_14.1: type) {
// CHECK:STDOUT:   %T.loc11_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T.loc11_14.2)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self.2: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%T.loc11_14.2) [symbolic = %Convert.type (constants.%Convert.type.ad1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.ad1) = struct_value () [symbolic = %Convert (constants.%Convert.0ed)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As.%As.type (%As.type.8ba) [symbolic = %As.assoc_type (constants.%As.assoc_type.600)]
// CHECK:STDOUT:   %assoc0.loc12_32.2: @As.%As.assoc_type (%As.assoc_type.600) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc12_32.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:     %Convert.decl: @As.%Convert.type (%Convert.type.ad1) = fn_decl @Convert.1 [symbolic = @As.%Convert (constants.%Convert.0ed)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type.7f0) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type.7f0) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @As.%T.loc11_14.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type.7f0) = value_param runtime_param0
// CHECK:STDOUT:       %.loc12_20.1: type = splice_block %.loc12_20.3 [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.7f0)] {
// CHECK:STDOUT:         %.loc12_20.2: @Convert.1.%As.type (%As.type.8ba) = specific_constant @As.%Self.1, @As(constants.%T) [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%As.type (%As.type.8ba) = name_ref Self, %.loc12_20.2 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.as_type.loc12_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:         %.loc12_20.3: type = converted %Self.ref, %Self.as_type.loc12_20.2 [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type.7f0) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc12_32.1: @As.%As.assoc_type (%As.assoc_type.600) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc12_32.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc12_32.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc15_22.1: type) {
// CHECK:STDOUT:   %T.loc15_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_22.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc15_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_22.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc15_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%T.loc15_22.2) [symbolic = %Convert.type (constants.%Convert.type.4cf)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.4cf) = struct_value () [symbolic = %Convert (constants.%Convert.147)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.94e)]
// CHECK:STDOUT:   %assoc0.loc16_32.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.94e) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc16_32.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.4cf) = fn_decl @Convert.2 [symbolic = @ImplicitAs.%Convert (constants.%Convert.147)] {
// CHECK:STDOUT:       %self.patt: @Convert.2.%Self.as_type.loc16_20.1 (%Self.as_type.419) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.2.%Self.as_type.loc16_20.1 (%Self.as_type.419) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Convert.2.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.2.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc15_22.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.2.%Self.as_type.loc16_20.1 (%Self.as_type.419) = value_param runtime_param0
// CHECK:STDOUT:       %.loc16_20.1: type = splice_block %.loc16_20.3 [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.419)] {
// CHECK:STDOUT:         %.loc16_20.2: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.ref: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc16_20.2 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.as_type.loc16_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:         %.loc16_20.3: type = converted %Self.ref, %Self.as_type.loc16_20.2 [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.2.%Self.as_type.loc16_20.1 (%Self.as_type.419) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.2.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Convert.2.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc16_32.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.94e) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc16_32.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc16_32.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %.loc19_6.2 as %Add.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.c2a = fn_decl @Op.2 [concrete = constants.%Op.4e3] {
// CHECK:STDOUT:     %self.patt: %i32.builtin = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %i32.builtin = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: %i32.builtin = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %i32.builtin = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32.builtin = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32.builtin = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc20_37: type = name_ref Self, @impl.1.%.loc19_6.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self.param: %i32.builtin = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref.loc20_15: type = name_ref Self, @impl.1.%.loc19_6.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self: %i32.builtin = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %i32.builtin = value_param runtime_param1
// CHECK:STDOUT:     %Self.ref.loc20_28: type = name_ref Self, @impl.1.%.loc19_6.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %other: %i32.builtin = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %.loc23_17.2 as %As.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.fc9 = fn_decl @Convert.3 [concrete = constants.%Convert.33c] {
// CHECK:STDOUT:     %self.patt: Core.IntLiteral = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: Core.IntLiteral = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32.builtin = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32.builtin = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc24_31.1: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc24_31.2: type = converted %int.make_type_signed, %.loc24_31.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.2.%.loc23_17.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %self: Core.IntLiteral = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %.loc27_17.2 as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.c2a = fn_decl @Convert.4 [concrete = constants.%Convert.40d] {
// CHECK:STDOUT:     %self.patt: Core.IntLiteral = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: Core.IntLiteral = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32.builtin = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32.builtin = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc28_31.1: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc28_31.2: type = converted %int.make_type_signed, %.loc28_31.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.3.%.loc27_17.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %self: Core.IntLiteral = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: %.loc31_6.2 as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.295 = fn_decl @Convert.5 [concrete = constants.%Convert.2bf] {
// CHECK:STDOUT:     %self.patt: %i32.builtin = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %i32.builtin = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc32_42.1: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc32_42.2: type = converted %int_literal.make_type, %.loc32_42.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %self.param: %i32.builtin = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.4.%.loc31_6.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self: %i32.builtin = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .IntLiteral = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@Add.%Self: %Add.type) {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.b3d)]
// CHECK:STDOUT:   %Self.as_type.loc8_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.f73)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.f73)](%other.param_patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.f73)) -> @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.f73);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@As.%T.loc11_14.1: type, @As.%Self.1: @As.%As.type (%As.type.8ba)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:   %Self.as_type.loc12_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type.7f0)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(@ImplicitAs.%T.loc15_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:   %Self.as_type.loc16_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.2.%Self.as_type.loc16_20.1 (%Self.as_type.419)]() -> @Convert.2.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self.param_patt: %i32.builtin](%other.param_patt: %i32.builtin) -> %i32.builtin = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self.param_patt: Core.IntLiteral]() -> %i32.builtin = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4[%self.param_patt: Core.IntLiteral]() -> %i32.builtin = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5[%self.param_patt: %i32.builtin]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.b3d) {
// CHECK:STDOUT:   %Self => constants.%Self.b3d
// CHECK:STDOUT:   %Self.as_type.loc8_15.1 => constants.%Self.as_type.f73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%T) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_14.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self.b4e) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.8ba
// CHECK:STDOUT:   %Self => constants.%Self.b4e
// CHECK:STDOUT:   %Self.as_type.loc12_20.1 => constants.%Self.as_type.7f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%T.loc11_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.loc15_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc15_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%T, constants.%Self.0f3) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self.0f3
// CHECK:STDOUT:   %Self.as_type.loc16_20.1 => constants.%Self.as_type.419
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T.loc15_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Add.facet) {
// CHECK:STDOUT:   %Self => constants.%Add.facet
// CHECK:STDOUT:   %Self.as_type.loc8_15.1 => constants.%i32.builtin
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%i32.builtin
// CHECK:STDOUT:   %T.patt.loc11_14.2 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.a09
// CHECK:STDOUT:   %Self.2 => constants.%Self.b4e
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.c0d
// CHECK:STDOUT:   %Convert => constants.%Convert.713
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.567
// CHECK:STDOUT:   %assoc0.loc12_32.2 => constants.%assoc0.3bd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%i32.builtin, constants.%As.facet) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.a09
// CHECK:STDOUT:   %Self => constants.%As.facet
// CHECK:STDOUT:   %Self.as_type.loc12_20.1 => Core.IntLiteral
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc15_22.2 => constants.%i32.builtin
// CHECK:STDOUT:   %T.patt.loc15_22.2 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.11a
// CHECK:STDOUT:   %Self.2 => constants.%Self.0f3
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.752
// CHECK:STDOUT:   %Convert => constants.%Convert.fcc
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.dd3
// CHECK:STDOUT:   %assoc0.loc16_32.2 => constants.%assoc0.7cc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%i32.builtin, constants.%ImplicitAs.facet.1ff) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.11a
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.1ff
// CHECK:STDOUT:   %Self.as_type.loc16_20.1 => Core.IntLiteral
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc15_22.2 => Core.IntLiteral
// CHECK:STDOUT:   %T.patt.loc15_22.2 => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.9fc
// CHECK:STDOUT:   %Self.2 => constants.%Self.0f3
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.60e
// CHECK:STDOUT:   %Convert => constants.%Convert.c73
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.81c
// CHECK:STDOUT:   %assoc0.loc16_32.2 => constants.%assoc0.80e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(Core.IntLiteral, constants.%ImplicitAs.facet.72d) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.9fc
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.72d
// CHECK:STDOUT:   %Self.as_type.loc16_20.1 => constants.%i32.builtin
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %As.type.eed: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %Self.65a: %As.type.eed = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.843: type = fn_type @Convert.1, @As(%T) [symbolic]
// CHECK:STDOUT:   %Convert.95f: %Convert.type.843 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.04d: type = facet_access_type %Self.65a [symbolic]
// CHECK:STDOUT:   %As.assoc_type.a44: type = assoc_entity_type %As.type.eed [symbolic]
// CHECK:STDOUT:   %assoc0.ea3: %As.assoc_type.a44 = assoc_entity element0, imports.%Core.import_ref.708 [symbolic]
// CHECK:STDOUT:   %As.type.a6d: type = facet_type <@As, @As(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Convert.type.378: type = fn_type @Convert.1, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %Convert.e51: %Convert.type.378 = struct_value () [concrete]
// CHECK:STDOUT:   %As.assoc_type.e21: type = assoc_entity_type %As.type.a6d [concrete]
// CHECK:STDOUT:   %assoc0.545: %As.assoc_type.e21 = assoc_entity element0, imports.%Core.import_ref.708 [concrete]
// CHECK:STDOUT:   %assoc0.5bc: %As.assoc_type.a44 = assoc_entity element0, imports.%Core.import_ref.4e8 [symbolic]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [concrete]
// CHECK:STDOUT:   %Self.a99: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.61e: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert.2, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.40a: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.837: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0.43db8b.1: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic]
// CHECK:STDOUT:   %Convert.type.059: type = fn_type @Convert.2, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %Convert.4d7: %Convert.type.059 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.740: type = assoc_entity_type %ImplicitAs.type.61e [concrete]
// CHECK:STDOUT:   %assoc0.a81: %ImplicitAs.assoc_type.740 = assoc_entity element0, imports.%Core.import_ref.1c752f.1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.2, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %Convert.0e2: %Convert.type.71e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.29f: type = assoc_entity_type %ImplicitAs.type.2fd [concrete]
// CHECK:STDOUT:   %assoc0.ed3: %ImplicitAs.assoc_type.29f = assoc_entity element0, imports.%Core.import_ref.1c752f.2 [concrete]
// CHECK:STDOUT:   %impl_witness.d9b: <witness> = impl_witness (imports.%Core.import_ref.73a) [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.a6d = facet_value Core.IntLiteral, %impl_witness.d9b [concrete]
// CHECK:STDOUT:   %.dc4: type = fn_type_with_self_type %Convert.type.378, %As.facet [concrete]
// CHECK:STDOUT:   %Convert.type.953: type = fn_type @Convert.3 [concrete]
// CHECK:STDOUT:   %Convert.5bc: %Convert.type.953 = struct_value () [concrete]
// CHECK:STDOUT:   %Convert.bound.b34: <bound method> = bound_method %int_1.5b8, %Convert.5bc [concrete]
// CHECK:STDOUT:   %int_1.f38: %i32.builtin = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.324: <bound method> = bound_method %int_2.ecc, %Convert.5bc [concrete]
// CHECK:STDOUT:   %int_2.5a1: %i32.builtin = int_value 2 [concrete]
// CHECK:STDOUT:   %Add.assoc_type: type = assoc_entity_type %Add.type [concrete]
// CHECK:STDOUT:   %assoc0.b3c: %Add.assoc_type = assoc_entity element0, imports.%Core.import_ref.6dd [concrete]
// CHECK:STDOUT:   %impl_witness.bd0: <witness> = impl_witness (imports.%Core.import_ref.db4) [concrete]
// CHECK:STDOUT:   %Op.type.545: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Self.as_type.da9: type = facet_access_type %Self.a99 [symbolic]
// CHECK:STDOUT:   %Add.facet: %Add.type = facet_value %i32.builtin, %impl_witness.bd0 [concrete]
// CHECK:STDOUT:   %.7c2: type = fn_type_with_self_type %Op.type.545, %Add.facet [concrete]
// CHECK:STDOUT:   %Op.type.240: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.0e2: %Op.type.240 = struct_value () [concrete]
// CHECK:STDOUT:   %Op.bound.393: <bound method> = bound_method %int_1.f38, %Op.0e2 [concrete]
// CHECK:STDOUT:   %int_3.a0f: %i32.builtin = int_value 3 [concrete]
// CHECK:STDOUT:   %assoc0.43db8b.2: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207961.2 [symbolic]
// CHECK:STDOUT:   %impl_witness.8f3: <witness> = impl_witness (imports.%Core.import_ref.4f9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.6ef: %ImplicitAs.type.2fd = facet_value %i32.builtin, %impl_witness.8f3 [concrete]
// CHECK:STDOUT:   %.38f: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet.6ef [concrete]
// CHECK:STDOUT:   %Convert.type.0e4: type = fn_type @Convert.4 [concrete]
// CHECK:STDOUT:   %Convert.b32: %Convert.type.0e4 = struct_value () [concrete]
// CHECK:STDOUT:   %Convert.bound.65a: <bound method> = bound_method %int_3.a0f, %Convert.b32 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32.builtin [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.94d: <bound method> = bound_method %int_3.1ba, %Convert.5bc [concrete]
// CHECK:STDOUT:   %Convert.bound.8fc: <bound method> = bound_method %int_4.0c1, %Convert.5bc [concrete]
// CHECK:STDOUT:   %int_4.4f1: %i32.builtin = int_value 4 [concrete]
// CHECK:STDOUT:   %Op.bound.423: <bound method> = bound_method %int_3.a0f, %Op.0e2 [concrete]
// CHECK:STDOUT:   %int_7: %i32.builtin = int_value 7 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, %i32.builtin) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %impl_witness.39c: <witness> = impl_witness (imports.%Core.import_ref.f35) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.085: %ImplicitAs.type.61e = facet_value Core.IntLiteral, %impl_witness.39c [concrete]
// CHECK:STDOUT:   %.624: type = fn_type_with_self_type %Convert.type.059, %ImplicitAs.facet.085 [concrete]
// CHECK:STDOUT:   %Convert.type.49f: type = fn_type @Convert.5 [concrete]
// CHECK:STDOUT:   %Convert.cb5: %Convert.type.49f = struct_value () [concrete]
// CHECK:STDOUT:   %Convert.bound.b6b: <bound method> = bound_method %int_3.1ba, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %Convert.bound.626: <bound method> = bound_method %int_4.0c1, %Convert.cb5 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_3.a0f, %int_4.4f1, %int_7) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .Add = %Core.Add
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.f6b058.1: type = import_ref Core//default, loc11_14, loaded [symbolic = @As.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.a7c = import_ref Core//default, inst85 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.713: @As.%As.assoc_type (%As.assoc_type.a44) = import_ref Core//default, loc12_32, loaded [symbolic = @As.%assoc0 (constants.%assoc0.5bc)]
// CHECK:STDOUT:   %Core.Convert.313 = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.f6b058.2: type = import_ref Core//default, loc11_14, loaded [symbolic = @As.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.996: @As.%As.type (%As.type.eed) = import_ref Core//default, inst85 [no loc], loaded [symbolic = @As.%Self (constants.%Self.65a)]
// CHECK:STDOUT:   %Core.import_ref.708: @As.%Convert.type (%Convert.type.843) = import_ref Core//default, loc12_32, loaded [symbolic = @As.%Convert (constants.%Convert.95f)]
// CHECK:STDOUT:   %Core.import_ref.07c = import_ref Core//default, inst39 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.bdf: %Add.assoc_type = import_ref Core//default, loc8_41, loaded [concrete = constants.%assoc0.b3c]
// CHECK:STDOUT:   %Core.Op = import_ref Core//default, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.595: <witness> = import_ref Core//default, loc19_17, loaded [concrete = constants.%impl_witness.bd0]
// CHECK:STDOUT:   %Core.import_ref.c8c7cd.1: type = import_ref Core//default, loc19_6, loaded [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %Core.import_ref.bf0: type = import_ref Core//default, loc19_13, loaded [concrete = constants.%Add.type]
// CHECK:STDOUT:   %Core.import_ref.8aa: <witness> = import_ref Core//default, loc23_30, loaded [concrete = constants.%impl_witness.d9b]
// CHECK:STDOUT:   %Core.import_ref.8721d7.1: type = import_ref Core//default, loc23_17, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.1e5: type = import_ref Core//default, loc23_28, loaded [concrete = constants.%As.type.a6d]
// CHECK:STDOUT:   %Core.import_ref.f6b058.3: type = import_ref Core//default, loc15_22, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst128 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.630: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = import_ref Core//default, loc16_32, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.43db8b.2)]
// CHECK:STDOUT:   %Core.Convert.e69 = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.de9: <witness> = import_ref Core//default, loc27_38, loaded [concrete = constants.%impl_witness.39c]
// CHECK:STDOUT:   %Core.import_ref.8721d7.2: type = import_ref Core//default, loc27_17, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.4d9: type = import_ref Core//default, loc27_36, loaded [concrete = constants.%ImplicitAs.type.61e]
// CHECK:STDOUT:   %Core.import_ref.f6b058.4: type = import_ref Core//default, loc15_22, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst128 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.207961.1 = import_ref Core//default, loc16_32, unloaded
// CHECK:STDOUT:   %Core.import_ref.3b6: <witness> = import_ref Core//default, loc31_38, loaded [concrete = constants.%impl_witness.8f3]
// CHECK:STDOUT:   %Core.import_ref.c8c7cd.2: type = import_ref Core//default, loc31_6, loaded [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %Core.import_ref.efb: type = import_ref Core//default, loc31_36, loaded [concrete = constants.%ImplicitAs.type.2fd]
// CHECK:STDOUT:   %Core.import_ref.442: %Add.type = import_ref Core//default, inst39 [no loc], loaded [symbolic = constants.%Self.a99]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %arr.patt: %array_type = binding_pattern arr
// CHECK:STDOUT:     %.loc4_1: %array_type = var_pattern %arr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr.var: ref %array_type = var arr
// CHECK:STDOUT:   %.loc4_44: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc4_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc4_16: init type = call constants.%Int(%int_32.loc4_16) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_32.loc4_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc4_27: init type = call constants.%Int(%int_32.loc4_27) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_signed.loc4_27 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_signed.loc4_27, %.loc4_27.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %impl.elem0.loc4_24: %.dc4 = impl_witness_access constants.%impl_witness.d9b, element0 [concrete = constants.%Convert.5bc]
// CHECK:STDOUT:     %bound_method.loc4_24: <bound method> = bound_method %int_1, %impl.elem0.loc4_24 [concrete = constants.%Convert.bound.b34]
// CHECK:STDOUT:     %int.convert_checked.loc4_24: init %i32.builtin = call %bound_method.loc4_24(%int_1) [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc4_24.1: %i32.builtin = value_of_initializer %int.convert_checked.loc4_24 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc4_24.2: %i32.builtin = converted %int_1, %.loc4_24.1 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %int_32.loc4_40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc4_40: init type = call constants.%Int(%int_32.loc4_40) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc4_40.1: type = value_of_initializer %int.make_type_signed.loc4_40 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc4_40.2: type = converted %int.make_type_signed.loc4_40, %.loc4_40.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %impl.elem0.loc4_37: %.dc4 = impl_witness_access constants.%impl_witness.d9b, element0 [concrete = constants.%Convert.5bc]
// CHECK:STDOUT:     %bound_method.loc4_37: <bound method> = bound_method %int_2, %impl.elem0.loc4_37 [concrete = constants.%Convert.bound.324]
// CHECK:STDOUT:     %int.convert_checked.loc4_37: init %i32.builtin = call %bound_method.loc4_37(%int_2) [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc4_37.1: %i32.builtin = value_of_initializer %int.convert_checked.loc4_37 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc4_37.2: %i32.builtin = converted %int_2, %.loc4_37.1 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %impl.elem0.loc4_32.1: %.7c2 = impl_witness_access constants.%impl_witness.bd0, element0 [concrete = constants.%Op.0e2]
// CHECK:STDOUT:     %bound_method.loc4_32.1: <bound method> = bound_method %.loc4_24.2, %impl.elem0.loc4_32.1 [concrete = constants.%Op.bound.393]
// CHECK:STDOUT:     %int.sadd: init %i32.builtin = call %bound_method.loc4_32.1(%.loc4_24.2, %.loc4_37.2) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %.loc4_16.1: type = value_of_initializer %int.make_type_signed.loc4_16 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc4_16.2: type = converted %int.make_type_signed.loc4_16, %.loc4_16.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %impl.elem0.loc4_32.2: %.38f = impl_witness_access constants.%impl_witness.8f3, element0 [concrete = constants.%Convert.b32]
// CHECK:STDOUT:     %bound_method.loc4_32.2: <bound method> = bound_method %int.sadd, %impl.elem0.loc4_32.2 [concrete = constants.%Convert.bound.65a]
// CHECK:STDOUT:     %.loc4_32.1: %i32.builtin = value_of_initializer %int.sadd [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %.loc4_32.2: %i32.builtin = converted %int.sadd, %.loc4_32.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %int.convert_checked.loc4_32: init Core.IntLiteral = call %bound_method.loc4_32.2(%.loc4_32.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc4_32.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc4_32 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc4_32.4: Core.IntLiteral = converted %int.sadd, %.loc4_32.3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %.loc4_32.4, %i32.builtin [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr: ref %array_type = bind_name arr, %arr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(imports.%Core.import_ref.f6b058.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.eed)]
// CHECK:STDOUT:   %Self: %As.type.eed = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.65a)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%T) [symbolic = %Convert.type (constants.%Convert.type.843)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.843) = struct_value () [symbolic = %Convert (constants.%Convert.95f)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As.%As.type (%As.type.eed) [symbolic = %As.assoc_type (constants.%As.assoc_type.a44)]
// CHECK:STDOUT:   %assoc0: @As.%As.assoc_type (%As.assoc_type.a44) = assoc_entity element0, imports.%Core.import_ref.708 [symbolic = %assoc0 (constants.%assoc0.ea3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.a7c
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.713
// CHECK:STDOUT:     witness = (imports.%Core.Convert.313)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.07c
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.bdf
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.f6b058.3: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.837)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic = %assoc0 (constants.%assoc0.43db8b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.630
// CHECK:STDOUT:     witness = (imports.%Core.Convert.e69)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%Core.import_ref.c8c7cd.1 as imports.%Core.import_ref.bf0 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.595
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: imports.%Core.import_ref.8721d7.1 as imports.%Core.import_ref.1e5 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.8aa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: imports.%Core.import_ref.8721d7.2 as imports.%Core.import_ref.4d9 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.de9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%Core.import_ref.c8c7cd.2 as imports.%Core.import_ref.efb [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.3b6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(imports.%Core.import_ref.f6b058.2: type, imports.%Core.import_ref.996: @As.%As.type (%As.type.eed)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.eed)]
// CHECK:STDOUT:   %Self: %As.type.eed = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.65a)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.04d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type (%Self.as_type.04d)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(imports.%Core.import_ref.f6b058.4: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.40a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.2.%Self.as_type (%Self.as_type.40a)]() -> @Convert.2.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self.param_patt: Core.IntLiteral]() -> %i32.builtin = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.442: %Add.type) [from "core.carbon"] {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.a99)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.da9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type (%Self.as_type.da9)](%other.param_patt: @Op.1.%Self.as_type (%Self.as_type.da9)) -> @Op.1.%Self.as_type (%Self.as_type.da9);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self.param_patt: %i32.builtin](%other.param_patt: %i32.builtin) -> %i32.builtin = "int.sadd" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4[%self.param_patt: %i32.builtin]() -> Core.IntLiteral = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5[%self.param_patt: Core.IntLiteral]() -> %i32.builtin = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_3.loc4_49: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc4_52: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %int_3.loc4_56: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_32.loc4_61: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc4_61: init type = call constants.%Int(%int_32.loc4_61) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_61.1: type = value_of_initializer %int.make_type_signed.loc4_61 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_61.2: type = converted %int.make_type_signed.loc4_61, %.loc4_61.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %impl.elem0.loc4_58: %.dc4 = impl_witness_access constants.%impl_witness.d9b, element0 [concrete = constants.%Convert.5bc]
// CHECK:STDOUT:   %bound_method.loc4_58: <bound method> = bound_method %int_3.loc4_56, %impl.elem0.loc4_58 [concrete = constants.%Convert.bound.94d]
// CHECK:STDOUT:   %int.convert_checked.loc4_58: init %i32.builtin = call %bound_method.loc4_58(%int_3.loc4_56) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_58.1: %i32.builtin = value_of_initializer %int.convert_checked.loc4_58 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_58.2: %i32.builtin = converted %int_3.loc4_56, %.loc4_58.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %int_4.loc4_69: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %int_32.loc4_74: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc4_74: init type = call constants.%Int(%int_32.loc4_74) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_74.1: type = value_of_initializer %int.make_type_signed.loc4_74 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_74.2: type = converted %int.make_type_signed.loc4_74, %.loc4_74.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %impl.elem0.loc4_71: %.dc4 = impl_witness_access constants.%impl_witness.d9b, element0 [concrete = constants.%Convert.5bc]
// CHECK:STDOUT:   %bound_method.loc4_71: <bound method> = bound_method %int_4.loc4_69, %impl.elem0.loc4_71 [concrete = constants.%Convert.bound.8fc]
// CHECK:STDOUT:   %int.convert_checked.loc4_71: init %i32.builtin = call %bound_method.loc4_71(%int_4.loc4_69) [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc4_71.1: %i32.builtin = value_of_initializer %int.convert_checked.loc4_71 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc4_71.2: %i32.builtin = converted %int_4.loc4_69, %.loc4_71.1 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %impl.elem0.loc4_66: %.7c2 = impl_witness_access constants.%impl_witness.bd0, element0 [concrete = constants.%Op.0e2]
// CHECK:STDOUT:   %bound_method.loc4_66: <bound method> = bound_method %.loc4_58.2, %impl.elem0.loc4_66 [concrete = constants.%Op.bound.423]
// CHECK:STDOUT:   %int.sadd: init %i32.builtin = call %bound_method.loc4_66(%.loc4_58.2, %.loc4_71.2) [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc4_78.1: %tuple.type = tuple_literal (%int_3.loc4_49, %int_4.loc4_52, %int.sadd)
// CHECK:STDOUT:   %impl.elem0.loc4_78.1: %.624 = impl_witness_access constants.%impl_witness.39c, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc4_78.1: <bound method> = bound_method %int_3.loc4_49, %impl.elem0.loc4_78.1 [concrete = constants.%Convert.bound.b6b]
// CHECK:STDOUT:   %int.convert_checked.loc4_78.1: init %i32.builtin = call %bound_method.loc4_78.1(%int_3.loc4_49) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_78.2: init %i32.builtin = converted %int_3.loc4_49, %int.convert_checked.loc4_78.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc4_78.3: ref %i32.builtin = array_index file.%arr.var, %int_0
// CHECK:STDOUT:   %.loc4_78.4: init %i32.builtin = initialize_from %.loc4_78.2 to %.loc4_78.3 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %impl.elem0.loc4_78.2: %.624 = impl_witness_access constants.%impl_witness.39c, element0 [concrete = constants.%Convert.cb5]
// CHECK:STDOUT:   %bound_method.loc4_78.2: <bound method> = bound_method %int_4.loc4_52, %impl.elem0.loc4_78.2 [concrete = constants.%Convert.bound.626]
// CHECK:STDOUT:   %int.convert_checked.loc4_78.2: init %i32.builtin = call %bound_method.loc4_78.2(%int_4.loc4_52) [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc4_78.5: init %i32.builtin = converted %int_4.loc4_52, %int.convert_checked.loc4_78.2 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc4_78.6: ref %i32.builtin = array_index file.%arr.var, %int_1
// CHECK:STDOUT:   %.loc4_78.7: init %i32.builtin = initialize_from %.loc4_78.5 to %.loc4_78.6 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc4_78.8: ref %i32.builtin = array_index file.%arr.var, %int_2
// CHECK:STDOUT:   %.loc4_78.9: init %i32.builtin = initialize_from %int.sadd to %.loc4_78.8 [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc4_78.10: init %array_type = array_init (%.loc4_78.4, %.loc4_78.7, %.loc4_78.9) to file.%arr.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc4_1: init %array_type = converted %.loc4_78.1, %.loc4_78.10 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%arr.var, %.loc4_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self.65a) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.eed
// CHECK:STDOUT:   %Self => constants.%Self.65a
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.04d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %T.patt => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.a6d
// CHECK:STDOUT:   %Self => constants.%Self.65a
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.378
// CHECK:STDOUT:   %Convert => constants.%Convert.e51
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.e21
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.545
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %T.patt => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.61e
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.059
// CHECK:STDOUT:   %Convert => constants.%Convert.4d7
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.740
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%T, constants.%Self.519) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.40a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %T.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2fd
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.71e
// CHECK:STDOUT:   %Convert => constants.%Convert.0e2
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.29f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ed3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.a99) {
// CHECK:STDOUT:   %Self => constants.%Self.a99
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.da9
// CHECK:STDOUT: }
// CHECK:STDOUT:
