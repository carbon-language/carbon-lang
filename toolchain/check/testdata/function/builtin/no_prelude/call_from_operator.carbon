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

var arr: [i32; (1 as i32) + (2 as i32)] = (3, 4, (3 as i32) + (4 as i32));

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [template]
// CHECK:STDOUT:   %Self.0: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.c: type = facet_access_type %Self.0 [symbolic]
// CHECK:STDOUT:   %Op.type.94cce3.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.01c4da.1: %Op.type.94cce3.1 = struct_value () [template]
// CHECK:STDOUT:   %Op.assoc_type: type = assoc_entity_type %Add.type, %Op.type.94cce3.1 [template]
// CHECK:STDOUT:   %assoc0.1: %Op.assoc_type = assoc_entity element0, @Add.%Op.decl [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %As.type.0: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As.generic: %As.type.0 = struct_value () [template]
// CHECK:STDOUT:   %As.type.c: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %Self.b: %As.type.c = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.9: type = facet_access_type %Self.b [symbolic]
// CHECK:STDOUT:   %Convert.type.c2: type = fn_type @Convert.1, @As(%T) [symbolic]
// CHECK:STDOUT:   %Convert.e: %Convert.type.c2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Convert.assoc_type.d: type = assoc_entity_type %As.type.c, %Convert.type.c2 [symbolic]
// CHECK:STDOUT:   %assoc0.2: %Convert.assoc_type.d = assoc_entity element0, @As.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.03: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.03 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.e: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.c: %ImplicitAs.type.e = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.0: type = facet_access_type %Self.c [symbolic]
// CHECK:STDOUT:   %Convert.type.36: type = fn_type @Convert.2, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.dd: %Convert.type.36 = struct_value () [symbolic]
// CHECK:STDOUT:   %Convert.assoc_type.2: type = assoc_entity_type %ImplicitAs.type.e, %Convert.type.36 [symbolic]
// CHECK:STDOUT:   %assoc0.cf: %Convert.assoc_type.2 = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %impl_witness.f: <witness> = impl_witness (@impl.1.%Op.decl) [template]
// CHECK:STDOUT:   %Op.type.94cce3.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.01c4da.2: %Op.type.94cce3.2 = struct_value () [template]
// CHECK:STDOUT:   %Add.facet: %Add.type = facet_value %i32.builtin, %impl_witness.f [template]
// CHECK:STDOUT:   %As.type.9: type = facet_type <@As, @As(%i32.builtin)> [template]
// CHECK:STDOUT:   %Convert.type.c9: type = fn_type @Convert.1, @As(%i32.builtin) [template]
// CHECK:STDOUT:   %Convert.f49: %Convert.type.c9 = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.8: type = assoc_entity_type %As.type.9, %Convert.type.c9 [template]
// CHECK:STDOUT:   %assoc0.5: %Convert.assoc_type.8 = assoc_entity element0, @As.%Convert.decl [template]
// CHECK:STDOUT:   %impl_witness.7560bb.1: <witness> = impl_witness (@impl.2.%Convert.decl) [template]
// CHECK:STDOUT:   %Convert.type.a33e4b.1: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.5baf2f.1: %Convert.type.a33e4b.1 = struct_value () [template]
// CHECK:STDOUT:   %As.facet: %As.type.c = facet_value Core.IntLiteral, %impl_witness.7560bb.1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.00: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [template]
// CHECK:STDOUT:   %Convert.type.31: type = fn_type @Convert.2, @ImplicitAs(%i32.builtin) [template]
// CHECK:STDOUT:   %Convert.f4c: %Convert.type.31 = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.7: type = assoc_entity_type %ImplicitAs.type.00, %Convert.type.31 [template]
// CHECK:STDOUT:   %assoc0.c6: %Convert.assoc_type.7 = assoc_entity element0, @ImplicitAs.%Convert.decl [template]
// CHECK:STDOUT:   %impl_witness.7560bb.2: <witness> = impl_witness (@impl.3.%Convert.decl) [template]
// CHECK:STDOUT:   %Convert.type.a33e4b.2: type = fn_type @Convert.4 [template]
// CHECK:STDOUT:   %Convert.5baf2f.2: %Convert.type.a33e4b.2 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet.6: %ImplicitAs.type.e = facet_value Core.IntLiteral, %impl_witness.7560bb.2 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.b: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.d: type = fn_type @Convert.2, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.d4: %Convert.type.d = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.3: type = assoc_entity_type %ImplicitAs.type.b, %Convert.type.d [template]
// CHECK:STDOUT:   %assoc0.3: %Convert.assoc_type.3 = assoc_entity element0, @ImplicitAs.%Convert.decl [template]
// CHECK:STDOUT:   %impl_witness.0: <witness> = impl_witness (@impl.4.%Convert.decl) [template]
// CHECK:STDOUT:   %Convert.type.a33e4b.3: type = fn_type @Convert.5 [template]
// CHECK:STDOUT:   %Convert.5baf2f.3: %Convert.type.a33e4b.3 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet.5: %ImplicitAs.type.e = facet_value %i32.builtin, %impl_witness.0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
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
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc5_22.1: type = splice_block %.loc5_22.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_22.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_22.3: type = converted %int_literal.make_type, %.loc5_22.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N: Core.IntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Add.decl: type = interface_decl @Add [template = constants.%Add.type] {} {}
// CHECK:STDOUT:   %As.decl: %As.type.0 = interface_decl @As [template = constants.%As.generic] {
// CHECK:STDOUT:     %T.patt.loc11_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_14.1, runtime_param<invalid> [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.03 = interface_decl @ImplicitAs [template = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %T.patt.loc15_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_22.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc15_22.1, runtime_param<invalid> [symbolic = %T.patt.loc15_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc15_22.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc15_22.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc19_6.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc19_6.2: type = converted %int.make_type_signed, %.loc19_6.1 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, file.%Add.decl [template = constants.%Add.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc19: <witness> = impl_witness (@impl.1.%Op.decl) [template = constants.%impl_witness.f]
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc23_17.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc23_17.2: type = converted %int_literal.make_type, %.loc23_17.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %As.ref: %As.type.0 = name_ref As, file.%As.decl [template = constants.%As.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc23_28.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc23_28.2: type = converted %int.make_type_signed, %.loc23_28.1 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %As.type: type = facet_type <@As, @As(constants.%i32.builtin)> [template = constants.%As.type.9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc23: <witness> = impl_witness (@impl.2.%Convert.decl) [template = constants.%impl_witness.7560bb.1]
// CHECK:STDOUT:   impl_decl @impl.3 [template] {} {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc27_17.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc27_17.2: type = converted %int_literal.make_type, %.loc27_17.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.03 = name_ref ImplicitAs, file.%ImplicitAs.decl [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc27_36.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc27_36.2: type = converted %int.make_type_signed, %.loc27_36.1 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32.builtin)> [template = constants.%ImplicitAs.type.00]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc27: <witness> = impl_witness (@impl.3.%Convert.decl) [template = constants.%impl_witness.7560bb.2]
// CHECK:STDOUT:   impl_decl @impl.4 [template] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc31_6.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc31_6.2: type = converted %int.make_type_signed, %.loc31_6.1 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.03 = name_ref ImplicitAs, file.%ImplicitAs.decl [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc31_36.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc31_36.2: type = converted %int_literal.make_type, %.loc31_36.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template = constants.%ImplicitAs.type.b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc31: <witness> = impl_witness (@impl.4.%Convert.decl) [template = constants.%impl_witness.0]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.0]
// CHECK:STDOUT:   %Op.decl: %Op.type.94cce3.1 = fn_decl @Op.1 [template = constants.%Op.01c4da.1] {
// CHECK:STDOUT:     %self.patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c) = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c) = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c) = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc8_37: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self.0)]
// CHECK:STDOUT:     %Self.as_type.loc8_37: type = facet_access_type %Self.ref.loc8_37 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.c)]
// CHECK:STDOUT:     %.loc8_37: type = converted %Self.ref.loc8_37, %Self.as_type.loc8_37 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.c)]
// CHECK:STDOUT:     %self.param: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c) = value_param runtime_param0
// CHECK:STDOUT:     %.loc8_15.1: type = splice_block %.loc8_15.2 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.c)] {
// CHECK:STDOUT:       %Self.ref.loc8_15: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self.0)]
// CHECK:STDOUT:       %Self.as_type.loc8_15.2: type = facet_access_type %Self.ref.loc8_15 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.c)]
// CHECK:STDOUT:       %.loc8_15.2: type = converted %Self.ref.loc8_15, %Self.as_type.loc8_15.2 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c) = value_param runtime_param1
// CHECK:STDOUT:     %.loc8_28.1: type = splice_block %.loc8_28.2 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.c)] {
// CHECK:STDOUT:       %Self.ref.loc8_28: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self.0)]
// CHECK:STDOUT:       %Self.as_type.loc8_28: type = facet_access_type %Self.ref.loc8_28 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.c)]
// CHECK:STDOUT:       %.loc8_28.2: type = converted %Self.ref.loc8_28, %Self.as_type.loc8_28 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c) = out_param runtime_param2
// CHECK:STDOUT:     %return: ref @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Op.assoc_type = assoc_entity element0, %Op.decl [template = constants.%assoc0.1]
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
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T.loc11_14.2)> [symbolic = %As.type (constants.%As.type.c)]
// CHECK:STDOUT:   %Self.2: %As.type.c = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%T.loc11_14.2) [symbolic = %Convert.type (constants.%Convert.type.c2)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.c2) = struct_value () [symbolic = %Convert (constants.%Convert.e)]
// CHECK:STDOUT:   %Convert.assoc_type: type = assoc_entity_type @As.%As.type (%As.type.c), @As.%Convert.type (%Convert.type.c2) [symbolic = %Convert.assoc_type (constants.%Convert.assoc_type.d)]
// CHECK:STDOUT:   %assoc0.loc12_32.2: @As.%Convert.assoc_type (%Convert.assoc_type.d) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc12_32.2 (constants.%assoc0.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @As.%As.type (%As.type.c) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b)]
// CHECK:STDOUT:     %Convert.decl: @As.%Convert.type (%Convert.type.c2) = fn_decl @Convert.1 [symbolic = @As.%Convert (constants.%Convert.e)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type.9) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type.9) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @As.%T.loc11_14.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type.9) = value_param runtime_param0
// CHECK:STDOUT:       %.loc12_20.1: type = splice_block %.loc12_20.3 [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.9)] {
// CHECK:STDOUT:         %.loc12_20.2: @Convert.1.%As.type (%As.type.c) = specific_constant @As.%Self.1, @As(constants.%T) [symbolic = %Self (constants.%Self.b)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%As.type (%As.type.c) = name_ref Self, %.loc12_20.2 [symbolic = %Self (constants.%Self.b)]
// CHECK:STDOUT:         %Self.as_type.loc12_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.9)]
// CHECK:STDOUT:         %.loc12_20.3: type = converted %Self.ref, %Self.as_type.loc12_20.2 [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.9)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type.9) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc12_32.1: @As.%Convert.assoc_type (%Convert.assoc_type.d) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc12_32.2 (constants.%assoc0.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
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
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc15_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.e)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.e = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.c)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%T.loc15_22.2) [symbolic = %Convert.type (constants.%Convert.type.36)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.36) = struct_value () [symbolic = %Convert (constants.%Convert.dd)]
// CHECK:STDOUT:   %Convert.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.e), @ImplicitAs.%Convert.type (%Convert.type.36) [symbolic = %Convert.assoc_type (constants.%Convert.assoc_type.2)]
// CHECK:STDOUT:   %assoc0.loc16_32.2: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.2) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc16_32.2 (constants.%assoc0.cf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.e) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.c)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.36) = fn_decl @Convert.2 [symbolic = @ImplicitAs.%Convert (constants.%Convert.dd)] {
// CHECK:STDOUT:       %self.patt: @Convert.2.%Self.as_type.loc16_20.1 (%Self.as_type.0) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.2.%Self.as_type.loc16_20.1 (%Self.as_type.0) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Convert.2.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.2.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc15_22.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.2.%Self.as_type.loc16_20.1 (%Self.as_type.0) = value_param runtime_param0
// CHECK:STDOUT:       %.loc16_20.1: type = splice_block %.loc16_20.3 [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.0)] {
// CHECK:STDOUT:         %.loc16_20.2: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.e) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self.c)]
// CHECK:STDOUT:         %Self.ref: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.e) = name_ref Self, %.loc16_20.2 [symbolic = %Self (constants.%Self.c)]
// CHECK:STDOUT:         %Self.as_type.loc16_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.0)]
// CHECK:STDOUT:         %.loc16_20.3: type = converted %Self.ref, %Self.as_type.loc16_20.2 [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.0)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.2.%Self.as_type.loc16_20.1 (%Self.as_type.0) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.2.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Convert.2.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc16_32.1: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.2) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc16_32.2 (constants.%assoc0.cf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Convert = %assoc0.loc16_32.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %.loc19_6.2 as %Add.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.94cce3.2 = fn_decl @Op.2 [template = constants.%Op.01c4da.2] {
// CHECK:STDOUT:     %self.patt: %i32.builtin = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %i32.builtin = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: %i32.builtin = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %i32.builtin = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32.builtin = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32.builtin = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc20_37: type = name_ref Self, @impl.1.%.loc19_6.2 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %self.param: %i32.builtin = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref.loc20_15: type = name_ref Self, @impl.1.%.loc19_6.2 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %self: %i32.builtin = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %i32.builtin = value_param runtime_param1
// CHECK:STDOUT:     %Self.ref.loc20_28: type = name_ref Self, @impl.1.%.loc19_6.2 [template = constants.%i32.builtin]
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
// CHECK:STDOUT:   %Convert.decl: %Convert.type.a33e4b.1 = fn_decl @Convert.3 [template = constants.%Convert.5baf2f.1] {
// CHECK:STDOUT:     %self.patt: Core.IntLiteral = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: Core.IntLiteral = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32.builtin = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32.builtin = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc24_31.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc24_31.2: type = converted %int.make_type_signed, %.loc24_31.1 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.2.%.loc23_17.2 [template = Core.IntLiteral]
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
// CHECK:STDOUT:   %Convert.decl: %Convert.type.a33e4b.2 = fn_decl @Convert.4 [template = constants.%Convert.5baf2f.2] {
// CHECK:STDOUT:     %self.patt: Core.IntLiteral = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: Core.IntLiteral = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32.builtin = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32.builtin = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc28_31.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc28_31.2: type = converted %int.make_type_signed, %.loc28_31.1 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.3.%.loc27_17.2 [template = Core.IntLiteral]
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
// CHECK:STDOUT:   %Convert.decl: %Convert.type.a33e4b.3 = fn_decl @Convert.5 [template = constants.%Convert.5baf2f.3] {
// CHECK:STDOUT:     %self.patt: %i32.builtin = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %i32.builtin = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc32_42.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc32_42.2: type = converted %int_literal.make_type, %.loc32_42.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %self.param: %i32.builtin = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.4.%.loc31_6.2 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %self: %i32.builtin = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@Add.%Self: %Add.type) {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.0)]
// CHECK:STDOUT:   %Self.as_type.loc8_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c)](%other.param_patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c)) -> @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@As.%T.loc11_14.1: type, @As.%Self.1: @As.%As.type (%As.type.c)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.c)]
// CHECK:STDOUT:   %Self: %As.type.c = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.b)]
// CHECK:STDOUT:   %Self.as_type.loc12_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type.9)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(@ImplicitAs.%T.loc15_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.e)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.e)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.e = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.c)]
// CHECK:STDOUT:   %Self.as_type.loc16_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.2.%Self.as_type.loc16_20.1 (%Self.as_type.0)]() -> @Convert.2.%T (%T);
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
// CHECK:STDOUT: specific @Op.1(constants.%Self.0) {
// CHECK:STDOUT:   %Self => constants.%Self.0
// CHECK:STDOUT:   %Self.as_type.loc8_15.1 => constants.%Self.as_type.c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%T) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_14.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self.b) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.c
// CHECK:STDOUT:   %Self => constants.%Self.b
// CHECK:STDOUT:   %Self.as_type.loc12_20.1 => constants.%Self.as_type.9
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
// CHECK:STDOUT: specific @Convert.2(constants.%T, constants.%Self.c) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.e
// CHECK:STDOUT:   %Self => constants.%Self.c
// CHECK:STDOUT:   %Self.as_type.loc16_20.1 => constants.%Self.as_type.0
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
// CHECK:STDOUT:   %As.type => constants.%As.type.9
// CHECK:STDOUT:   %Self.2 => constants.%Self.b
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.c9
// CHECK:STDOUT:   %Convert => constants.%Convert.f49
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.8
// CHECK:STDOUT:   %assoc0.loc12_32.2 => constants.%assoc0.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%i32.builtin, constants.%As.facet) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.9
// CHECK:STDOUT:   %Self => constants.%As.facet
// CHECK:STDOUT:   %Self.as_type.loc12_20.1 => Core.IntLiteral
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc15_22.2 => constants.%i32.builtin
// CHECK:STDOUT:   %T.patt.loc15_22.2 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.00
// CHECK:STDOUT:   %Self.2 => constants.%Self.c
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.31
// CHECK:STDOUT:   %Convert => constants.%Convert.f4c
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.7
// CHECK:STDOUT:   %assoc0.loc16_32.2 => constants.%assoc0.c6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%i32.builtin, constants.%ImplicitAs.facet.6) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.00
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.6
// CHECK:STDOUT:   %Self.as_type.loc16_20.1 => Core.IntLiteral
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc15_22.2 => Core.IntLiteral
// CHECK:STDOUT:   %T.patt.loc15_22.2 => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.b
// CHECK:STDOUT:   %Self.2 => constants.%Self.c
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.d
// CHECK:STDOUT:   %Convert => constants.%Convert.d4
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.3
// CHECK:STDOUT:   %assoc0.loc16_32.2 => constants.%assoc0.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(Core.IntLiteral, constants.%ImplicitAs.facet.5) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.b
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.5
// CHECK:STDOUT:   %Self.as_type.loc16_20.1 => constants.%i32.builtin
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_1.5: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %As.type.5: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %Self.3: %As.type.5 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.1, @As(%T) [symbolic]
// CHECK:STDOUT:   %Convert.f: %Convert.type.12 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.a5: type = facet_access_type %Self.3 [symbolic]
// CHECK:STDOUT:   %Convert.assoc_type.7: type = assoc_entity_type %As.type.5, %Convert.type.12 [symbolic]
// CHECK:STDOUT:   %assoc0.ba1273.1: %Convert.assoc_type.7 = assoc_entity element0, imports.%import_ref.912b7f.1 [symbolic]
// CHECK:STDOUT:   %As.type.0: type = facet_type <@As, @As(%i32.builtin)> [template]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.1, @As(%i32.builtin) [template]
// CHECK:STDOUT:   %Convert.29: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.d: type = assoc_entity_type %As.type.0, %Convert.type.9 [template]
// CHECK:STDOUT:   %assoc0.93: %Convert.assoc_type.d = assoc_entity element0, imports.%import_ref.912b7f.1 [template]
// CHECK:STDOUT:   %assoc0.ba1273.2: %Convert.assoc_type.7 = assoc_entity element0, imports.%import_ref.912b7f.2 [symbolic]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [template]
// CHECK:STDOUT:   %Self.d5d: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.d5f: %ImplicitAs.type.d = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.e: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [template]
// CHECK:STDOUT:   %Convert.type.29f: type = fn_type @Convert.2, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.9: %Convert.type.29f = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.8: type = facet_access_type %Self.d5f [symbolic]
// CHECK:STDOUT:   %Convert.assoc_type.5b: type = assoc_entity_type %ImplicitAs.type.d, %Convert.type.29f [symbolic]
// CHECK:STDOUT:   %assoc0.ae5e08.1: %Convert.assoc_type.5b = assoc_entity element0, imports.%import_ref.a52db1.1 [symbolic]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.2, @ImplicitAs(%i32.builtin) [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.13 = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.5f: type = assoc_entity_type %ImplicitAs.type.e, %Convert.type.13 [template]
// CHECK:STDOUT:   %assoc0.97: %Convert.assoc_type.5f = assoc_entity element0, imports.%import_ref.a52db1.2 [template]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.c: type = fn_type @Convert.2, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.b: %Convert.type.c = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.1: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.c [template]
// CHECK:STDOUT:   %assoc0.f: %Convert.assoc_type.1 = assoc_entity element0, imports.%import_ref.a52db1.3 [template]
// CHECK:STDOUT:   %impl_witness.2ccba5.1: <witness> = impl_witness (imports.%import_ref.e59014.1) [template]
// CHECK:STDOUT:   %Convert.type.2957cc.1: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.2b8aa5.1: %Convert.type.2957cc.1 = struct_value () [template]
// CHECK:STDOUT:   %int_2.e: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Op.type.dd8875.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Self.as_type.a9: type = facet_access_type %Self.d5d [symbolic]
// CHECK:STDOUT:   %Op.assoc_type: type = assoc_entity_type %Add.type, %Op.type.dd8875.1 [template]
// CHECK:STDOUT:   %assoc0.0: %Op.assoc_type = assoc_entity element0, imports.%import_ref.592 [template]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (imports.%import_ref.12) [template]
// CHECK:STDOUT:   %Op.type.dd8875.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.9f2095.2: %Op.type.dd8875.2 = struct_value () [template]
// CHECK:STDOUT:   %int_3.a: %i32.builtin = int_value 3 [template]
// CHECK:STDOUT:   %assoc0.ae5e08.2: %Convert.assoc_type.5b = assoc_entity element0, imports.%import_ref.a52db1.4 [symbolic]
// CHECK:STDOUT:   %impl_witness.b: <witness> = impl_witness (imports.%import_ref.ed) [template]
// CHECK:STDOUT:   %int_3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1, %i32.builtin [template]
// CHECK:STDOUT:   %int_4.0: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.34e9bc.1: <bound method> = bound_method %int_3.1, %Convert.2b8aa5.1 [template]
// CHECK:STDOUT:   %Convert.bound.87a6cb.1: <bound method> = bound_method %int_4.0, %Convert.2b8aa5.1 [template]
// CHECK:STDOUT:   %int_4.4: %i32.builtin = int_value 4 [template]
// CHECK:STDOUT:   %Op.bound.7: <bound method> = bound_method %int_3.a, %Op.9f2095.2 [template]
// CHECK:STDOUT:   %int_7: %i32.builtin = int_value 7 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, %i32.builtin) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %impl_witness.2ccba5.2: <witness> = impl_witness (imports.%import_ref.e59014.2) [template]
// CHECK:STDOUT:   %Convert.type.2957cc.3: type = fn_type @Convert.5 [template]
// CHECK:STDOUT:   %Convert.2b8aa5.3: %Convert.type.2957cc.3 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.34e9bc.2: <bound method> = bound_method %int_3.1, %Convert.2b8aa5.3 [template]
// CHECK:STDOUT:   %Convert.bound.87a6cb.2: <bound method> = bound_method %int_4.0, %Convert.2b8aa5.3 [template]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_3.a, %int_4.4, %int_7) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.4d
// CHECK:STDOUT:     .As = %import_ref.a8
// CHECK:STDOUT:     .Add = %import_ref.64
// CHECK:STDOUT:     .ImplicitAs = %import_ref.7
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.22 = import_ref Core//default, inst85 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.d: @As.%Convert.assoc_type (%Convert.assoc_type.7) = import_ref Core//default, loc12_32, loaded [symbolic = @As.%assoc0 (constants.%assoc0.ba1273.2)]
// CHECK:STDOUT:   %import_ref.80c = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %import_ref.912b7f.1 = import_ref Core//default, loc12_32, unloaded
// CHECK:STDOUT:   %import_ref.98: <witness> = import_ref Core//default, loc19_17, loaded [template = constants.%impl_witness.5]
// CHECK:STDOUT:   %import_ref.6c = import_ref Core//default, inst39 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.16: %Op.assoc_type = import_ref Core//default, loc8_41, loaded [template = constants.%assoc0.0]
// CHECK:STDOUT:   %import_ref.47 = import_ref Core//default, Op, unloaded
// CHECK:STDOUT:   %import_ref.5b1455.1: type = import_ref Core//default, loc19_6, loaded [template = constants.%i32.builtin]
// CHECK:STDOUT:   %import_ref.e7: type = import_ref Core//default, loc19_13, loaded [template = constants.%Add.type]
// CHECK:STDOUT:   %import_ref.21938e.1: <witness> = import_ref Core//default, loc23_30, loaded [template = constants.%impl_witness.2ccba5.1]
// CHECK:STDOUT:   %import_ref.c8538c.1: type = import_ref Core//default, loc23_17, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.b: type = import_ref Core//default, loc23_28, loaded [template = constants.%As.type.0]
// CHECK:STDOUT:   %import_ref.21938e.2: <witness> = import_ref Core//default, loc27_38, loaded [template = constants.%impl_witness.2ccba5.2]
// CHECK:STDOUT:   %import_ref.809 = import_ref Core//default, inst128 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.3: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.5b) = import_ref Core//default, loc16_32, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.ae5e08.2)]
// CHECK:STDOUT:   %import_ref.44 = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %import_ref.c8538c.2: type = import_ref Core//default, loc27_17, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.9f: type = import_ref Core//default, loc27_36, loaded [template = constants.%ImplicitAs.type.e]
// CHECK:STDOUT:   %import_ref.a52db1.1 = import_ref Core//default, loc16_32, unloaded
// CHECK:STDOUT:   %import_ref.45: <witness> = import_ref Core//default, loc31_38, loaded [template = constants.%impl_witness.b]
// CHECK:STDOUT:   %import_ref.5b1455.2: type = import_ref Core//default, loc31_6, loaded [template = constants.%i32.builtin]
// CHECK:STDOUT:   %import_ref.593: type = import_ref Core//default, loc31_36, loaded [template = constants.%ImplicitAs.type.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %arr.var: ref %array_type = var arr
// CHECK:STDOUT:   %arr: ref %array_type = bind_name arr, %arr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(constants.%T: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.5)]
// CHECK:STDOUT:   %Self: %As.type.5 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%T) [symbolic = %Convert.type (constants.%Convert.type.12)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.12) = struct_value () [symbolic = %Convert (constants.%Convert.f)]
// CHECK:STDOUT:   %Convert.assoc_type: type = assoc_entity_type @As.%As.type (%As.type.5), @As.%Convert.type (%Convert.type.12) [symbolic = %Convert.assoc_type (constants.%Convert.assoc_type.7)]
// CHECK:STDOUT:   %assoc0: @As.%Convert.assoc_type (%Convert.assoc_type.7) = assoc_entity element0, imports.%import_ref.912b7f.1 [symbolic = %assoc0 (constants.%assoc0.ba1273.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.22
// CHECK:STDOUT:     .Convert = imports.%import_ref.d
// CHECK:STDOUT:     witness = (imports.%import_ref.80c)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.6c
// CHECK:STDOUT:   .Op = imports.%import_ref.16
// CHECK:STDOUT:   witness = (imports.%import_ref.47)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%T: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.d5f)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.29f)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.29f) = struct_value () [symbolic = %Convert (constants.%Convert.9)]
// CHECK:STDOUT:   %Convert.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d), @ImplicitAs.%Convert.type (%Convert.type.29f) [symbolic = %Convert.assoc_type (constants.%Convert.assoc_type.5b)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.5b) = assoc_entity element0, imports.%import_ref.a52db1.1 [symbolic = %assoc0 (constants.%assoc0.ae5e08.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.809
// CHECK:STDOUT:     .Convert = imports.%import_ref.3
// CHECK:STDOUT:     witness = (imports.%import_ref.44)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.5b1455.1 as imports.%import_ref.e7 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: imports.%import_ref.c8538c.1 as imports.%import_ref.b [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.21938e.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: imports.%import_ref.c8538c.2 as imports.%import_ref.9f [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.21938e.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%import_ref.5b1455.2 as imports.%import_ref.593 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%T: type, constants.%Self.3: %As.type.5) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.5)]
// CHECK:STDOUT:   %Self: %As.type.5 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.a5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type (%Self.as_type.a5)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(constants.%T: type, constants.%Self.d5f: %ImplicitAs.type.d) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.d5f)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.2.%Self.as_type (%Self.as_type.8)]() -> @Convert.2.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self.param_patt: Core.IntLiteral]() -> %i32.builtin = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(constants.%Self.d5d: %Add.type) [from "core.carbon"] {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.d5d)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.a9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type (%Self.as_type.a9)](%other.param_patt: @Op.1.%Self.as_type (%Self.as_type.a9)) -> @Op.1.%Self.as_type (%Self.as_type.a9);
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
// CHECK:STDOUT:   %int_3.loc4_44: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %int_4.loc4_47: Core.IntLiteral = int_value 4 [template = constants.%int_4.0]
// CHECK:STDOUT:   %int_3.loc4_51: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %int_32.loc4_56: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc4_56: init type = call constants.%Int(%int_32.loc4_56) [template = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_56.1: type = value_of_initializer %int.make_type_signed.loc4_56 [template = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_56.2: type = converted %int.make_type_signed.loc4_56, %.loc4_56.1 [template = constants.%i32.builtin]
// CHECK:STDOUT:   %impl.elem0.loc4_53: %Convert.type.9 = impl_witness_access constants.%impl_witness.2ccba5.1, element0 [template = constants.%Convert.2b8aa5.1]
// CHECK:STDOUT:   %Convert.bound.loc4_53: <bound method> = bound_method %int_3.loc4_51, %impl.elem0.loc4_53 [template = constants.%Convert.bound.34e9bc.1]
// CHECK:STDOUT:   %int.convert_checked.loc4_53: init %i32.builtin = call %Convert.bound.loc4_53(%int_3.loc4_51) [template = constants.%int_3.a]
// CHECK:STDOUT:   %.loc4_53.1: %i32.builtin = value_of_initializer %int.convert_checked.loc4_53 [template = constants.%int_3.a]
// CHECK:STDOUT:   %.loc4_53.2: %i32.builtin = converted %int_3.loc4_51, %.loc4_53.1 [template = constants.%int_3.a]
// CHECK:STDOUT:   %int_4.loc4_64: Core.IntLiteral = int_value 4 [template = constants.%int_4.0]
// CHECK:STDOUT:   %int_32.loc4_69: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc4_69: init type = call constants.%Int(%int_32.loc4_69) [template = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_69.1: type = value_of_initializer %int.make_type_signed.loc4_69 [template = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_69.2: type = converted %int.make_type_signed.loc4_69, %.loc4_69.1 [template = constants.%i32.builtin]
// CHECK:STDOUT:   %impl.elem0.loc4_66: %Convert.type.9 = impl_witness_access constants.%impl_witness.2ccba5.1, element0 [template = constants.%Convert.2b8aa5.1]
// CHECK:STDOUT:   %Convert.bound.loc4_66: <bound method> = bound_method %int_4.loc4_64, %impl.elem0.loc4_66 [template = constants.%Convert.bound.87a6cb.1]
// CHECK:STDOUT:   %int.convert_checked.loc4_66: init %i32.builtin = call %Convert.bound.loc4_66(%int_4.loc4_64) [template = constants.%int_4.4]
// CHECK:STDOUT:   %.loc4_66.1: %i32.builtin = value_of_initializer %int.convert_checked.loc4_66 [template = constants.%int_4.4]
// CHECK:STDOUT:   %.loc4_66.2: %i32.builtin = converted %int_4.loc4_64, %.loc4_66.1 [template = constants.%int_4.4]
// CHECK:STDOUT:   %impl.elem0.loc4_61: %Op.type.dd8875.1 = impl_witness_access constants.%impl_witness.5, element0 [template = constants.%Op.9f2095.2]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc4_53.2, %impl.elem0.loc4_61 [template = constants.%Op.bound.7]
// CHECK:STDOUT:   %int.sadd: init %i32.builtin = call %Op.bound(%.loc4_53.2, %.loc4_66.2) [template = constants.%int_7]
// CHECK:STDOUT:   %.loc4_73.1: %tuple.type = tuple_literal (%int_3.loc4_44, %int_4.loc4_47, %int.sadd)
// CHECK:STDOUT:   %impl.elem0.loc4_73.1: %Convert.type.13 = impl_witness_access constants.%impl_witness.2ccba5.2, element0 [template = constants.%Convert.2b8aa5.3]
// CHECK:STDOUT:   %Convert.bound.loc4_73.1: <bound method> = bound_method %int_3.loc4_44, %impl.elem0.loc4_73.1 [template = constants.%Convert.bound.34e9bc.2]
// CHECK:STDOUT:   %int.convert_checked.loc4_73.1: init %i32.builtin = call %Convert.bound.loc4_73.1(%int_3.loc4_44) [template = constants.%int_3.a]
// CHECK:STDOUT:   %.loc4_73.2: init %i32.builtin = converted %int_3.loc4_44, %int.convert_checked.loc4_73.1 [template = constants.%int_3.a]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc4_73.3: ref %i32.builtin = array_index file.%arr.var, %int_0
// CHECK:STDOUT:   %.loc4_73.4: init %i32.builtin = initialize_from %.loc4_73.2 to %.loc4_73.3 [template = constants.%int_3.a]
// CHECK:STDOUT:   %impl.elem0.loc4_73.2: %Convert.type.13 = impl_witness_access constants.%impl_witness.2ccba5.2, element0 [template = constants.%Convert.2b8aa5.3]
// CHECK:STDOUT:   %Convert.bound.loc4_73.2: <bound method> = bound_method %int_4.loc4_47, %impl.elem0.loc4_73.2 [template = constants.%Convert.bound.87a6cb.2]
// CHECK:STDOUT:   %int.convert_checked.loc4_73.2: init %i32.builtin = call %Convert.bound.loc4_73.2(%int_4.loc4_47) [template = constants.%int_4.4]
// CHECK:STDOUT:   %.loc4_73.5: init %i32.builtin = converted %int_4.loc4_47, %int.convert_checked.loc4_73.2 [template = constants.%int_4.4]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5]
// CHECK:STDOUT:   %.loc4_73.6: ref %i32.builtin = array_index file.%arr.var, %int_1
// CHECK:STDOUT:   %.loc4_73.7: init %i32.builtin = initialize_from %.loc4_73.5 to %.loc4_73.6 [template = constants.%int_4.4]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.e]
// CHECK:STDOUT:   %.loc4_73.8: ref %i32.builtin = array_index file.%arr.var, %int_2
// CHECK:STDOUT:   %.loc4_73.9: init %i32.builtin = initialize_from %int.sadd to %.loc4_73.8 [template = constants.%int_7]
// CHECK:STDOUT:   %.loc4_73.10: init %array_type = array_init (%.loc4_73.4, %.loc4_73.7, %.loc4_73.9) to file.%arr.var [template = constants.%array]
// CHECK:STDOUT:   %.loc4_74: init %array_type = converted %.loc4_73.1, %.loc4_73.10 [template = constants.%array]
// CHECK:STDOUT:   assign file.%arr.var, %.loc4_74
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
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self.3) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.a5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %T.patt => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.0
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.9
// CHECK:STDOUT:   %Convert => constants.%Convert.29
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.93
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
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.e
// CHECK:STDOUT:   %Self => constants.%Self.d5f
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.13
// CHECK:STDOUT:   %Convert => constants.%Convert.4
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.5f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%T, constants.%Self.d5f) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d
// CHECK:STDOUT:   %Self => constants.%Self.d5f
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %T.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.d5f
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.c
// CHECK:STDOUT:   %Convert => constants.%Convert.b
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.d5d) {
// CHECK:STDOUT:   %Self => constants.%Self.d5d
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
