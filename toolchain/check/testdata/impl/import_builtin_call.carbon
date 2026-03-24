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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/import_builtin_call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/import_builtin_call.carbon

// --- generic_impl.carbon

library "[[@TEST_NAME]]";

interface Add {
  fn Op[self: Self](other: Self) -> Self;
}

fn IntLiteral() -> type = "int_literal.make_type";
fn Int(n: IntLiteral()) -> type = "int.make_type_signed";

class MyInt(N:! IntLiteral()) {
  adapt Int(N);
}

impl forall [N:! IntLiteral()] MyInt(N) as Add {
  fn Op[self: Self](other: Self) -> Self = "int.sadd";
}

fn Double[N:! IntLiteral()](x: MyInt(N)) -> MyInt(N) {
  return x.(Add.Op)(x);
}

// --- use_generic_impl.carbon

library "[[@TEST_NAME]]";

import library "generic_impl";

fn CallImportedDouble(n: MyInt(64)) -> MyInt(64) {
  return Double(n);
}

// --- convert_symbolic.carbon

library "[[@TEST_NAME]]";

fn IntLiteral() -> type = "int_literal.make_type";
fn Int(n: IntLiteral()) -> type = "int.make_type_signed";
fn ToLiteral(n: Int(32)) -> IntLiteral() = "int.convert_checked";
fn FromLiteral(n: IntLiteral()) -> Int(32) = "int.convert_checked";

fn Make(N:! Int(32)) -> Int(ToLiteral(N)) { return Make(N); }

class OtherInt {
  adapt Int(32);
  fn ToLiteral[self: Self]() -> IntLiteral();
};

fn OtherInt.ToLiteral[self: Self]() -> IntLiteral() = "int.convert_checked";

fn MakeFromClass(N:! OtherInt) -> Int(N.ToLiteral()) { return MakeFromClass(N); }

// --- use_convert_symbolic.carbon

library "[[@TEST_NAME]]";

import library "convert_symbolic";

var m: Int(64) = Make(FromLiteral(64));
var n: Int(64) = MakeFromClass(FromLiteral(64) as OtherInt);

// CHECK:STDOUT: --- generic_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [concrete]
// CHECK:STDOUT:   %Self: %Add.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.f75: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %.e62: Core.Form = init_form %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %Add.WithSelf.Op.type.f48: type = fn_type @Add.WithSelf.Op, @Add.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %Add.WithSelf.Op.060: %Add.WithSelf.Op.type.f48 = struct_value () [symbolic]
// CHECK:STDOUT:   %Add.assoc_type: type = assoc_entity_type @Add [concrete]
// CHECK:STDOUT:   %assoc0: %Add.assoc_type = assoc_entity element0, @Add.WithSelf.%Add.WithSelf.Op.decl [concrete]
// CHECK:STDOUT:   %.805: Core.Form = init_form type [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %MyInt.type: type = generic_class_type @MyInt [concrete]
// CHECK:STDOUT:   %MyInt.generic: %MyInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %MyInt: type = class_type @MyInt, @MyInt(%N) [symbolic]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %require_complete.269: <witness> = require_complete_type %iN.builtin [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %iN.builtin [symbolic]
// CHECK:STDOUT:   %Add.impl_witness: <witness> = impl_witness @MyInt.as.Add.impl.%Add.impl_witness_table, @MyInt.as.Add.impl(%N) [symbolic]
// CHECK:STDOUT:   %pattern_type.1a3: type = pattern_type %MyInt [symbolic]
// CHECK:STDOUT:   %.6e6: Core.Form = init_form %MyInt [symbolic]
// CHECK:STDOUT:   %MyInt.as.Add.impl.Op.type: type = fn_type @MyInt.as.Add.impl.Op, @MyInt.as.Add.impl(%N) [symbolic]
// CHECK:STDOUT:   %MyInt.as.Add.impl.Op: %MyInt.as.Add.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.9fa: <witness> = require_complete_type %MyInt [symbolic]
// CHECK:STDOUT:   %Add.facet.fbb: %Add.type = facet_value %MyInt, (%Add.impl_witness) [symbolic]
// CHECK:STDOUT:   %Add.WithSelf.Op.type.6d2: type = fn_type @Add.WithSelf.Op, @Add.WithSelf(%Add.facet.fbb) [symbolic]
// CHECK:STDOUT:   %Add.WithSelf.Op.935: %Add.WithSelf.Op.type.6d2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Double.type: type = fn_type @Double [concrete]
// CHECK:STDOUT:   %Double: %Double.type = struct_value () [concrete]
// CHECK:STDOUT:   %.c60: require_specific_def_type = require_specific_def @MyInt.as.Add.impl(%N) [symbolic]
// CHECK:STDOUT:   %Add.lookup_impl_witness: <witness> = lookup_impl_witness %MyInt, @Add [symbolic]
// CHECK:STDOUT:   %Add.facet.9ab: %Add.type = facet_value %MyInt, (%Add.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Add.WithSelf.Op.type.3e8: type = fn_type @Add.WithSelf.Op, @Add.WithSelf(%Add.facet.9ab) [symbolic]
// CHECK:STDOUT:   %Add.WithSelf.Op.451: %Add.WithSelf.Op.type.3e8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.d02: type = fn_type_with_self_type %Add.WithSelf.Op.type.3e8, %Add.facet.9ab [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.d02 = impl_witness_access %Add.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Add.WithSelf.Op(%Add.facet.9ab) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .MyInt = %MyInt.decl
// CHECK:STDOUT:     .Double = %Double.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Add.decl: type = interface_decl @Add [concrete = constants.%Add.type] {} {}
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [concrete = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern %return.param_patt, %.loc23_20.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc23_20.1: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %.loc23_20.2: Core.Form = init_form %.loc23_20.1 [concrete = constants.%.805]
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [concrete = constants.%Int] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.dc0 = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.dc0 = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern %return.param_patt, %.loc24_28.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc24_28.1: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %.loc24_28.2: Core.Form = init_form %.loc24_28.1 [concrete = constants.%.805]
// CHECK:STDOUT:     %n.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %.loc24_22.1: type = splice_block %.loc24_22.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc24_22.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc24_22.3: type = converted %IntLiteral.call, %.loc24_22.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: Core.IntLiteral = value_binding n, %n.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyInt.decl: %MyInt.type = class_decl @MyInt [concrete = constants.%MyInt.generic] {
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26_28.1: type = splice_block %.loc26_28.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc26_28.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc26_28.3: type = converted %IntLiteral.call, %.loc26_28.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc26_13.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc26_13.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @MyInt.as.Add.impl [concrete] {
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %MyInt.ref: %MyInt.type = name_ref MyInt, file.%MyInt.decl [concrete = constants.%MyInt.generic]
// CHECK:STDOUT:     %N.ref: Core.IntLiteral = name_ref N, %N.loc30_14.2 [symbolic = %N.loc30_14.1 (constants.%N)]
// CHECK:STDOUT:     %MyInt.loc30_39.2: type = class_type @MyInt, @MyInt(constants.%N) [symbolic = %MyInt.loc30_39.1 (constants.%MyInt)]
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, file.%Add.decl [concrete = constants.%Add.type]
// CHECK:STDOUT:     %.loc30_29.1: type = splice_block %.loc30_29.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc30_29.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc30_29.3: type = converted %IntLiteral.call, %.loc30_29.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc30_14.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc30_14.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Double.decl: %Double.type = fn_decl @Double [concrete = constants.%Double] {
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @Double.%pattern_type (%pattern_type.1a3) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @Double.%pattern_type (%pattern_type.1a3) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @Double.%pattern_type (%pattern_type.1a3) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @Double.%pattern_type (%pattern_type.1a3) = return_slot_pattern %return.param_patt, %MyInt.loc34_52 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %MyInt.ref.loc34_45: %MyInt.type = name_ref MyInt, file.%MyInt.decl [concrete = constants.%MyInt.generic]
// CHECK:STDOUT:     %N.ref.loc34_51: Core.IntLiteral = name_ref N, %N.loc34_11.2 [symbolic = %N.loc34_11.1 (constants.%N)]
// CHECK:STDOUT:     %MyInt.loc34_52: type = class_type @MyInt, @MyInt(constants.%N) [symbolic = %MyInt.loc34_39.1 (constants.%MyInt)]
// CHECK:STDOUT:     %.loc34_52.2: Core.Form = init_form %MyInt.loc34_52 [symbolic = %.loc34_52.1 (constants.%.6e6)]
// CHECK:STDOUT:     %.loc34_26.1: type = splice_block %.loc34_26.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc34_26.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc34_26.3: type = converted %IntLiteral.call, %.loc34_26.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc34_11.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc34_11.1 (constants.%N)]
// CHECK:STDOUT:     %x.param: @Double.%MyInt.loc34_39.1 (%MyInt) = value_param call_param0
// CHECK:STDOUT:     %.loc34_39: type = splice_block %MyInt.loc34_39.2 [symbolic = %MyInt.loc34_39.1 (constants.%MyInt)] {
// CHECK:STDOUT:       %MyInt.ref.loc34_32: %MyInt.type = name_ref MyInt, file.%MyInt.decl [concrete = constants.%MyInt.generic]
// CHECK:STDOUT:       %N.ref.loc34_38: Core.IntLiteral = name_ref N, %N.loc34_11.2 [symbolic = %N.loc34_11.1 (constants.%N)]
// CHECK:STDOUT:       %MyInt.loc34_39.2: type = class_type @MyInt, @MyInt(constants.%N) [symbolic = %MyInt.loc34_39.1 (constants.%MyInt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Double.%MyInt.loc34_39.1 (%MyInt) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @Double.%MyInt.loc34_39.1 (%MyInt) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Double.%MyInt.loc34_39.1 (%MyInt) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT:   %Self: %Add.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Add.WithSelf.decl = interface_with_self_decl @Add [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Add.WithSelf.Op.decl: @Add.WithSelf.%Add.WithSelf.Op.type (%Add.WithSelf.Op.type.f48) = fn_decl @Add.WithSelf.Op [symbolic = @Add.WithSelf.%Add.WithSelf.Op (constants.%Add.WithSelf.Op.060)] {
// CHECK:STDOUT:     %self.param_patt: @Add.WithSelf.Op.%pattern_type (%pattern_type.f75) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @Add.WithSelf.Op.%pattern_type (%pattern_type.f75) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %other.param_patt: @Add.WithSelf.Op.%pattern_type (%pattern_type.f75) = value_param_pattern [concrete]
// CHECK:STDOUT:     %other.patt: @Add.WithSelf.Op.%pattern_type (%pattern_type.f75) = at_binding_pattern other, %other.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @Add.WithSelf.Op.%pattern_type (%pattern_type.f75) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @Add.WithSelf.Op.%pattern_type (%pattern_type.f75) = return_slot_pattern %return.param_patt, %.loc20_37.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc20_37: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc20_37: type = facet_access_type %Self.ref.loc20_37 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %.loc20_37.2: type = converted %Self.ref.loc20_37, %Self.as_type.loc20_37 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %.loc20_37.3: Core.Form = init_form %.loc20_37.2 [symbolic = %.loc20_37.1 (constants.%.e62)]
// CHECK:STDOUT:     %self.param: @Add.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc20_15.1: type = splice_block %.loc20_15.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc20_15: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc20_15: type = facet_access_type %Self.ref.loc20_15 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc20_15.2: type = converted %Self.ref.loc20_15, %Self.as_type.loc20_15 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Add.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:     %other.param: @Add.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param1
// CHECK:STDOUT:     %.loc20_28.1: type = splice_block %.loc20_28.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc20_28: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc20_28: type = facet_access_type %Self.ref.loc20_28 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc20_28.2: type = converted %Self.ref.loc20_28, %Self.as_type.loc20_28 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @Add.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type) = value_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref @Add.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Add.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Add.assoc_type = assoc_entity element0, %Add.WithSelf.Op.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = @Add.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@Add.WithSelf.%Add.WithSelf.Op.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @MyInt.as.Add.impl(%N.loc30_14.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc30_14.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc30_14.1 (constants.%N)]
// CHECK:STDOUT:   %MyInt.loc30_39.1: type = class_type @MyInt, @MyInt(%N.loc30_14.1) [symbolic = %MyInt.loc30_39.1 (constants.%MyInt)]
// CHECK:STDOUT:   %Add.impl_witness.loc30_48.2: <witness> = impl_witness %Add.impl_witness_table, @MyInt.as.Add.impl(%N.loc30_14.1) [symbolic = %Add.impl_witness.loc30_48.2 (constants.%Add.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %MyInt.as.Add.impl.Op.type: type = fn_type @MyInt.as.Add.impl.Op, @MyInt.as.Add.impl(%N.loc30_14.1) [symbolic = %MyInt.as.Add.impl.Op.type (constants.%MyInt.as.Add.impl.Op.type)]
// CHECK:STDOUT:   %MyInt.as.Add.impl.Op: @MyInt.as.Add.impl.%MyInt.as.Add.impl.Op.type (%MyInt.as.Add.impl.Op.type) = struct_value () [symbolic = %MyInt.as.Add.impl.Op (constants.%MyInt.as.Add.impl.Op)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %MyInt.loc30_39.1 [symbolic = %require_complete (constants.%require_complete.9fa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %MyInt.loc30_39.2 as %Add.ref {
// CHECK:STDOUT:     %MyInt.as.Add.impl.Op.decl: @MyInt.as.Add.impl.%MyInt.as.Add.impl.Op.type (%MyInt.as.Add.impl.Op.type) = fn_decl @MyInt.as.Add.impl.Op [symbolic = @MyInt.as.Add.impl.%MyInt.as.Add.impl.Op (constants.%MyInt.as.Add.impl.Op)] {
// CHECK:STDOUT:       %self.param_patt: @MyInt.as.Add.impl.Op.%pattern_type (%pattern_type.1a3) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @MyInt.as.Add.impl.Op.%pattern_type (%pattern_type.1a3) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %other.param_patt: @MyInt.as.Add.impl.Op.%pattern_type (%pattern_type.1a3) = value_param_pattern [concrete]
// CHECK:STDOUT:       %other.patt: @MyInt.as.Add.impl.Op.%pattern_type (%pattern_type.1a3) = at_binding_pattern other, %other.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @MyInt.as.Add.impl.Op.%pattern_type (%pattern_type.1a3) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @MyInt.as.Add.impl.Op.%pattern_type (%pattern_type.1a3) = return_slot_pattern %return.param_patt, %Self.ref.loc31_37 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref.loc31_37: type = name_ref Self, @MyInt.as.Add.impl.%MyInt.loc30_39.2 [symbolic = %MyInt (constants.%MyInt)]
// CHECK:STDOUT:       %.loc31_37.2: Core.Form = init_form %Self.ref.loc31_37 [symbolic = %.loc31_37.1 (constants.%.6e6)]
// CHECK:STDOUT:       %self.param: @MyInt.as.Add.impl.Op.%MyInt (%MyInt) = value_param call_param0
// CHECK:STDOUT:       %Self.ref.loc31_15: type = name_ref Self, @MyInt.as.Add.impl.%MyInt.loc30_39.2 [symbolic = %MyInt (constants.%MyInt)]
// CHECK:STDOUT:       %self: @MyInt.as.Add.impl.Op.%MyInt (%MyInt) = value_binding self, %self.param
// CHECK:STDOUT:       %other.param: @MyInt.as.Add.impl.Op.%MyInt (%MyInt) = value_param call_param1
// CHECK:STDOUT:       %Self.ref.loc31_28: type = name_ref Self, @MyInt.as.Add.impl.%MyInt.loc30_39.2 [symbolic = %MyInt (constants.%MyInt)]
// CHECK:STDOUT:       %other: @MyInt.as.Add.impl.Op.%MyInt (%MyInt) = value_binding other, %other.param
// CHECK:STDOUT:       %return.param: ref @MyInt.as.Add.impl.Op.%MyInt (%MyInt) = out_param call_param2
// CHECK:STDOUT:       %return: ref @MyInt.as.Add.impl.Op.%MyInt (%MyInt) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Add.impl_witness_table = impl_witness_table (%MyInt.as.Add.impl.Op.decl), @MyInt.as.Add.impl [concrete]
// CHECK:STDOUT:     %Add.impl_witness.loc30_48.1: <witness> = impl_witness %Add.impl_witness_table, @MyInt.as.Add.impl(constants.%N) [symbolic = %Add.impl_witness.loc30_48.2 (constants.%Add.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %MyInt.as.Add.impl.Op.decl
// CHECK:STDOUT:     witness = %Add.impl_witness.loc30_48.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @MyInt(%N.loc26_13.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc26_13.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc26_13.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %N.loc26_13.1 [symbolic = %iN.builtin (constants.%iN.builtin)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %iN.builtin [symbolic = %require_complete (constants.%require_complete.269)]
// CHECK:STDOUT:   %complete_type.loc28_1.2: <witness> = complete_type_witness %iN.builtin [symbolic = %complete_type.loc28_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Int.ref: %Int.type = name_ref Int, file.%Int.decl [concrete = constants.%Int]
// CHECK:STDOUT:     %N.ref: Core.IntLiteral = name_ref N, %N.loc26_13.2 [symbolic = %N.loc26_13.1 (constants.%N)]
// CHECK:STDOUT:     %Int.call: init type = call %Int.ref(%N.ref) [symbolic = %iN.builtin (constants.%iN.builtin)]
// CHECK:STDOUT:     %.loc27_15.1: type = value_of_initializer %Int.call [symbolic = %iN.builtin (constants.%iN.builtin)]
// CHECK:STDOUT:     %.loc27_15.2: type = converted %Int.call, %.loc27_15.1 [symbolic = %iN.builtin (constants.%iN.builtin)]
// CHECK:STDOUT:     adapt_decl %.loc27_15.2 [concrete]
// CHECK:STDOUT:     %complete_type.loc28_1.1: <witness> = complete_type_witness constants.%iN.builtin [symbolic = %complete_type.loc28_1.2 (constants.%complete_type)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc28_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%MyInt
// CHECK:STDOUT:     .Int = <poisoned>
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Add.WithSelf.Op(@Add.%Self: %Add.type) {
// CHECK:STDOUT:   %Self: %Add.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.f75)]
// CHECK:STDOUT:   %.loc20_37.1: Core.Form = init_form %Self.binding.as_type [symbolic = %.loc20_37.1 (constants.%.e62)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Add.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type), %other.param: @Add.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type)) -> out %return.param: @Add.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> out %return.param: type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param: Core.IntLiteral) -> out %return.param: type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MyInt.as.Add.impl.Op(@MyInt.as.Add.impl.%N.loc30_14.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %MyInt: type = class_type @MyInt, @MyInt(%N) [symbolic = %MyInt (constants.%MyInt)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %MyInt [symbolic = %pattern_type (constants.%pattern_type.1a3)]
// CHECK:STDOUT:   %.loc31_37.1: Core.Form = init_form %MyInt [symbolic = %.loc31_37.1 (constants.%.6e6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @MyInt.as.Add.impl.Op.%MyInt (%MyInt), %other.param: @MyInt.as.Add.impl.Op.%MyInt (%MyInt)) -> out %return.param: @MyInt.as.Add.impl.Op.%MyInt (%MyInt) = "int.sadd";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Double(%N.loc34_11.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc34_11.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc34_11.1 (constants.%N)]
// CHECK:STDOUT:   %MyInt.loc34_39.1: type = class_type @MyInt, @MyInt(%N.loc34_11.1) [symbolic = %MyInt.loc34_39.1 (constants.%MyInt)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %MyInt.loc34_39.1 [symbolic = %pattern_type (constants.%pattern_type.1a3)]
// CHECK:STDOUT:   %.loc34_52.1: Core.Form = init_form %MyInt.loc34_39.1 [symbolic = %.loc34_52.1 (constants.%.6e6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %MyInt.loc34_39.1 [symbolic = %require_complete (constants.%require_complete.9fa)]
// CHECK:STDOUT:   %.loc35_11.1: require_specific_def_type = require_specific_def @MyInt.as.Add.impl(%N.loc34_11.1) [symbolic = %.loc35_11.1 (constants.%.c60)]
// CHECK:STDOUT:   %Add.lookup_impl_witness: <witness> = lookup_impl_witness %MyInt.loc34_39.1, @Add [symbolic = %Add.lookup_impl_witness (constants.%Add.lookup_impl_witness)]
// CHECK:STDOUT:   %Add.facet.loc35_11: %Add.type = facet_value %MyInt.loc34_39.1, (%Add.lookup_impl_witness) [symbolic = %Add.facet.loc35_11 (constants.%Add.facet.9ab)]
// CHECK:STDOUT:   %Add.WithSelf.Op.type: type = fn_type @Add.WithSelf.Op, @Add.WithSelf(%Add.facet.loc35_11) [symbolic = %Add.WithSelf.Op.type (constants.%Add.WithSelf.Op.type.3e8)]
// CHECK:STDOUT:   %.loc35_11.2: type = fn_type_with_self_type %Add.WithSelf.Op.type, %Add.facet.loc35_11 [symbolic = %.loc35_11.2 (constants.%.d02)]
// CHECK:STDOUT:   %impl.elem0.loc35_11.2: @Double.%.loc35_11.2 (%.d02) = impl_witness_access %Add.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc35_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_11.2: <specific function> = specific_impl_function %impl.elem0.loc35_11.2, @Add.WithSelf.Op(%Add.facet.loc35_11) [symbolic = %specific_impl_fn.loc35_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Double.%MyInt.loc34_39.1 (%MyInt)) -> out %return.param: @Double.%MyInt.loc34_39.1 (%MyInt) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc35_10: @Double.%MyInt.loc34_39.1 (%MyInt) = name_ref x, %x
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, file.%Add.decl [concrete = constants.%Add.type]
// CHECK:STDOUT:     %Op.ref: %Add.assoc_type = name_ref Op, @Add.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc35_11.1: @Double.%.loc35_11.2 (%.d02) = impl_witness_access constants.%Add.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc35_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc35_11: <bound method> = bound_method %x.ref.loc35_10, %impl.elem0.loc35_11.1
// CHECK:STDOUT:     %x.ref.loc35_21: @Double.%MyInt.loc34_39.1 (%MyInt) = name_ref x, %x
// CHECK:STDOUT:     %Add.facet.loc35_22: %Add.type = facet_value constants.%MyInt, (constants.%Add.lookup_impl_witness) [symbolic = %Add.facet.loc35_11 (constants.%Add.facet.9ab)]
// CHECK:STDOUT:     %.loc35_22: %Add.type = converted constants.%MyInt, %Add.facet.loc35_22 [symbolic = %Add.facet.loc35_11 (constants.%Add.facet.9ab)]
// CHECK:STDOUT:     %specific_impl_fn.loc35_11.1: <specific function> = specific_impl_function %impl.elem0.loc35_11.1, @Add.WithSelf.Op(constants.%Add.facet.9ab) [symbolic = %specific_impl_fn.loc35_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc35_22: <bound method> = bound_method %x.ref.loc35_10, %specific_impl_fn.loc35_11.1
// CHECK:STDOUT:     %Add.WithSelf.Op.call: init @Double.%MyInt.loc34_39.1 (%MyInt) = call %bound_method.loc35_22(%x.ref.loc35_10, %x.ref.loc35_21)
// CHECK:STDOUT:     return %Add.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Add.WithSelf.Op.type => constants.%Add.WithSelf.Op.type.f48
// CHECK:STDOUT:   %Add.WithSelf.Op => constants.%Add.WithSelf.Op.060
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.WithSelf.Op(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f75
// CHECK:STDOUT:   %.loc20_37.1 => constants.%.e62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MyInt(constants.%N) {
// CHECK:STDOUT:   %N.loc26_13.1 => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %iN.builtin => constants.%iN.builtin
// CHECK:STDOUT:   %require_complete => constants.%require_complete.269
// CHECK:STDOUT:   %complete_type.loc28_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MyInt.as.Add.impl(constants.%N) {
// CHECK:STDOUT:   %N.loc30_14.1 => constants.%N
// CHECK:STDOUT:   %MyInt.loc30_39.1 => constants.%MyInt
// CHECK:STDOUT:   %Add.impl_witness.loc30_48.2 => constants.%Add.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %MyInt.as.Add.impl.Op.type => constants.%MyInt.as.Add.impl.Op.type
// CHECK:STDOUT:   %MyInt.as.Add.impl.Op => constants.%MyInt.as.Add.impl.Op
// CHECK:STDOUT:   %require_complete => constants.%require_complete.9fa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MyInt.as.Add.impl.Op(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %MyInt => constants.%MyInt
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1a3
// CHECK:STDOUT:   %.loc31_37.1 => constants.%.6e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.WithSelf(constants.%Add.facet.fbb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Add.facet.fbb
// CHECK:STDOUT:   %Add.WithSelf.Op.type => constants.%Add.WithSelf.Op.type.6d2
// CHECK:STDOUT:   %Add.WithSelf.Op => constants.%Add.WithSelf.Op.935
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.WithSelf.Op(constants.%Add.facet.fbb) {
// CHECK:STDOUT:   %Self => constants.%Add.facet.fbb
// CHECK:STDOUT:   %Self.binding.as_type => constants.%MyInt
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1a3
// CHECK:STDOUT:   %.loc20_37.1 => constants.%.6e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Double(constants.%N) {
// CHECK:STDOUT:   %N.loc34_11.1 => constants.%N
// CHECK:STDOUT:   %MyInt.loc34_39.1 => constants.%MyInt
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1a3
// CHECK:STDOUT:   %.loc34_52.1 => constants.%.6e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.WithSelf(constants.%Add.facet.9ab) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Add.facet.9ab
// CHECK:STDOUT:   %Add.WithSelf.Op.type => constants.%Add.WithSelf.Op.type.3e8
// CHECK:STDOUT:   %Add.WithSelf.Op => constants.%Add.WithSelf.Op.451
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.WithSelf.Op(constants.%Add.facet.9ab) {
// CHECK:STDOUT:   %Self => constants.%Add.facet.9ab
// CHECK:STDOUT:   %Self.binding.as_type => constants.%MyInt
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1a3
// CHECK:STDOUT:   %.loc20_37.1 => constants.%.6e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_generic_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MyInt.type: type = generic_class_type @MyInt [concrete]
// CHECK:STDOUT:   %MyInt.generic: %MyInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %complete_type.d26: <witness> = complete_type_witness %iN.builtin [symbolic]
// CHECK:STDOUT:   %MyInt.396: type = class_type @MyInt, @MyInt(%N) [symbolic]
// CHECK:STDOUT:   %require_complete.269: <witness> = require_complete_type %iN.builtin [symbolic]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %MyInt.f9e: type = class_type @MyInt, @MyInt(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.48d: type = pattern_type %MyInt.f9e [concrete]
// CHECK:STDOUT:   %.5d4: Core.Form = init_form %MyInt.f9e [concrete]
// CHECK:STDOUT:   %CallImportedDouble.type: type = fn_type @CallImportedDouble [concrete]
// CHECK:STDOUT:   %CallImportedDouble: %CallImportedDouble.type = struct_value () [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %complete_type.4a1: <witness> = complete_type_witness %i64.builtin [concrete]
// CHECK:STDOUT:   %Double.type: type = fn_type @Double [concrete]
// CHECK:STDOUT:   %Double: %Double.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.1a3: type = pattern_type %MyInt.396 [symbolic]
// CHECK:STDOUT:   %.6e6: Core.Form = init_form %MyInt.396 [symbolic]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [concrete]
// CHECK:STDOUT:   %Self: %Add.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Add.WithSelf.Op.type.099: type = fn_type @Add.WithSelf.Op, @Add.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %Add.WithSelf.Op.99a: %Add.WithSelf.Op.type.099 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.5af: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %.32a: Core.Form = init_form %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %Add.lookup_impl_witness: <witness> = lookup_impl_witness %MyInt.396, @Add [symbolic]
// CHECK:STDOUT:   %Add.facet.9ab: %Add.type = facet_value %MyInt.396, (%Add.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Add.WithSelf.Op.type.3e8: type = fn_type @Add.WithSelf.Op, @Add.WithSelf(%Add.facet.9ab) [symbolic]
// CHECK:STDOUT:   %Add.WithSelf.Op.451: %Add.WithSelf.Op.type.3e8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.d02: type = fn_type_with_self_type %Add.WithSelf.Op.type.3e8, %Add.facet.9ab [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.d02 = impl_witness_access %Add.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Add.WithSelf.Op(%Add.facet.9ab) [symbolic]
// CHECK:STDOUT:   %Add.impl_witness.a35: <witness> = impl_witness imports.%Add.impl_witness_table, @MyInt.as.Add.impl(%N) [symbolic]
// CHECK:STDOUT:   %require_complete.9fa: <witness> = require_complete_type %MyInt.396 [symbolic]
// CHECK:STDOUT:   %MyInt.as.Add.impl.Op.type.ca5: type = fn_type @MyInt.as.Add.impl.Op, @MyInt.as.Add.impl(%N) [symbolic]
// CHECK:STDOUT:   %MyInt.as.Add.impl.Op.d3e: %MyInt.as.Add.impl.Op.type.ca5 = struct_value () [symbolic]
// CHECK:STDOUT:   %.c60: require_specific_def_type = require_specific_def @MyInt.as.Add.impl(%N) [symbolic]
// CHECK:STDOUT:   %Double.specific_fn: <specific function> = specific_function %Double, @Double(%int_64) [concrete]
// CHECK:STDOUT:   %Add.impl_witness.868: <witness> = impl_witness imports.%Add.impl_witness_table, @MyInt.as.Add.impl(%int_64) [concrete]
// CHECK:STDOUT:   %MyInt.as.Add.impl.Op.type.1d4: type = fn_type @MyInt.as.Add.impl.Op, @MyInt.as.Add.impl(%int_64) [concrete]
// CHECK:STDOUT:   %MyInt.as.Add.impl.Op.ab3: %MyInt.as.Add.impl.Op.type.1d4 = struct_value () [concrete]
// CHECK:STDOUT:   %.3de: require_specific_def_type = require_specific_def @MyInt.as.Add.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Add.facet.1f9: %Add.type = facet_value %MyInt.f9e, (%Add.impl_witness.868) [concrete]
// CHECK:STDOUT:   %Add.WithSelf.Op.type.626: type = fn_type @Add.WithSelf.Op, @Add.WithSelf(%Add.facet.1f9) [concrete]
// CHECK:STDOUT:   %.166: type = fn_type_with_self_type %Add.WithSelf.Op.type.626, %Add.facet.1f9 [concrete]
// CHECK:STDOUT:   %MyInt.as.Add.impl.Op.specific_fn: <specific function> = specific_function %MyInt.as.Add.impl.Op.ab3, @MyInt.as.Add.impl.Op(%int_64) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Add = import_ref Main//generic_impl, Add, unloaded
// CHECK:STDOUT:   %Main.IntLiteral = import_ref Main//generic_impl, IntLiteral, unloaded
// CHECK:STDOUT:   %Main.Int = import_ref Main//generic_impl, Int, unloaded
// CHECK:STDOUT:   %Main.MyInt: %MyInt.type = import_ref Main//generic_impl, MyInt, loaded [concrete = constants.%MyInt.generic]
// CHECK:STDOUT:   %Main.Double: %Double.type = import_ref Main//generic_impl, Double, loaded [concrete = constants.%Double]
// CHECK:STDOUT:   %Main.import_ref.f9e: <witness> = import_ref Main//generic_impl, loc28_1, loaded [symbolic = @MyInt.%complete_type (constants.%complete_type.d26)]
// CHECK:STDOUT:   %Main.import_ref.1ae = import_ref Main//generic_impl, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.6b552a.1: Core.IntLiteral = import_ref Main//generic_impl, loc26_13, loaded [symbolic = @MyInt.%N (constants.%N)]
// CHECK:STDOUT:   %Main.import_ref.6b552a.2: Core.IntLiteral = import_ref Main//generic_impl, loc34_11, loaded [symbolic = @Double.%N (constants.%N)]
// CHECK:STDOUT:   %Main.import_ref.9ec = import_ref Main//generic_impl, loc20_41, unloaded
// CHECK:STDOUT:   %Main.Op = import_ref Main//generic_impl, Op, unloaded
// CHECK:STDOUT:   %Main.import_ref.bb084d.2: %Add.type = import_ref Main//generic_impl, loc19_15, loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %Main.import_ref.17c = import_ref Main//generic_impl, loc19_15, unloaded
// CHECK:STDOUT:   %Main.import_ref.9c7: <witness> = import_ref Main//generic_impl, loc30_48, loaded [symbolic = @MyInt.as.Add.impl.%Add.impl_witness (constants.%Add.impl_witness.a35)]
// CHECK:STDOUT:   %Main.import_ref.078: @MyInt.as.Add.impl.%MyInt.as.Add.impl.Op.type (%MyInt.as.Add.impl.Op.type.ca5) = import_ref Main//generic_impl, loc31_42, loaded [symbolic = @MyInt.as.Add.impl.%MyInt.as.Add.impl.Op (constants.%MyInt.as.Add.impl.Op.d3e)]
// CHECK:STDOUT:   %Add.impl_witness_table = impl_witness_table (%Main.import_ref.078), @MyInt.as.Add.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.6b552a.3: Core.IntLiteral = import_ref Main//generic_impl, loc30_14, loaded [symbolic = @MyInt.as.Add.impl.%N (constants.%N)]
// CHECK:STDOUT:   %Main.import_ref.bbe59c.2: type = import_ref Main//generic_impl, loc30_39, loaded [symbolic = @MyInt.as.Add.impl.%MyInt (constants.%MyInt.396)]
// CHECK:STDOUT:   %Main.import_ref.1d5: type = import_ref Main//generic_impl, loc30_44, loaded [concrete = constants.%Add.type]
// CHECK:STDOUT:   %Main.import_ref.6b552a.4: Core.IntLiteral = import_ref Main//generic_impl, loc30_14, loaded [symbolic = @MyInt.as.Add.impl.%N (constants.%N)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Add = imports.%Main.Add
// CHECK:STDOUT:     .IntLiteral = imports.%Main.IntLiteral
// CHECK:STDOUT:     .Int = imports.%Main.Int
// CHECK:STDOUT:     .MyInt = imports.%Main.MyInt
// CHECK:STDOUT:     .Double = imports.%Main.Double
// CHECK:STDOUT:     .CallImportedDouble = %CallImportedDouble.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %CallImportedDouble.decl: %CallImportedDouble.type = fn_decl @CallImportedDouble [concrete = constants.%CallImportedDouble] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.48d = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.48d = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.48d = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.48d = return_slot_pattern %return.param_patt, %MyInt.loc44_48 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %MyInt.ref.loc44_40: %MyInt.type = name_ref MyInt, imports.%Main.MyInt [concrete = constants.%MyInt.generic]
// CHECK:STDOUT:     %int_64.loc44_46: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %MyInt.loc44_48: type = class_type @MyInt, @MyInt(constants.%int_64) [concrete = constants.%MyInt.f9e]
// CHECK:STDOUT:     %.loc44_48: Core.Form = init_form %MyInt.loc44_48 [concrete = constants.%.5d4]
// CHECK:STDOUT:     %n.param: %MyInt.f9e = value_param call_param0
// CHECK:STDOUT:     %.loc44_34: type = splice_block %MyInt.loc44_34 [concrete = constants.%MyInt.f9e] {
// CHECK:STDOUT:       %MyInt.ref.loc44_26: %MyInt.type = name_ref MyInt, imports.%Main.MyInt [concrete = constants.%MyInt.generic]
// CHECK:STDOUT:       %int_64.loc44_32: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %MyInt.loc44_34: type = class_type @MyInt, @MyInt(constants.%int_64) [concrete = constants.%MyInt.f9e]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %MyInt.f9e = value_binding n, %n.param
// CHECK:STDOUT:     %return.param: ref %MyInt.f9e = out_param call_param1
// CHECK:STDOUT:     %return: ref %MyInt.f9e = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add [from "generic_impl.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.17c
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.9ec
// CHECK:STDOUT:   witness = (imports.%Main.Op)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @MyInt.as.Add.impl(imports.%Main.import_ref.6b552a.4: Core.IntLiteral) [from "generic_impl.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %MyInt: type = class_type @MyInt, @MyInt(%N) [symbolic = %MyInt (constants.%MyInt.396)]
// CHECK:STDOUT:   %Add.impl_witness: <witness> = impl_witness imports.%Add.impl_witness_table, @MyInt.as.Add.impl(%N) [symbolic = %Add.impl_witness (constants.%Add.impl_witness.a35)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %MyInt.as.Add.impl.Op.type: type = fn_type @MyInt.as.Add.impl.Op, @MyInt.as.Add.impl(%N) [symbolic = %MyInt.as.Add.impl.Op.type (constants.%MyInt.as.Add.impl.Op.type.ca5)]
// CHECK:STDOUT:   %MyInt.as.Add.impl.Op: @MyInt.as.Add.impl.%MyInt.as.Add.impl.Op.type (%MyInt.as.Add.impl.Op.type.ca5) = struct_value () [symbolic = %MyInt.as.Add.impl.Op (constants.%MyInt.as.Add.impl.Op.d3e)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %MyInt [symbolic = %require_complete (constants.%require_complete.9fa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.bbe59c.2 as imports.%Main.import_ref.1d5 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.9c7
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @MyInt(imports.%Main.import_ref.6b552a.1: Core.IntLiteral) [from "generic_impl.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %N [symbolic = %iN.builtin (constants.%iN.builtin)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %iN.builtin [symbolic = %require_complete (constants.%require_complete.269)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %iN.builtin [symbolic = %complete_type (constants.%complete_type.d26)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.f9e
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.1ae
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallImportedDouble(%n.param: %MyInt.f9e) -> out %return.param: %MyInt.f9e {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Double.ref: %Double.type = name_ref Double, imports.%Main.Double [concrete = constants.%Double]
// CHECK:STDOUT:   %n.ref: %MyInt.f9e = name_ref n, %n
// CHECK:STDOUT:   %Double.specific_fn: <specific function> = specific_function %Double.ref, @Double(constants.%int_64) [concrete = constants.%Double.specific_fn]
// CHECK:STDOUT:   %Double.call: init %MyInt.f9e = call %Double.specific_fn(%n.ref)
// CHECK:STDOUT:   return %Double.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Double(imports.%Main.import_ref.6b552a.2: Core.IntLiteral) [from "generic_impl.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %MyInt: type = class_type @MyInt, @MyInt(%N) [symbolic = %MyInt (constants.%MyInt.396)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %MyInt [symbolic = %pattern_type (constants.%pattern_type.1a3)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %MyInt [symbolic = %.1 (constants.%.6e6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %MyInt [symbolic = %require_complete (constants.%require_complete.9fa)]
// CHECK:STDOUT:   %.2: require_specific_def_type = require_specific_def @MyInt.as.Add.impl(%N) [symbolic = %.2 (constants.%.c60)]
// CHECK:STDOUT:   %Add.lookup_impl_witness: <witness> = lookup_impl_witness %MyInt, @Add [symbolic = %Add.lookup_impl_witness (constants.%Add.lookup_impl_witness)]
// CHECK:STDOUT:   %Add.facet: %Add.type = facet_value %MyInt, (%Add.lookup_impl_witness) [symbolic = %Add.facet (constants.%Add.facet.9ab)]
// CHECK:STDOUT:   %Add.WithSelf.Op.type: type = fn_type @Add.WithSelf.Op, @Add.WithSelf(%Add.facet) [symbolic = %Add.WithSelf.Op.type (constants.%Add.WithSelf.Op.type.3e8)]
// CHECK:STDOUT:   %.3: type = fn_type_with_self_type %Add.WithSelf.Op.type, %Add.facet [symbolic = %.3 (constants.%.d02)]
// CHECK:STDOUT:   %impl.elem0: @Double.%.3 (%.d02) = impl_witness_access %Add.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Add.WithSelf.Op(%Add.facet) [symbolic = %specific_impl_fn (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Add.WithSelf.Op(imports.%Main.import_ref.bb084d.2: %Add.type) [from "generic_impl.carbon"] {
// CHECK:STDOUT:   %Self: %Add.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.5af)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %Self.binding.as_type [symbolic = %.1 (constants.%.32a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MyInt.as.Add.impl.Op(imports.%Main.import_ref.6b552a.3: Core.IntLiteral) [from "generic_impl.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %MyInt: type = class_type @MyInt, @MyInt(%N) [symbolic = %MyInt (constants.%MyInt.396)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %MyInt [symbolic = %pattern_type (constants.%pattern_type.1a3)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %MyInt [symbolic = %.1 (constants.%.6e6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "int.sadd";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MyInt(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %iN.builtin => constants.%iN.builtin
// CHECK:STDOUT:   %require_complete => constants.%require_complete.269
// CHECK:STDOUT:   %complete_type => constants.%complete_type.d26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MyInt(constants.%int_64) {
// CHECK:STDOUT:   %N => constants.%int_64
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %iN.builtin => constants.%i64.builtin
// CHECK:STDOUT:   %require_complete => constants.%complete_type.4a1
// CHECK:STDOUT:   %complete_type => constants.%complete_type.4a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Add.WithSelf.Op.type => constants.%Add.WithSelf.Op.type.099
// CHECK:STDOUT:   %Add.WithSelf.Op => constants.%Add.WithSelf.Op.99a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.WithSelf.Op(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.5af
// CHECK:STDOUT:   %.1 => constants.%.32a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.WithSelf(constants.%Add.facet.9ab) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Add.facet.9ab
// CHECK:STDOUT:   %Add.WithSelf.Op.type => constants.%Add.WithSelf.Op.type.3e8
// CHECK:STDOUT:   %Add.WithSelf.Op => constants.%Add.WithSelf.Op.451
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.WithSelf.Op(constants.%Add.facet.9ab) {
// CHECK:STDOUT:   %Self => constants.%Add.facet.9ab
// CHECK:STDOUT:   %Self.binding.as_type => constants.%MyInt.396
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1a3
// CHECK:STDOUT:   %.1 => constants.%.6e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MyInt.as.Add.impl(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %MyInt => constants.%MyInt.396
// CHECK:STDOUT:   %Add.impl_witness => constants.%Add.impl_witness.a35
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %MyInt.as.Add.impl.Op.type => constants.%MyInt.as.Add.impl.Op.type.ca5
// CHECK:STDOUT:   %MyInt.as.Add.impl.Op => constants.%MyInt.as.Add.impl.Op.d3e
// CHECK:STDOUT:   %require_complete => constants.%require_complete.9fa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MyInt.as.Add.impl.Op(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %MyInt => constants.%MyInt.396
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1a3
// CHECK:STDOUT:   %.1 => constants.%.6e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Double(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %MyInt => constants.%MyInt.396
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1a3
// CHECK:STDOUT:   %.1 => constants.%.6e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Double(constants.%int_64) {
// CHECK:STDOUT:   %N => constants.%int_64
// CHECK:STDOUT:   %MyInt => constants.%MyInt.f9e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.48d
// CHECK:STDOUT:   %.1 => constants.%.5d4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.4a1
// CHECK:STDOUT:   %.2 => constants.%.3de
// CHECK:STDOUT:   %Add.lookup_impl_witness => constants.%Add.impl_witness.868
// CHECK:STDOUT:   %Add.facet => constants.%Add.facet.1f9
// CHECK:STDOUT:   %Add.WithSelf.Op.type => constants.%Add.WithSelf.Op.type.626
// CHECK:STDOUT:   %.3 => constants.%.166
// CHECK:STDOUT:   %impl.elem0 => constants.%MyInt.as.Add.impl.Op.ab3
// CHECK:STDOUT:   %specific_impl_fn => constants.%MyInt.as.Add.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MyInt.as.Add.impl(constants.%int_64) {
// CHECK:STDOUT:   %N => constants.%int_64
// CHECK:STDOUT:   %MyInt => constants.%MyInt.f9e
// CHECK:STDOUT:   %Add.impl_witness => constants.%Add.impl_witness.868
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %MyInt.as.Add.impl.Op.type => constants.%MyInt.as.Add.impl.Op.type.1d4
// CHECK:STDOUT:   %MyInt.as.Add.impl.Op => constants.%MyInt.as.Add.impl.Op.ab3
// CHECK:STDOUT:   %require_complete => constants.%complete_type.4a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.WithSelf(constants.%Add.facet.1f9) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.WithSelf.Op(constants.%Add.facet.1f9) {
// CHECK:STDOUT:   %Self => constants.%Add.facet.1f9
// CHECK:STDOUT:   %Self.binding.as_type => constants.%MyInt.f9e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.48d
// CHECK:STDOUT:   %.1 => constants.%.5d4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MyInt.as.Add.impl.Op(constants.%int_64) {
// CHECK:STDOUT:   %N => constants.%int_64
// CHECK:STDOUT:   %MyInt => constants.%MyInt.f9e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.48d
// CHECK:STDOUT:   %.1 => constants.%.5d4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- convert_symbolic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.805: Core.Form = init_form type [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.956: type = pattern_type %i32.builtin [concrete]
// CHECK:STDOUT:   %.f7d: Core.Form = init_form Core.IntLiteral [concrete]
// CHECK:STDOUT:   %ToLiteral.type: type = fn_type @ToLiteral [concrete]
// CHECK:STDOUT:   %ToLiteral: %ToLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %.9cb: Core.Form = init_form %i32.builtin [concrete]
// CHECK:STDOUT:   %FromLiteral.type: type = fn_type @FromLiteral [concrete]
// CHECK:STDOUT:   %FromLiteral: %FromLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %N.7e8: %i32.builtin = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %ToLiteral.call: init Core.IntLiteral = call %ToLiteral(%N.7e8) [symbolic]
// CHECK:STDOUT:   %iN.builtin.b5d: type = int_type signed, %ToLiteral.call [symbolic]
// CHECK:STDOUT:   %.e4e: Core.Form = init_form %iN.builtin.b5d [symbolic]
// CHECK:STDOUT:   %pattern_type.073: type = pattern_type %iN.builtin.b5d [symbolic]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.8ad: <witness> = require_complete_type %iN.builtin.b5d [symbolic]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make, @Make(%N.7e8) [symbolic]
// CHECK:STDOUT:   %OtherInt: type = class_type @OtherInt [concrete]
// CHECK:STDOUT:   %pattern_type.08d: type = pattern_type %OtherInt [concrete]
// CHECK:STDOUT:   %OtherInt.ToLiteral.type: type = fn_type @OtherInt.ToLiteral [concrete]
// CHECK:STDOUT:   %OtherInt.ToLiteral: %OtherInt.ToLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %N.0f7: %OtherInt = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %OtherInt.ToLiteral.bound: <bound method> = bound_method %N.0f7, %OtherInt.ToLiteral [symbolic]
// CHECK:STDOUT:   %OtherInt.ToLiteral.call: init Core.IntLiteral = call %OtherInt.ToLiteral.bound(%N.0f7) [symbolic]
// CHECK:STDOUT:   %iN.builtin.63a: type = int_type signed, %OtherInt.ToLiteral.call [symbolic]
// CHECK:STDOUT:   %.4f8: Core.Form = init_form %iN.builtin.63a [symbolic]
// CHECK:STDOUT:   %pattern_type.bd7: type = pattern_type %iN.builtin.63a [symbolic]
// CHECK:STDOUT:   %MakeFromClass.type: type = fn_type @MakeFromClass [concrete]
// CHECK:STDOUT:   %MakeFromClass: %MakeFromClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.a44: <witness> = require_complete_type %iN.builtin.63a [symbolic]
// CHECK:STDOUT:   %MakeFromClass.specific_fn: <specific function> = specific_function %MakeFromClass, @MakeFromClass(%N.0f7) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .ToLiteral = %ToLiteral.decl
// CHECK:STDOUT:     .FromLiteral = %FromLiteral.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .OtherInt = %OtherInt.decl
// CHECK:STDOUT:     .MakeFromClass = %MakeFromClass.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [concrete = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern %return.param_patt, %.loc52_20.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc52_20.1: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %.loc52_20.2: Core.Form = init_form %.loc52_20.1 [concrete = constants.%.805]
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [concrete = constants.%Int] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.dc0 = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.dc0 = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern %return.param_patt, %.loc53_28.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc53_28.1: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %.loc53_28.2: Core.Form = init_form %.loc53_28.1 [concrete = constants.%.805]
// CHECK:STDOUT:     %n.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %.loc53_22.1: type = splice_block %.loc53_22.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc53_22.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc53_22.3: type = converted %IntLiteral.call, %.loc53_22.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: Core.IntLiteral = value_binding n, %n.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ToLiteral.decl: %ToLiteral.type = fn_decl @ToLiteral [concrete = constants.%ToLiteral] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.956 = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.956 = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.dc0 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.dc0 = return_slot_pattern %return.param_patt, %.loc54_40.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc54_40.1: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc54_40.2: type = converted %IntLiteral.call, %.loc54_40.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc54_40.3: Core.Form = init_form %.loc54_40.2 [concrete = constants.%.f7d]
// CHECK:STDOUT:     %n.param: %i32.builtin = value_param call_param0
// CHECK:STDOUT:     %.loc54_23.1: type = splice_block %.loc54_23.3 [concrete = constants.%i32.builtin] {
// CHECK:STDOUT:       %Int.ref: %Int.type = name_ref Int, file.%Int.decl [concrete = constants.%Int]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %Int.call: init type = call %Int.ref(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc54_23.2: type = value_of_initializer %Int.call [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc54_23.3: type = converted %Int.call, %.loc54_23.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32.builtin = value_binding n, %n.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param call_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FromLiteral.decl: %FromLiteral.type = fn_decl @FromLiteral [concrete = constants.%FromLiteral] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.dc0 = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.dc0 = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.956 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.956 = return_slot_pattern %return.param_patt, %.loc55_42.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Int.ref: %Int.type = name_ref Int, file.%Int.decl [concrete = constants.%Int]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %Int.call: init type = call %Int.ref(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc55_42.1: type = value_of_initializer %Int.call [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc55_42.2: type = converted %Int.call, %.loc55_42.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc55_42.3: Core.Form = init_form %.loc55_42.2 [concrete = constants.%.9cb]
// CHECK:STDOUT:     %n.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %.loc55_30.1: type = splice_block %.loc55_30.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc55_30.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc55_30.3: type = converted %IntLiteral.call, %.loc55_30.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: Core.IntLiteral = value_binding n, %n.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %N.patt: %pattern_type.956 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:     %return.param_patt: @Make.%pattern_type (%pattern_type.073) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @Make.%pattern_type (%pattern_type.073) = return_slot_pattern %return.param_patt, %.loc57_41.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Int.ref.loc57_25: %Int.type = name_ref Int, file.%Int.decl [concrete = constants.%Int]
// CHECK:STDOUT:     %ToLiteral.ref: %ToLiteral.type = name_ref ToLiteral, file.%ToLiteral.decl [concrete = constants.%ToLiteral]
// CHECK:STDOUT:     %N.ref.loc57_39: %i32.builtin = name_ref N, %N.loc57_9.2 [symbolic = %N.loc57_9.1 (constants.%N.7e8)]
// CHECK:STDOUT:     %ToLiteral.call.loc57_40.2: init Core.IntLiteral = call %ToLiteral.ref(%N.ref.loc57_39) [symbolic = %ToLiteral.call.loc57_40.1 (constants.%ToLiteral.call)]
// CHECK:STDOUT:     %.loc57_40.1: Core.IntLiteral = value_of_initializer %ToLiteral.call.loc57_40.2 [symbolic = %ToLiteral.call.loc57_40.1 (constants.%ToLiteral.call)]
// CHECK:STDOUT:     %.loc57_40.2: Core.IntLiteral = converted %ToLiteral.call.loc57_40.2, %.loc57_40.1 [symbolic = %ToLiteral.call.loc57_40.1 (constants.%ToLiteral.call)]
// CHECK:STDOUT:     %Int.call.loc57_41: init type = call %Int.ref.loc57_25(%.loc57_40.2) [symbolic = %iN.builtin (constants.%iN.builtin.b5d)]
// CHECK:STDOUT:     %.loc57_41.2: type = value_of_initializer %Int.call.loc57_41 [symbolic = %iN.builtin (constants.%iN.builtin.b5d)]
// CHECK:STDOUT:     %.loc57_41.3: type = converted %Int.call.loc57_41, %.loc57_41.2 [symbolic = %iN.builtin (constants.%iN.builtin.b5d)]
// CHECK:STDOUT:     %.loc57_41.4: Core.Form = init_form %.loc57_41.3 [symbolic = %.loc57_41.1 (constants.%.e4e)]
// CHECK:STDOUT:     %.loc57_19.1: type = splice_block %.loc57_19.3 [concrete = constants.%i32.builtin] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Int.ref.loc57_13: %Int.type = name_ref Int, file.%Int.decl [concrete = constants.%Int]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %Int.call.loc57_19: init type = call %Int.ref.loc57_13(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc57_19.2: type = value_of_initializer %Int.call.loc57_19 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc57_19.3: type = converted %Int.call.loc57_19, %.loc57_19.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc57_9.2: %i32.builtin = symbolic_binding N, 0 [symbolic = %N.loc57_9.1 (constants.%N.7e8)]
// CHECK:STDOUT:     %return.param: ref @Make.%iN.builtin (%iN.builtin.b5d) = out_param call_param0
// CHECK:STDOUT:     %return: ref @Make.%iN.builtin (%iN.builtin.b5d) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OtherInt.decl: type = class_decl @OtherInt [concrete = constants.%OtherInt] {} {}
// CHECK:STDOUT:   %OtherInt.ToLiteral.decl: %OtherInt.ToLiteral.type = fn_decl @OtherInt.ToLiteral [concrete = constants.%OtherInt.ToLiteral] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.08d = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.08d = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.dc0 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.dc0 = return_slot_pattern %return.param_patt, %.loc64_51.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref.loc64: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call.loc64: init type = call %IntLiteral.ref.loc64() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc64_51.1: type = value_of_initializer %IntLiteral.call.loc64 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc64_51.2: type = converted %IntLiteral.call.loc64, %.loc64_51.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc64_51.3: Core.Form = init_form %.loc64_51.2 [concrete = constants.%.f7d]
// CHECK:STDOUT:     %self.param.loc64: %OtherInt = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc64: type = name_ref Self, constants.%OtherInt [concrete = constants.%OtherInt]
// CHECK:STDOUT:     %self.loc64: %OtherInt = value_binding self, %self.param.loc64
// CHECK:STDOUT:     %return.param.loc64: ref Core.IntLiteral = out_param call_param1
// CHECK:STDOUT:     %return.loc64: ref Core.IntLiteral = return_slot %return.param.loc64
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeFromClass.decl: %MakeFromClass.type = fn_decl @MakeFromClass [concrete = constants.%MakeFromClass] {
// CHECK:STDOUT:     %N.patt: %pattern_type.08d = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:     %return.param_patt: @MakeFromClass.%pattern_type (%pattern_type.bd7) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @MakeFromClass.%pattern_type (%pattern_type.bd7) = return_slot_pattern %return.param_patt, %.loc66_52.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Int.ref: %Int.type = name_ref Int, file.%Int.decl [concrete = constants.%Int]
// CHECK:STDOUT:     %N.ref.loc66_39: %OtherInt = name_ref N, %N.loc66_18.2 [symbolic = %N.loc66_18.1 (constants.%N.0f7)]
// CHECK:STDOUT:     %ToLiteral.ref: %OtherInt.ToLiteral.type = name_ref ToLiteral, @OtherInt.%OtherInt.ToLiteral.decl [concrete = constants.%OtherInt.ToLiteral]
// CHECK:STDOUT:     %OtherInt.ToLiteral.bound.loc66_40.2: <bound method> = bound_method %N.ref.loc66_39, %ToLiteral.ref [symbolic = %OtherInt.ToLiteral.bound.loc66_40.1 (constants.%OtherInt.ToLiteral.bound)]
// CHECK:STDOUT:     %OtherInt.ToLiteral.call.loc66_51.2: init Core.IntLiteral = call %OtherInt.ToLiteral.bound.loc66_40.2(%N.ref.loc66_39) [symbolic = %OtherInt.ToLiteral.call.loc66_51.1 (constants.%OtherInt.ToLiteral.call)]
// CHECK:STDOUT:     %.loc66_51.1: Core.IntLiteral = value_of_initializer %OtherInt.ToLiteral.call.loc66_51.2 [symbolic = %OtherInt.ToLiteral.call.loc66_51.1 (constants.%OtherInt.ToLiteral.call)]
// CHECK:STDOUT:     %.loc66_51.2: Core.IntLiteral = converted %OtherInt.ToLiteral.call.loc66_51.2, %.loc66_51.1 [symbolic = %OtherInt.ToLiteral.call.loc66_51.1 (constants.%OtherInt.ToLiteral.call)]
// CHECK:STDOUT:     %Int.call: init type = call %Int.ref(%.loc66_51.2) [symbolic = %iN.builtin (constants.%iN.builtin.63a)]
// CHECK:STDOUT:     %.loc66_52.2: type = value_of_initializer %Int.call [symbolic = %iN.builtin (constants.%iN.builtin.63a)]
// CHECK:STDOUT:     %.loc66_52.3: type = converted %Int.call, %.loc66_52.2 [symbolic = %iN.builtin (constants.%iN.builtin.63a)]
// CHECK:STDOUT:     %.loc66_52.4: Core.Form = init_form %.loc66_52.3 [symbolic = %.loc66_52.1 (constants.%.4f8)]
// CHECK:STDOUT:     %.loc66_22: type = splice_block %OtherInt.ref [concrete = constants.%OtherInt] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %OtherInt.ref: type = name_ref OtherInt, file.%OtherInt.decl [concrete = constants.%OtherInt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc66_18.2: %OtherInt = symbolic_binding N, 0 [symbolic = %N.loc66_18.1 (constants.%N.0f7)]
// CHECK:STDOUT:     %return.param: ref @MakeFromClass.%iN.builtin (%iN.builtin.63a) = out_param call_param0
// CHECK:STDOUT:     %return: ref @MakeFromClass.%iN.builtin (%iN.builtin.63a) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @OtherInt {
// CHECK:STDOUT:   %Int.ref: %Int.type = name_ref Int, file.%Int.decl [concrete = constants.%Int]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %Int.call: init type = call %Int.ref(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc60_16.1: type = value_of_initializer %Int.call [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc60_16.2: type = converted %Int.call, %.loc60_16.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   adapt_decl %.loc60_16.2 [concrete]
// CHECK:STDOUT:   %OtherInt.ToLiteral.decl: %OtherInt.ToLiteral.type = fn_decl @OtherInt.ToLiteral [concrete = constants.%OtherInt.ToLiteral] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.08d = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.08d = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.dc0 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.dc0 = return_slot_pattern %return.param_patt, %.loc64_51.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref.loc61: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call.loc61: init type = call %IntLiteral.ref.loc61() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc61_44.1: type = value_of_initializer %IntLiteral.call.loc61 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc61_44.2: type = converted %IntLiteral.call.loc61, %.loc61_44.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc61_44.3: Core.Form = init_form %.loc61_44.2 [concrete = constants.%.f7d]
// CHECK:STDOUT:     %self.param.loc61: %OtherInt = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc61: type = name_ref Self, constants.%OtherInt [concrete = constants.%OtherInt]
// CHECK:STDOUT:     %self.loc61: %OtherInt = value_binding self, %self.param.loc61
// CHECK:STDOUT:     %return.param.loc61: ref Core.IntLiteral = out_param call_param1
// CHECK:STDOUT:     %return.loc61: ref Core.IntLiteral = return_slot %return.param.loc61
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%i32.builtin [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%OtherInt
// CHECK:STDOUT:   .Int = <poisoned>
// CHECK:STDOUT:   .IntLiteral = <poisoned>
// CHECK:STDOUT:   .ToLiteral = %OtherInt.ToLiteral.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> out %return.param: type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param: Core.IntLiteral) -> out %return.param: type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ToLiteral(%n.param: %i32.builtin) -> out %return.param: Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FromLiteral(%n.param: Core.IntLiteral) -> out %return.param: %i32.builtin = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(%N.loc57_9.2: %i32.builtin) {
// CHECK:STDOUT:   %N.loc57_9.1: %i32.builtin = symbolic_binding N, 0 [symbolic = %N.loc57_9.1 (constants.%N.7e8)]
// CHECK:STDOUT:   %ToLiteral.call.loc57_40.1: init Core.IntLiteral = call constants.%ToLiteral(%N.loc57_9.1) [symbolic = %ToLiteral.call.loc57_40.1 (constants.%ToLiteral.call)]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %ToLiteral.call.loc57_40.1 [symbolic = %iN.builtin (constants.%iN.builtin.b5d)]
// CHECK:STDOUT:   %.loc57_41.1: Core.Form = init_form %iN.builtin [symbolic = %.loc57_41.1 (constants.%.e4e)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %iN.builtin [symbolic = %pattern_type (constants.%pattern_type.073)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %iN.builtin [symbolic = %require_complete (constants.%require_complete.8ad)]
// CHECK:STDOUT:   %Make.specific_fn.loc57_52.2: <specific function> = specific_function constants.%Make, @Make(%N.loc57_9.1) [symbolic = %Make.specific_fn.loc57_52.2 (constants.%Make.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Make.%iN.builtin (%iN.builtin.b5d) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:     %N.ref.loc57_57: %i32.builtin = name_ref N, %N.loc57_9.2 [symbolic = %N.loc57_9.1 (constants.%N.7e8)]
// CHECK:STDOUT:     %Make.specific_fn.loc57_52.1: <specific function> = specific_function %Make.ref, @Make(constants.%N.7e8) [symbolic = %Make.specific_fn.loc57_52.2 (constants.%Make.specific_fn)]
// CHECK:STDOUT:     %Make.call: init @Make.%iN.builtin (%iN.builtin.b5d) = call %Make.specific_fn.loc57_52.1()
// CHECK:STDOUT:     return %Make.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @OtherInt.ToLiteral(%self.param.loc64: %OtherInt) -> out %return.param.loc64: Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MakeFromClass(%N.loc66_18.2: %OtherInt) {
// CHECK:STDOUT:   %N.loc66_18.1: %OtherInt = symbolic_binding N, 0 [symbolic = %N.loc66_18.1 (constants.%N.0f7)]
// CHECK:STDOUT:   %OtherInt.ToLiteral.bound.loc66_40.1: <bound method> = bound_method %N.loc66_18.1, constants.%OtherInt.ToLiteral [symbolic = %OtherInt.ToLiteral.bound.loc66_40.1 (constants.%OtherInt.ToLiteral.bound)]
// CHECK:STDOUT:   %OtherInt.ToLiteral.call.loc66_51.1: init Core.IntLiteral = call %OtherInt.ToLiteral.bound.loc66_40.1(%N.loc66_18.1) [symbolic = %OtherInt.ToLiteral.call.loc66_51.1 (constants.%OtherInt.ToLiteral.call)]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %OtherInt.ToLiteral.call.loc66_51.1 [symbolic = %iN.builtin (constants.%iN.builtin.63a)]
// CHECK:STDOUT:   %.loc66_52.1: Core.Form = init_form %iN.builtin [symbolic = %.loc66_52.1 (constants.%.4f8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %iN.builtin [symbolic = %pattern_type (constants.%pattern_type.bd7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %iN.builtin [symbolic = %require_complete (constants.%require_complete.a44)]
// CHECK:STDOUT:   %MakeFromClass.specific_fn.loc66_63.2: <specific function> = specific_function constants.%MakeFromClass, @MakeFromClass(%N.loc66_18.1) [symbolic = %MakeFromClass.specific_fn.loc66_63.2 (constants.%MakeFromClass.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @MakeFromClass.%iN.builtin (%iN.builtin.63a) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %MakeFromClass.ref: %MakeFromClass.type = name_ref MakeFromClass, file.%MakeFromClass.decl [concrete = constants.%MakeFromClass]
// CHECK:STDOUT:     %N.ref.loc66_77: %OtherInt = name_ref N, %N.loc66_18.2 [symbolic = %N.loc66_18.1 (constants.%N.0f7)]
// CHECK:STDOUT:     %MakeFromClass.specific_fn.loc66_63.1: <specific function> = specific_function %MakeFromClass.ref, @MakeFromClass(constants.%N.0f7) [symbolic = %MakeFromClass.specific_fn.loc66_63.2 (constants.%MakeFromClass.specific_fn)]
// CHECK:STDOUT:     %MakeFromClass.call: init @MakeFromClass.%iN.builtin (%iN.builtin.63a) = call %MakeFromClass.specific_fn.loc66_63.1()
// CHECK:STDOUT:     return %MakeFromClass.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%N.7e8) {
// CHECK:STDOUT:   %N.loc57_9.1 => constants.%N.7e8
// CHECK:STDOUT:   %ToLiteral.call.loc57_40.1 => constants.%ToLiteral.call
// CHECK:STDOUT:   %iN.builtin => constants.%iN.builtin.b5d
// CHECK:STDOUT:   %.loc57_41.1 => constants.%.e4e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.073
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.8ad
// CHECK:STDOUT:   %Make.specific_fn.loc57_52.2 => constants.%Make.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeFromClass(constants.%N.0f7) {
// CHECK:STDOUT:   %N.loc66_18.1 => constants.%N.0f7
// CHECK:STDOUT:   %OtherInt.ToLiteral.bound.loc66_40.1 => constants.%OtherInt.ToLiteral.bound
// CHECK:STDOUT:   %OtherInt.ToLiteral.call.loc66_51.1 => constants.%OtherInt.ToLiteral.call
// CHECK:STDOUT:   %iN.builtin => constants.%iN.builtin.63a
// CHECK:STDOUT:   %.loc66_52.1 => constants.%.4f8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bd7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.a44
// CHECK:STDOUT:   %MakeFromClass.specific_fn.loc66_63.2 => constants.%MakeFromClass.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_convert_symbolic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64.fab: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64.fab [concrete]
// CHECK:STDOUT:   %pattern_type.e13: type = pattern_type %i64.builtin [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %N.7e8: %i32.builtin = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %ToLiteral.type: type = fn_type @ToLiteral [concrete]
// CHECK:STDOUT:   %ToLiteral: %ToLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %ToLiteral.call: init Core.IntLiteral = call %ToLiteral(%N.7e8) [symbolic]
// CHECK:STDOUT:   %iN.builtin.b5d: type = int_type signed, %ToLiteral.call [symbolic]
// CHECK:STDOUT:   %pattern_type.073: type = pattern_type %iN.builtin.b5d [symbolic]
// CHECK:STDOUT:   %.e4e: Core.Form = init_form %iN.builtin.b5d [symbolic]
// CHECK:STDOUT:   %Make.specific_fn.9b6: <specific function> = specific_function %Make, @Make(%N.7e8) [symbolic]
// CHECK:STDOUT:   %require_complete.8ad: <witness> = require_complete_type %iN.builtin.b5d [symbolic]
// CHECK:STDOUT:   %FromLiteral.type: type = fn_type @FromLiteral [concrete]
// CHECK:STDOUT:   %FromLiteral: %FromLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64.f82: %i32.builtin = int_value 64 [concrete]
// CHECK:STDOUT:   %.df2: Core.Form = init_form %i64.builtin [concrete]
// CHECK:STDOUT:   %Make.specific_fn.ff0: <specific function> = specific_function %Make, @Make(%int_64.f82) [concrete]
// CHECK:STDOUT:   %MakeFromClass.type: type = fn_type @MakeFromClass [concrete]
// CHECK:STDOUT:   %MakeFromClass: %MakeFromClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %OtherInt: type = class_type @OtherInt [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %N.0f7: %OtherInt = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %OtherInt.ToLiteral.type: type = fn_type @OtherInt.ToLiteral [concrete]
// CHECK:STDOUT:   %OtherInt.ToLiteral: %OtherInt.ToLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %OtherInt.ToLiteral.bound.58f: <bound method> = bound_method %N.0f7, %OtherInt.ToLiteral [symbolic]
// CHECK:STDOUT:   %OtherInt.ToLiteral.call: init Core.IntLiteral = call %OtherInt.ToLiteral.bound.58f(%N.0f7) [symbolic]
// CHECK:STDOUT:   %iN.builtin.63a: type = int_type signed, %OtherInt.ToLiteral.call [symbolic]
// CHECK:STDOUT:   %pattern_type.bd7: type = pattern_type %iN.builtin.63a [symbolic]
// CHECK:STDOUT:   %.4f8: Core.Form = init_form %iN.builtin.63a [symbolic]
// CHECK:STDOUT:   %MakeFromClass.specific_fn.ce0: <specific function> = specific_function %MakeFromClass, @MakeFromClass(%N.0f7) [symbolic]
// CHECK:STDOUT:   %require_complete.a44: <witness> = require_complete_type %iN.builtin.63a [symbolic]
// CHECK:STDOUT:   %int_64.424: %OtherInt = int_value 64 [concrete]
// CHECK:STDOUT:   %OtherInt.ToLiteral.bound.67b: <bound method> = bound_method %int_64.424, %OtherInt.ToLiteral [concrete]
// CHECK:STDOUT:   %MakeFromClass.specific_fn.02a: <specific function> = specific_function %MakeFromClass, @MakeFromClass(%int_64.424) [concrete]
// CHECK:STDOUT:   %complete_type.4a1: <witness> = complete_type_witness %i64.builtin [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.IntLiteral = import_ref Main//convert_symbolic, IntLiteral, unloaded
// CHECK:STDOUT:   %Main.Int: %Int.type = import_ref Main//convert_symbolic, Int, loaded [concrete = constants.%Int]
// CHECK:STDOUT:   %Main.ToLiteral = import_ref Main//convert_symbolic, ToLiteral, unloaded
// CHECK:STDOUT:   %Main.FromLiteral: %FromLiteral.type = import_ref Main//convert_symbolic, FromLiteral, loaded [concrete = constants.%FromLiteral]
// CHECK:STDOUT:   %Main.Make: %Make.type = import_ref Main//convert_symbolic, Make, loaded [concrete = constants.%Make]
// CHECK:STDOUT:   %Main.OtherInt: type = import_ref Main//convert_symbolic, OtherInt, loaded [concrete = constants.%OtherInt]
// CHECK:STDOUT:   %Main.MakeFromClass: %MakeFromClass.type = import_ref Main//convert_symbolic, MakeFromClass, loaded [concrete = constants.%MakeFromClass]
// CHECK:STDOUT:   %Main.import_ref.87d: %i32.builtin = import_ref Main//convert_symbolic, loc57_9, loaded [symbolic = @Make.%N (constants.%N.7e8)]
// CHECK:STDOUT:   %Main.import_ref.b03: <witness> = import_ref Main//convert_symbolic, loc62_1, loaded [concrete = constants.%complete_type.f8a]
// CHECK:STDOUT:   %Main.import_ref.451 = import_ref Main//convert_symbolic, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.d55 = import_ref Main//convert_symbolic, loc61_45, unloaded
// CHECK:STDOUT:   %Main.import_ref.f59: %OtherInt = import_ref Main//convert_symbolic, loc66_18, loaded [symbolic = @MakeFromClass.%N (constants.%N.0f7)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IntLiteral = imports.%Main.IntLiteral
// CHECK:STDOUT:     .Int = imports.%Main.Int
// CHECK:STDOUT:     .ToLiteral = imports.%Main.ToLiteral
// CHECK:STDOUT:     .FromLiteral = imports.%Main.FromLiteral
// CHECK:STDOUT:     .Make = imports.%Main.Make
// CHECK:STDOUT:     .OtherInt = imports.%Main.OtherInt
// CHECK:STDOUT:     .MakeFromClass = imports.%Main.MakeFromClass
// CHECK:STDOUT:     .m = %m
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %m.patt: %pattern_type.e13 = ref_binding_pattern m [concrete]
// CHECK:STDOUT:     %m.var_patt: %pattern_type.e13 = var_pattern %m.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.var: ref %i64.builtin = var %m.var_patt [concrete]
// CHECK:STDOUT:   %.loc74_14.1: type = splice_block %.loc74_14.3 [concrete = constants.%i64.builtin] {
// CHECK:STDOUT:     %Int.ref.loc74: %Int.type = name_ref Int, imports.%Main.Int [concrete = constants.%Int]
// CHECK:STDOUT:     %int_64.loc74: Core.IntLiteral = int_value 64 [concrete = constants.%int_64.fab]
// CHECK:STDOUT:     %Int.call.loc74: init type = call %Int.ref.loc74(%int_64.loc74) [concrete = constants.%i64.builtin]
// CHECK:STDOUT:     %.loc74_14.2: type = value_of_initializer %Int.call.loc74 [concrete = constants.%i64.builtin]
// CHECK:STDOUT:     %.loc74_14.3: type = converted %Int.call.loc74, %.loc74_14.2 [concrete = constants.%i64.builtin]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m: ref %i64.builtin = ref_binding m, %m.var [concrete = %m.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.e13 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.e13 = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %i64.builtin = var %n.var_patt [concrete]
// CHECK:STDOUT:   %.loc75_14.1: type = splice_block %.loc75_14.3 [concrete = constants.%i64.builtin] {
// CHECK:STDOUT:     %Int.ref.loc75: %Int.type = name_ref Int, imports.%Main.Int [concrete = constants.%Int]
// CHECK:STDOUT:     %int_64.loc75: Core.IntLiteral = int_value 64 [concrete = constants.%int_64.fab]
// CHECK:STDOUT:     %Int.call.loc75: init type = call %Int.ref.loc75(%int_64.loc75) [concrete = constants.%i64.builtin]
// CHECK:STDOUT:     %.loc75_14.2: type = value_of_initializer %Int.call.loc75 [concrete = constants.%i64.builtin]
// CHECK:STDOUT:     %.loc75_14.3: type = converted %Int.call.loc75, %.loc75_14.2 [concrete = constants.%i64.builtin]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %i64.builtin = ref_binding n, %n.var [concrete = %n.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @OtherInt [from "convert_symbolic.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.b03
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.451
// CHECK:STDOUT:   .ToLiteral = imports.%Main.import_ref.d55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int = "int.make_type_signed" [from "convert_symbolic.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(imports.%Main.import_ref.87d: %i32.builtin) [from "convert_symbolic.carbon"] {
// CHECK:STDOUT:   %N: %i32.builtin = symbolic_binding N, 0 [symbolic = %N (constants.%N.7e8)]
// CHECK:STDOUT:   %ToLiteral.call: init Core.IntLiteral = call constants.%ToLiteral(%N) [symbolic = %ToLiteral.call (constants.%ToLiteral.call)]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %ToLiteral.call [symbolic = %iN.builtin (constants.%iN.builtin.b5d)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %iN.builtin [symbolic = %.1 (constants.%.e4e)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %iN.builtin [symbolic = %pattern_type (constants.%pattern_type.073)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %iN.builtin [symbolic = %require_complete (constants.%require_complete.8ad)]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function constants.%Make, @Make(%N) [symbolic = %Make.specific_fn (constants.%Make.specific_fn.9b6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ToLiteral = "int.convert_checked" [from "convert_symbolic.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FromLiteral = "int.convert_checked" [from "convert_symbolic.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MakeFromClass(imports.%Main.import_ref.f59: %OtherInt) [from "convert_symbolic.carbon"] {
// CHECK:STDOUT:   %N: %OtherInt = symbolic_binding N, 0 [symbolic = %N (constants.%N.0f7)]
// CHECK:STDOUT:   %OtherInt.ToLiteral.bound: <bound method> = bound_method %N, constants.%OtherInt.ToLiteral [symbolic = %OtherInt.ToLiteral.bound (constants.%OtherInt.ToLiteral.bound.58f)]
// CHECK:STDOUT:   %OtherInt.ToLiteral.call: init Core.IntLiteral = call %OtherInt.ToLiteral.bound(%N) [symbolic = %OtherInt.ToLiteral.call (constants.%OtherInt.ToLiteral.call)]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %OtherInt.ToLiteral.call [symbolic = %iN.builtin (constants.%iN.builtin.63a)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %iN.builtin [symbolic = %.1 (constants.%.4f8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %iN.builtin [symbolic = %pattern_type (constants.%pattern_type.bd7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %iN.builtin [symbolic = %require_complete (constants.%require_complete.a44)]
// CHECK:STDOUT:   %MakeFromClass.specific_fn: <specific function> = specific_function constants.%MakeFromClass, @MakeFromClass(%N) [symbolic = %MakeFromClass.specific_fn (constants.%MakeFromClass.specific_fn.ce0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @OtherInt.ToLiteral = "int.convert_checked" [from "convert_symbolic.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, imports.%Main.Make [concrete = constants.%Make]
// CHECK:STDOUT:   %FromLiteral.ref.loc74: %FromLiteral.type = name_ref FromLiteral, imports.%Main.FromLiteral [concrete = constants.%FromLiteral]
// CHECK:STDOUT:   %int_64.loc74: Core.IntLiteral = int_value 64 [concrete = constants.%int_64.fab]
// CHECK:STDOUT:   %FromLiteral.call.loc74: init %i32.builtin = call %FromLiteral.ref.loc74(%int_64.loc74) [concrete = constants.%int_64.f82]
// CHECK:STDOUT:   %.loc74_38.1: %i32.builtin = value_of_initializer %FromLiteral.call.loc74 [concrete = constants.%int_64.f82]
// CHECK:STDOUT:   %.loc74_38.2: %i32.builtin = converted %FromLiteral.call.loc74, %.loc74_38.1 [concrete = constants.%int_64.f82]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make.ref, @Make(constants.%int_64.f82) [concrete = constants.%Make.specific_fn.ff0]
// CHECK:STDOUT:   %Make.call: init %i64.builtin = call %Make.specific_fn()
// CHECK:STDOUT:   assign file.%m.var, %Make.call
// CHECK:STDOUT:   %MakeFromClass.ref: %MakeFromClass.type = name_ref MakeFromClass, imports.%Main.MakeFromClass [concrete = constants.%MakeFromClass]
// CHECK:STDOUT:   %FromLiteral.ref.loc75: %FromLiteral.type = name_ref FromLiteral, imports.%Main.FromLiteral [concrete = constants.%FromLiteral]
// CHECK:STDOUT:   %int_64.loc75: Core.IntLiteral = int_value 64 [concrete = constants.%int_64.fab]
// CHECK:STDOUT:   %FromLiteral.call.loc75: init %i32.builtin = call %FromLiteral.ref.loc75(%int_64.loc75) [concrete = constants.%int_64.f82]
// CHECK:STDOUT:   %OtherInt.ref: type = name_ref OtherInt, imports.%Main.OtherInt [concrete = constants.%OtherInt]
// CHECK:STDOUT:   %.loc75_48.1: init %OtherInt = as_compatible %FromLiteral.call.loc75 [concrete = constants.%int_64.424]
// CHECK:STDOUT:   %.loc75_48.2: init %OtherInt = converted %FromLiteral.call.loc75, %.loc75_48.1 [concrete = constants.%int_64.424]
// CHECK:STDOUT:   %.loc75_59.1: %OtherInt = value_of_initializer %.loc75_48.2 [concrete = constants.%int_64.424]
// CHECK:STDOUT:   %.loc75_59.2: %OtherInt = converted %.loc75_48.2, %.loc75_59.1 [concrete = constants.%int_64.424]
// CHECK:STDOUT:   %MakeFromClass.specific_fn: <specific function> = specific_function %MakeFromClass.ref, @MakeFromClass(constants.%int_64.424) [concrete = constants.%MakeFromClass.specific_fn.02a]
// CHECK:STDOUT:   %MakeFromClass.call: init %i64.builtin = call %MakeFromClass.specific_fn()
// CHECK:STDOUT:   assign file.%n.var, %MakeFromClass.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%N.7e8) {
// CHECK:STDOUT:   %N => constants.%N.7e8
// CHECK:STDOUT:   %ToLiteral.call => constants.%ToLiteral.call
// CHECK:STDOUT:   %iN.builtin => constants.%iN.builtin.b5d
// CHECK:STDOUT:   %.1 => constants.%.e4e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.073
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.8ad
// CHECK:STDOUT:   %Make.specific_fn => constants.%Make.specific_fn.9b6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%int_64.f82) {
// CHECK:STDOUT:   %N => constants.%int_64.f82
// CHECK:STDOUT:   %ToLiteral.call => constants.%int_64.fab
// CHECK:STDOUT:   %iN.builtin => constants.%i64.builtin
// CHECK:STDOUT:   %.1 => constants.%.df2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e13
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.4a1
// CHECK:STDOUT:   %Make.specific_fn => constants.%Make.specific_fn.ff0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeFromClass(constants.%N.0f7) {
// CHECK:STDOUT:   %N => constants.%N.0f7
// CHECK:STDOUT:   %OtherInt.ToLiteral.bound => constants.%OtherInt.ToLiteral.bound.58f
// CHECK:STDOUT:   %OtherInt.ToLiteral.call => constants.%OtherInt.ToLiteral.call
// CHECK:STDOUT:   %iN.builtin => constants.%iN.builtin.63a
// CHECK:STDOUT:   %.1 => constants.%.4f8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bd7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.a44
// CHECK:STDOUT:   %MakeFromClass.specific_fn => constants.%MakeFromClass.specific_fn.ce0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeFromClass(constants.%int_64.424) {
// CHECK:STDOUT:   %N => constants.%int_64.424
// CHECK:STDOUT:   %OtherInt.ToLiteral.bound => constants.%OtherInt.ToLiteral.bound.67b
// CHECK:STDOUT:   %OtherInt.ToLiteral.call => constants.%int_64.fab
// CHECK:STDOUT:   %iN.builtin => constants.%i64.builtin
// CHECK:STDOUT:   %.1 => constants.%.df2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e13
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.4a1
// CHECK:STDOUT:   %MakeFromClass.specific_fn => constants.%MakeFromClass.specific_fn.02a
// CHECK:STDOUT: }
// CHECK:STDOUT:
