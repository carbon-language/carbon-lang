// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/import_builtin_call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/import_builtin_call.carbon

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

// TODO: The call below crashes if this one is not present because the generics
// get imported in a bad order.
fn LocalDouble(x: MyInt(64)) -> MyInt(64) {
  return x.(Add.Op)(x);
}

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
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Op.type.31b: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.d59: %Op.type.31b = struct_value () [concrete]
// CHECK:STDOUT:   %Add.assoc_type: type = assoc_entity_type %Add.type [concrete]
// CHECK:STDOUT:   %assoc0: %Add.assoc_type = assoc_entity element0, @Add.%Op.decl [concrete]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %MyInt.type: type = generic_class_type @MyInt [concrete]
// CHECK:STDOUT:   %MyInt.generic: %MyInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %MyInt: type = class_type @MyInt, @MyInt(%N) [symbolic]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %require_complete.f1b: <witness> = require_complete_type %iN.builtin [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %iN.builtin [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl), @impl(%N) [symbolic]
// CHECK:STDOUT:   %Op.type.883: type = fn_type @Op.2, @impl(%N) [symbolic]
// CHECK:STDOUT:   %Op.8bc: %Op.type.883 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.fc7: <witness> = require_complete_type %MyInt [symbolic]
// CHECK:STDOUT:   %Add.facet.c2b: %Add.type = facet_value %MyInt, (%impl_witness) [symbolic]
// CHECK:STDOUT:   %Double.type: type = fn_type @Double [concrete]
// CHECK:STDOUT:   %Double: %Double.type = struct_value () [concrete]
// CHECK:STDOUT:   %Add.impl_symbolic_witness: <witness> = impl_symbolic_witness %MyInt, @Add [symbolic]
// CHECK:STDOUT:   %Add.facet.f76: %Add.type = facet_value %MyInt, (%Add.impl_symbolic_witness) [symbolic]
// CHECK:STDOUT:   %.dfa: type = fn_type_with_self_type %Op.type.31b, %Add.facet.f76 [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.dfa = impl_witness_access %Add.impl_symbolic_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Op.1(%Add.facet.f76) [symbolic]
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
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [concrete = constants.%Int] {
// CHECK:STDOUT:     %n.patt: Core.IntLiteral = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: Core.IntLiteral = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %.loc9_22.1: type = splice_block %.loc9_22.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc9_22.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc9_22.3: type = converted %int_literal.make_type, %.loc9_22.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: Core.IntLiteral = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyInt.decl: %MyInt.type = class_decl @MyInt [concrete = constants.%MyInt.generic] {
// CHECK:STDOUT:     %N.patt.loc11_13.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc11_13.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_28.1: type = splice_block %.loc11_28.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc11_28.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc11_28.3: type = converted %int_literal.make_type, %.loc11_28.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc11_13.1: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc11_13.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %N.patt.loc15_14.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc15_14.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %MyInt.ref: %MyInt.type = name_ref MyInt, file.%MyInt.decl [concrete = constants.%MyInt.generic]
// CHECK:STDOUT:     %N.ref: Core.IntLiteral = name_ref N, %N.loc15_14.1 [symbolic = %N.loc15_14.2 (constants.%N)]
// CHECK:STDOUT:     %MyInt.loc15_39.1: type = class_type @MyInt, @MyInt(constants.%N) [symbolic = %MyInt.loc15_39.2 (constants.%MyInt)]
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, file.%Add.decl [concrete = constants.%Add.type]
// CHECK:STDOUT:     %.loc15_29.1: type = splice_block %.loc15_29.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc15_29.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc15_29.3: type = converted %int_literal.make_type, %.loc15_29.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc15_14.1: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc15_14.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl), @impl(constants.%N) [symbolic = @impl.%impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:   %Double.decl: %Double.type = fn_decl @Double [concrete = constants.%Double] {
// CHECK:STDOUT:     %N.patt.loc19_11.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc19_11.2 (constants.%N.patt)]
// CHECK:STDOUT:     %x.patt: @Double.%MyInt.loc19_39.2 (%MyInt) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Double.%MyInt.loc19_39.2 (%MyInt) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @Double.%MyInt.loc19_39.2 (%MyInt) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Double.%MyInt.loc19_39.2 (%MyInt) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %MyInt.ref.loc19_45: %MyInt.type = name_ref MyInt, file.%MyInt.decl [concrete = constants.%MyInt.generic]
// CHECK:STDOUT:     %N.ref.loc19_51: Core.IntLiteral = name_ref N, %N.loc19_11.1 [symbolic = %N.loc19_11.2 (constants.%N)]
// CHECK:STDOUT:     %MyInt.loc19_52: type = class_type @MyInt, @MyInt(constants.%N) [symbolic = %MyInt.loc19_39.2 (constants.%MyInt)]
// CHECK:STDOUT:     %.loc19_26.1: type = splice_block %.loc19_26.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc19_26.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc19_26.3: type = converted %int_literal.make_type, %.loc19_26.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc19_11.1: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc19_11.2 (constants.%N)]
// CHECK:STDOUT:     %x.param: @Double.%MyInt.loc19_39.2 (%MyInt) = value_param call_param0
// CHECK:STDOUT:     %.loc19_39: type = splice_block %MyInt.loc19_39.1 [symbolic = %MyInt.loc19_39.2 (constants.%MyInt)] {
// CHECK:STDOUT:       %MyInt.ref.loc19_32: %MyInt.type = name_ref MyInt, file.%MyInt.decl [concrete = constants.%MyInt.generic]
// CHECK:STDOUT:       %N.ref.loc19_38: Core.IntLiteral = name_ref N, %N.loc19_11.1 [symbolic = %N.loc19_11.2 (constants.%N)]
// CHECK:STDOUT:       %MyInt.loc19_39.1: type = class_type @MyInt, @MyInt(constants.%N) [symbolic = %MyInt.loc19_39.2 (constants.%MyInt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Double.%MyInt.loc19_39.2 (%MyInt) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Double.%MyInt.loc19_39.2 (%MyInt) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Double.%MyInt.loc19_39.2 (%MyInt) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Op.decl: %Op.type.31b = fn_decl @Op.1 [concrete = constants.%Op.d59] {
// CHECK:STDOUT:     %self.patt: @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %other.patt: @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type) = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:     %return.patt: @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc5_37: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc5_37: type = facet_access_type %Self.ref.loc5_37 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc5_37: type = converted %Self.ref.loc5_37, %Self.as_type.loc5_37 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %self.param: @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_15.1: type = splice_block %.loc5_15.2 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc5_15: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_15.2: type = facet_access_type %Self.ref.loc5_15 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_15.2: type = converted %Self.ref.loc5_15, %Self.as_type.loc5_15.2 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type) = value_param call_param1
// CHECK:STDOUT:     %.loc5_28.1: type = splice_block %.loc5_28.2 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc5_28: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_28: type = facet_access_type %Self.ref.loc5_28 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_28.2: type = converted %Self.ref.loc5_28, %Self.as_type.loc5_28 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Add.assoc_type = assoc_entity element0, %Op.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%N.loc15_14.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc15_14.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc15_14.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc15_14.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc15_14.2 (constants.%N.patt)]
// CHECK:STDOUT:   %MyInt.loc15_39.2: type = class_type @MyInt, @MyInt(%N.loc15_14.2) [symbolic = %MyInt.loc15_39.2 (constants.%MyInt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Op.decl), @impl(%N.loc15_14.2) [symbolic = %impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%N.loc15_14.2) [symbolic = %Op.type (constants.%Op.type.883)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.883) = struct_value () [symbolic = %Op (constants.%Op.8bc)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%MyInt.loc15_39.2 (%MyInt) [symbolic = %require_complete (constants.%require_complete.fc7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %MyInt.loc15_39.1 as %Add.ref {
// CHECK:STDOUT:     %Op.decl: @impl.%Op.type (%Op.type.883) = fn_decl @Op.2 [symbolic = @impl.%Op (constants.%Op.8bc)] {
// CHECK:STDOUT:       %self.patt: @Op.2.%MyInt (%MyInt) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Op.2.%MyInt (%MyInt) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %other.patt: @Op.2.%MyInt (%MyInt) = binding_pattern other
// CHECK:STDOUT:       %other.param_patt: @Op.2.%MyInt (%MyInt) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:       %return.patt: @Op.2.%MyInt (%MyInt) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Op.2.%MyInt (%MyInt) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref.loc16_37: type = name_ref Self, @impl.%MyInt.loc15_39.1 [symbolic = %MyInt (constants.%MyInt)]
// CHECK:STDOUT:       %self.param: @Op.2.%MyInt (%MyInt) = value_param call_param0
// CHECK:STDOUT:       %Self.ref.loc16_15: type = name_ref Self, @impl.%MyInt.loc15_39.1 [symbolic = %MyInt (constants.%MyInt)]
// CHECK:STDOUT:       %self: @Op.2.%MyInt (%MyInt) = bind_name self, %self.param
// CHECK:STDOUT:       %other.param: @Op.2.%MyInt (%MyInt) = value_param call_param1
// CHECK:STDOUT:       %Self.ref.loc16_28: type = name_ref Self, @impl.%MyInt.loc15_39.1 [symbolic = %MyInt (constants.%MyInt)]
// CHECK:STDOUT:       %other: @Op.2.%MyInt (%MyInt) = bind_name other, %other.param
// CHECK:STDOUT:       %return.param: ref @Op.2.%MyInt (%MyInt) = out_param call_param2
// CHECK:STDOUT:       %return: ref @Op.2.%MyInt (%MyInt) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Op.decl
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @MyInt(%N.loc11_13.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc11_13.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc11_13.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc11_13.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc11_13.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %N.loc11_13.2 [symbolic = %iN.builtin (constants.%iN.builtin)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @MyInt.%iN.builtin (%iN.builtin) [symbolic = %require_complete (constants.%require_complete.f1b)]
// CHECK:STDOUT:   %complete_type.loc13_1.2: <witness> = complete_type_witness @MyInt.%iN.builtin (%iN.builtin) [symbolic = %complete_type.loc13_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Int.ref: %Int.type = name_ref Int, file.%Int.decl [concrete = constants.%Int]
// CHECK:STDOUT:     %N.ref: Core.IntLiteral = name_ref N, %N.loc11_13.1 [symbolic = %N.loc11_13.2 (constants.%N)]
// CHECK:STDOUT:     %int.make_type_signed: init type = call %Int.ref(%N.ref) [symbolic = %iN.builtin (constants.%iN.builtin)]
// CHECK:STDOUT:     %.loc12_15.1: type = value_of_initializer %int.make_type_signed [symbolic = %iN.builtin (constants.%iN.builtin)]
// CHECK:STDOUT:     %.loc12_15.2: type = converted %int.make_type_signed, %.loc12_15.1 [symbolic = %iN.builtin (constants.%iN.builtin)]
// CHECK:STDOUT:     adapt_decl %.loc12_15.2 [concrete]
// CHECK:STDOUT:     %complete_type.loc13_1.1: <witness> = complete_type_witness %iN.builtin [symbolic = %complete_type.loc13_1.2 (constants.%complete_type)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc13_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%MyInt
// CHECK:STDOUT:     .Int = <poisoned>
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@Add.%Self: %Add.type) {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type)](%other.param_patt: @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type)) -> @Op.1.%Self.as_type.loc5_15.1 (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(@impl.%N.loc15_14.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %MyInt: type = class_type @MyInt, @MyInt(%N) [symbolic = %MyInt (constants.%MyInt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%MyInt (%MyInt)](%other.param_patt: @Op.2.%MyInt (%MyInt)) -> @Op.2.%MyInt (%MyInt) = "int.sadd";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Double(%N.loc19_11.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc19_11.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc19_11.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc19_11.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc19_11.2 (constants.%N.patt)]
// CHECK:STDOUT:   %MyInt.loc19_39.2: type = class_type @MyInt, @MyInt(%N.loc19_11.2) [symbolic = %MyInt.loc19_39.2 (constants.%MyInt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Double.%MyInt.loc19_39.2 (%MyInt) [symbolic = %require_complete (constants.%require_complete.fc7)]
// CHECK:STDOUT:   %Add.impl_symbolic_witness: <witness> = impl_symbolic_witness %MyInt.loc19_39.2, @Add [symbolic = %Add.impl_symbolic_witness (constants.%Add.impl_symbolic_witness)]
// CHECK:STDOUT:   %Add.facet.loc20_11: %Add.type = facet_value %MyInt.loc19_39.2, (%Add.impl_symbolic_witness) [symbolic = %Add.facet.loc20_11 (constants.%Add.facet.f76)]
// CHECK:STDOUT:   %.loc20_11: type = fn_type_with_self_type constants.%Op.type.31b, %Add.facet.loc20_11 [symbolic = %.loc20_11 (constants.%.dfa)]
// CHECK:STDOUT:   %impl.elem0.loc20_11.2: @Double.%.loc20_11 (%.dfa) = impl_witness_access %Add.impl_symbolic_witness, element0 [symbolic = %impl.elem0.loc20_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc20_11.2: <specific function> = specific_impl_function %impl.elem0.loc20_11.2, @Op.1(%Add.facet.loc20_11) [symbolic = %specific_impl_fn.loc20_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.patt.loc19_11.1: Core.IntLiteral](%x.param_patt: @Double.%MyInt.loc19_39.2 (%MyInt)) -> @Double.%MyInt.loc19_39.2 (%MyInt) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc20_10: @Double.%MyInt.loc19_39.2 (%MyInt) = name_ref x, %x
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, file.%Add.decl [concrete = constants.%Add.type]
// CHECK:STDOUT:     %Op.ref: %Add.assoc_type = name_ref Op, @Add.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc20_11.1: @Double.%.loc20_11 (%.dfa) = impl_witness_access constants.%Add.impl_symbolic_witness, element0 [symbolic = %impl.elem0.loc20_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc20_11: <bound method> = bound_method %x.ref.loc20_10, %impl.elem0.loc20_11.1
// CHECK:STDOUT:     %x.ref.loc20_21: @Double.%MyInt.loc19_39.2 (%MyInt) = name_ref x, %x
// CHECK:STDOUT:     %Add.facet.loc20_22.1: %Add.type = facet_value constants.%MyInt, (constants.%Add.impl_symbolic_witness) [symbolic = %Add.facet.loc20_11 (constants.%Add.facet.f76)]
// CHECK:STDOUT:     %.loc20_22.1: %Add.type = converted constants.%MyInt, %Add.facet.loc20_22.1 [symbolic = %Add.facet.loc20_11 (constants.%Add.facet.f76)]
// CHECK:STDOUT:     %Add.facet.loc20_22.2: %Add.type = facet_value constants.%MyInt, (constants.%Add.impl_symbolic_witness) [symbolic = %Add.facet.loc20_11 (constants.%Add.facet.f76)]
// CHECK:STDOUT:     %.loc20_22.2: %Add.type = converted constants.%MyInt, %Add.facet.loc20_22.2 [symbolic = %Add.facet.loc20_11 (constants.%Add.facet.f76)]
// CHECK:STDOUT:     %specific_impl_fn.loc20_11.1: <specific function> = specific_impl_function %impl.elem0.loc20_11.1, @Op.1(constants.%Add.facet.f76) [symbolic = %specific_impl_fn.loc20_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc20_22: <bound method> = bound_method %x.ref.loc20_10, %specific_impl_fn.loc20_11.1
// CHECK:STDOUT:     %.loc20_22.3: init @Double.%MyInt.loc19_39.2 (%MyInt) = call %bound_method.loc20_22(%x.ref.loc20_10, %x.ref.loc20_21)
// CHECK:STDOUT:     %.loc20_23.1: @Double.%MyInt.loc19_39.2 (%MyInt) = value_of_initializer %.loc20_22.3
// CHECK:STDOUT:     %.loc20_23.2: @Double.%MyInt.loc19_39.2 (%MyInt) = converted %.loc20_22.3, %.loc20_23.1
// CHECK:STDOUT:     return %.loc20_23.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_15.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MyInt(constants.%N) {
// CHECK:STDOUT:   %N.loc11_13.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc11_13.2 => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %iN.builtin => constants.%iN.builtin
// CHECK:STDOUT:   %require_complete => constants.%require_complete.f1b
// CHECK:STDOUT:   %complete_type.loc13_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%N) {
// CHECK:STDOUT:   %N.loc15_14.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc15_14.2 => constants.%N
// CHECK:STDOUT:   %MyInt.loc15_39.2 => constants.%MyInt
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.883
// CHECK:STDOUT:   %Op => constants.%Op.8bc
// CHECK:STDOUT:   %require_complete => constants.%require_complete.fc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MyInt(@impl.%N.loc15_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%N.loc15_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %MyInt => constants.%MyInt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MyInt(@Op.2.%N) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Add.facet.c2b) {
// CHECK:STDOUT:   %Self => constants.%Add.facet.c2b
// CHECK:STDOUT:   %Self.as_type.loc5_15.1 => constants.%MyInt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Double(constants.%N) {
// CHECK:STDOUT:   %N.loc19_11.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc19_11.2 => constants.%N
// CHECK:STDOUT:   %MyInt.loc19_39.2 => constants.%MyInt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MyInt(@Double.%N.loc19_11.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Add.facet.f76) {
// CHECK:STDOUT:   %Self => constants.%Add.facet.f76
// CHECK:STDOUT:   %Self.as_type.loc5_15.1 => constants.%MyInt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(@Double.%Add.facet.loc20_11) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_generic_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MyInt.type: type = generic_class_type @MyInt [concrete]
// CHECK:STDOUT:   %MyInt.generic: %MyInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %complete_type.a87: <witness> = complete_type_witness %iN.builtin [symbolic]
// CHECK:STDOUT:   %MyInt.09f: type = class_type @MyInt, @MyInt(%N) [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %require_complete.f1b: <witness> = require_complete_type %iN.builtin [symbolic]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %MyInt.f30: type = class_type @MyInt, @MyInt(%int_64) [concrete]
// CHECK:STDOUT:   %LocalDouble.type: type = fn_type @LocalDouble [concrete]
// CHECK:STDOUT:   %LocalDouble: %LocalDouble.type = struct_value () [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64 [concrete]
// CHECK:STDOUT:   %complete_type.4a1: <witness> = complete_type_witness %i64.builtin [concrete]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [concrete]
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Add.assoc_type: type = assoc_entity_type %Add.type [concrete]
// CHECK:STDOUT:   %assoc0: %Add.assoc_type = assoc_entity element0, imports.%Main.import_ref.5a3 [concrete]
// CHECK:STDOUT:   %Op.type.31b: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %impl_witness.3a3: <witness> = impl_witness (imports.%Main.import_ref.19b), @impl(%N) [symbolic]
// CHECK:STDOUT:   %Op.type.883: type = fn_type @Op.2, @impl(%N) [symbolic]
// CHECK:STDOUT:   %Op.8bc: %Op.type.883 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.fc7: <witness> = require_complete_type %MyInt.09f [symbolic]
// CHECK:STDOUT:   %impl_witness.8d6: <witness> = impl_witness (imports.%Main.import_ref.19b), @impl(%int_64) [concrete]
// CHECK:STDOUT:   %impl_witness.7e5: <witness> = impl_witness (imports.%Main.import_ref.464), @impl(%N) [symbolic]
// CHECK:STDOUT:   %Op.type.5a6: type = fn_type @Op.2, @impl(%int_64) [concrete]
// CHECK:STDOUT:   %Op.cf9: %Op.type.5a6 = struct_value () [concrete]
// CHECK:STDOUT:   %Add.facet.8db: %Add.type = facet_value %MyInt.f30, (%impl_witness.8d6) [concrete]
// CHECK:STDOUT:   %.1ca: type = fn_type_with_self_type %Op.type.31b, %Add.facet.8db [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.cf9, @Op.2(%int_64) [concrete]
// CHECK:STDOUT:   %CallImportedDouble.type: type = fn_type @CallImportedDouble [concrete]
// CHECK:STDOUT:   %CallImportedDouble: %CallImportedDouble.type = struct_value () [concrete]
// CHECK:STDOUT:   %Double.type: type = fn_type @Double [concrete]
// CHECK:STDOUT:   %Double: %Double.type = struct_value () [concrete]
// CHECK:STDOUT:   %Add.impl_symbolic_witness: <witness> = impl_symbolic_witness %MyInt.09f, @Add [symbolic]
// CHECK:STDOUT:   %Add.facet.f76: %Add.type = facet_value %MyInt.09f, (%Add.impl_symbolic_witness) [symbolic]
// CHECK:STDOUT:   %.dfa: type = fn_type_with_self_type %Op.type.31b, %Add.facet.f76 [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.dfa = impl_witness_access %Add.impl_symbolic_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Op.1(%Add.facet.f76) [symbolic]
// CHECK:STDOUT:   %Double.specific_fn: <specific function> = specific_function %Double, @Double(%int_64) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Add: type = import_ref Main//generic_impl, Add, loaded [concrete = constants.%Add.type]
// CHECK:STDOUT:   %Main.IntLiteral = import_ref Main//generic_impl, IntLiteral, unloaded
// CHECK:STDOUT:   %Main.Int = import_ref Main//generic_impl, Int, unloaded
// CHECK:STDOUT:   %Main.MyInt: %MyInt.type = import_ref Main//generic_impl, MyInt, loaded [concrete = constants.%MyInt.generic]
// CHECK:STDOUT:   %Main.Double: %Double.type = import_ref Main//generic_impl, Double, loaded [concrete = constants.%Double]
// CHECK:STDOUT:   %Main.import_ref.f1e294.1: Core.IntLiteral = import_ref Main//generic_impl, loc11_13, loaded [symbolic = @MyInt.%N (constants.%N)]
// CHECK:STDOUT:   %Main.import_ref.9e9: <witness> = import_ref Main//generic_impl, loc13_1, loaded [symbolic = @MyInt.%complete_type (constants.%complete_type.a87)]
// CHECK:STDOUT:   %Main.import_ref.697 = import_ref Main//generic_impl, inst88 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.07c = import_ref Main//generic_impl, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.db7: %Add.assoc_type = import_ref Main//generic_impl, loc5_41, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.Op = import_ref Main//generic_impl, Op, unloaded
// CHECK:STDOUT:   %Main.import_ref.e5e: %Add.type = import_ref Main//generic_impl, inst16 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %Main.import_ref.33b: <witness> = import_ref Main//generic_impl, loc15_48, loaded [symbolic = @impl.%impl_witness (constants.%impl_witness.7e5)]
// CHECK:STDOUT:   %Main.import_ref.f1e294.2: Core.IntLiteral = import_ref Main//generic_impl, loc15_14, loaded [symbolic = @impl.%N (constants.%N)]
// CHECK:STDOUT:   %Main.import_ref.719: type = import_ref Main//generic_impl, loc15_39, loaded [symbolic = @impl.%MyInt (constants.%MyInt.09f)]
// CHECK:STDOUT:   %Main.import_ref.bf0: type = import_ref Main//generic_impl, loc15_44, loaded [concrete = constants.%Add.type]
// CHECK:STDOUT:   %Main.import_ref.19b: @impl.%Op.type (%Op.type.883) = import_ref Main//generic_impl, loc16_42, loaded [symbolic = @impl.%Op (constants.%Op.8bc)]
// CHECK:STDOUT:   %Main.import_ref.f1e294.3: Core.IntLiteral = import_ref Main//generic_impl, loc15_14, loaded [symbolic = @impl.%N (constants.%N)]
// CHECK:STDOUT:   %Main.import_ref.f1e294.4: Core.IntLiteral = import_ref Main//generic_impl, loc19_11, loaded [symbolic = @Double.%N (constants.%N)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Add = imports.%Main.Add
// CHECK:STDOUT:     .IntLiteral = imports.%Main.IntLiteral
// CHECK:STDOUT:     .Int = imports.%Main.Int
// CHECK:STDOUT:     .MyInt = imports.%Main.MyInt
// CHECK:STDOUT:     .Double = imports.%Main.Double
// CHECK:STDOUT:     .LocalDouble = %LocalDouble.decl
// CHECK:STDOUT:     .CallImportedDouble = %CallImportedDouble.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %LocalDouble.decl: %LocalDouble.type = fn_decl @LocalDouble [concrete = constants.%LocalDouble] {
// CHECK:STDOUT:     %x.patt: %MyInt.f30 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %MyInt.f30 = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: %MyInt.f30 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %MyInt.f30 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %MyInt.ref.loc8_33: %MyInt.type = name_ref MyInt, imports.%Main.MyInt [concrete = constants.%MyInt.generic]
// CHECK:STDOUT:     %int_64.loc8_39: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %MyInt.loc8_41: type = class_type @MyInt, @MyInt(constants.%int_64) [concrete = constants.%MyInt.f30]
// CHECK:STDOUT:     %x.param: %MyInt.f30 = value_param call_param0
// CHECK:STDOUT:     %.loc8: type = splice_block %MyInt.loc8_27 [concrete = constants.%MyInt.f30] {
// CHECK:STDOUT:       %MyInt.ref.loc8_19: %MyInt.type = name_ref MyInt, imports.%Main.MyInt [concrete = constants.%MyInt.generic]
// CHECK:STDOUT:       %int_64.loc8_25: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %MyInt.loc8_27: type = class_type @MyInt, @MyInt(constants.%int_64) [concrete = constants.%MyInt.f30]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %MyInt.f30 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %MyInt.f30 = out_param call_param1
// CHECK:STDOUT:     %return: ref %MyInt.f30 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallImportedDouble.decl: %CallImportedDouble.type = fn_decl @CallImportedDouble [concrete = constants.%CallImportedDouble] {
// CHECK:STDOUT:     %n.patt: %MyInt.f30 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %MyInt.f30 = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:     %return.patt: %MyInt.f30 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %MyInt.f30 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %MyInt.ref.loc12_40: %MyInt.type = name_ref MyInt, imports.%Main.MyInt [concrete = constants.%MyInt.generic]
// CHECK:STDOUT:     %int_64.loc12_46: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %MyInt.loc12_48: type = class_type @MyInt, @MyInt(constants.%int_64) [concrete = constants.%MyInt.f30]
// CHECK:STDOUT:     %n.param: %MyInt.f30 = value_param call_param0
// CHECK:STDOUT:     %.loc12: type = splice_block %MyInt.loc12_34 [concrete = constants.%MyInt.f30] {
// CHECK:STDOUT:       %MyInt.ref.loc12_26: %MyInt.type = name_ref MyInt, imports.%Main.MyInt [concrete = constants.%MyInt.generic]
// CHECK:STDOUT:       %int_64.loc12_32: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %MyInt.loc12_34: type = class_type @MyInt, @MyInt(constants.%int_64) [concrete = constants.%MyInt.f30]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %MyInt.f30 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %MyInt.f30 = out_param call_param1
// CHECK:STDOUT:     %return: ref %MyInt.f30 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add [from "generic_impl.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.07c
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.db7
// CHECK:STDOUT:   witness = (imports.%Main.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(imports.%Main.import_ref.f1e294.2: Core.IntLiteral) [from "generic_impl.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %MyInt: type = class_type @MyInt, @MyInt(%N) [symbolic = %MyInt (constants.%MyInt.09f)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Main.import_ref.19b), @impl(%N) [symbolic = %impl_witness (constants.%impl_witness.3a3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%N) [symbolic = %Op.type (constants.%Op.type.883)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.883) = struct_value () [symbolic = %Op (constants.%Op.8bc)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%MyInt (%MyInt.09f) [symbolic = %require_complete (constants.%require_complete.fc7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.719 as imports.%Main.import_ref.bf0 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.33b
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @MyInt(imports.%Main.import_ref.f1e294.1: Core.IntLiteral) [from "generic_impl.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %N [symbolic = %iN.builtin (constants.%iN.builtin)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @MyInt.%iN.builtin (%iN.builtin) [symbolic = %require_complete (constants.%require_complete.f1b)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness @MyInt.%iN.builtin (%iN.builtin) [symbolic = %complete_type (constants.%complete_type.a87)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.9e9
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.697
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LocalDouble(%x.param_patt: %MyInt.f30) -> %MyInt.f30 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc9_10: %MyInt.f30 = name_ref x, %x
// CHECK:STDOUT:   %Add.ref: type = name_ref Add, imports.%Main.Add [concrete = constants.%Add.type]
// CHECK:STDOUT:   %Op.ref: %Add.assoc_type = name_ref Op, imports.%Main.import_ref.db7 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.1ca = impl_witness_access constants.%impl_witness.8d6, element0 [concrete = constants.%Op.cf9]
// CHECK:STDOUT:   %bound_method.loc9_11: <bound method> = bound_method %x.ref.loc9_10, %impl.elem0
// CHECK:STDOUT:   %x.ref.loc9_21: %MyInt.f30 = name_ref x, %x
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Op.2(constants.%int_64) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_22: <bound method> = bound_method %x.ref.loc9_10, %specific_fn
// CHECK:STDOUT:   %int.sadd: init %MyInt.f30 = call %bound_method.loc9_22(%x.ref.loc9_10, %x.ref.loc9_21)
// CHECK:STDOUT:   %.loc9_23.1: %MyInt.f30 = value_of_initializer %int.sadd
// CHECK:STDOUT:   %.loc9_23.2: %MyInt.f30 = converted %int.sadd, %.loc9_23.1
// CHECK:STDOUT:   return %.loc9_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Main.import_ref.e5e: %Add.type) [from "generic_impl.carbon"] {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type (%Self.as_type)](%other.param_patt: @Op.1.%Self.as_type (%Self.as_type)) -> @Op.1.%Self.as_type (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Main.import_ref.f1e294.3: Core.IntLiteral) [from "generic_impl.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %MyInt: type = class_type @MyInt, @MyInt(%N) [symbolic = %MyInt (constants.%MyInt.09f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%MyInt (%MyInt.09f)](%other.param_patt: @Op.2.%MyInt (%MyInt.09f)) -> @Op.2.%MyInt (%MyInt.09f) = "int.sadd";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallImportedDouble(%n.param_patt: %MyInt.f30) -> %MyInt.f30 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Double.ref: %Double.type = name_ref Double, imports.%Main.Double [concrete = constants.%Double]
// CHECK:STDOUT:   %n.ref: %MyInt.f30 = name_ref n, %n
// CHECK:STDOUT:   %Double.specific_fn: <specific function> = specific_function %Double.ref, @Double(constants.%int_64) [concrete = constants.%Double.specific_fn]
// CHECK:STDOUT:   %Double.call: init %MyInt.f30 = call %Double.specific_fn(%n.ref)
// CHECK:STDOUT:   %.loc13_19.1: %MyInt.f30 = value_of_initializer %Double.call
// CHECK:STDOUT:   %.loc13_19.2: %MyInt.f30 = converted %Double.call, %.loc13_19.1
// CHECK:STDOUT:   return %.loc13_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Double(imports.%Main.import_ref.f1e294.4: Core.IntLiteral) [from "generic_impl.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.2 (constants.%N.patt)]
// CHECK:STDOUT:   %MyInt: type = class_type @MyInt, @MyInt(%N) [symbolic = %MyInt (constants.%MyInt.09f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Double.%MyInt (%MyInt.09f) [symbolic = %require_complete (constants.%require_complete.fc7)]
// CHECK:STDOUT:   %Add.impl_symbolic_witness: <witness> = impl_symbolic_witness %MyInt, @Add [symbolic = %Add.impl_symbolic_witness (constants.%Add.impl_symbolic_witness)]
// CHECK:STDOUT:   %Add.facet: %Add.type = facet_value %MyInt, (%Add.impl_symbolic_witness) [symbolic = %Add.facet (constants.%Add.facet.f76)]
// CHECK:STDOUT:   %.1: type = fn_type_with_self_type constants.%Op.type.31b, %Add.facet [symbolic = %.1 (constants.%.dfa)]
// CHECK:STDOUT:   %impl.elem0: @Double.%.1 (%.dfa) = impl_witness_access %Add.impl_symbolic_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Op.1(%Add.facet) [symbolic = %specific_impl_fn (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.patt.1: Core.IntLiteral](%x.param_patt: @Double.%MyInt (%MyInt.09f)) -> @Double.%MyInt (%MyInt.09f);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MyInt(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %iN.builtin => constants.%iN.builtin
// CHECK:STDOUT:   %require_complete => constants.%require_complete.f1b
// CHECK:STDOUT:   %complete_type => constants.%complete_type.a87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MyInt(constants.%int_64) {
// CHECK:STDOUT:   %N => constants.%int_64
// CHECK:STDOUT:   %N.patt => constants.%int_64
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %iN.builtin => constants.%i64.builtin
// CHECK:STDOUT:   %require_complete => constants.%complete_type.4a1
// CHECK:STDOUT:   %complete_type => constants.%complete_type.4a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %MyInt => constants.%MyInt.09f
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.3a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.883
// CHECK:STDOUT:   %Op => constants.%Op.8bc
// CHECK:STDOUT:   %require_complete => constants.%require_complete.fc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MyInt(@impl.%N) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%N) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MyInt(@Op.2.%N) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %MyInt => constants.%MyInt.09f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%int_64) {
// CHECK:STDOUT:   %N => constants.%int_64
// CHECK:STDOUT:   %N.patt => constants.%int_64
// CHECK:STDOUT:   %MyInt => constants.%MyInt.f30
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.8d6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.5a6
// CHECK:STDOUT:   %Op => constants.%Op.cf9
// CHECK:STDOUT:   %require_complete => constants.%complete_type.4a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%int_64) {
// CHECK:STDOUT:   %N => constants.%int_64
// CHECK:STDOUT:   %MyInt => constants.%MyInt.f30
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MyInt(@Double.%N) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Double(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt.2 => constants.%N
// CHECK:STDOUT:   %MyInt => constants.%MyInt.09f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Add.facet.f76) {
// CHECK:STDOUT:   %Self => constants.%Add.facet.f76
// CHECK:STDOUT:   %Self.as_type => constants.%MyInt.09f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(@Double.%Add.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Double(constants.%int_64) {
// CHECK:STDOUT:   %N => constants.%int_64
// CHECK:STDOUT:   %N.patt.2 => constants.%int_64
// CHECK:STDOUT:   %MyInt => constants.%MyInt.f30
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.4a1
// CHECK:STDOUT:   %Add.impl_symbolic_witness => constants.%impl_witness.8d6
// CHECK:STDOUT:   %Add.facet => constants.%Add.facet.8db
// CHECK:STDOUT:   %.1 => constants.%.1ca
// CHECK:STDOUT:   %impl.elem0 => constants.%Op.cf9
// CHECK:STDOUT:   %specific_impl_fn => constants.%Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Add.facet.8db) {
// CHECK:STDOUT:   %Self => constants.%Add.facet.8db
// CHECK:STDOUT:   %Self.as_type => constants.%MyInt.f30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- convert_symbolic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %ToLiteral.type.e7c: type = fn_type @ToLiteral.1 [concrete]
// CHECK:STDOUT:   %ToLiteral.cf3: %ToLiteral.type.e7c = struct_value () [concrete]
// CHECK:STDOUT:   %FromLiteral.type: type = fn_type @FromLiteral [concrete]
// CHECK:STDOUT:   %FromLiteral: %FromLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N.987: %i32.builtin = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.36b: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %int.convert_checked.346: init Core.IntLiteral = call %ToLiteral.cf3(%N.987) [symbolic]
// CHECK:STDOUT:   %iN.builtin.016: type = int_type signed, %int.convert_checked.346 [symbolic]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.88f: <witness> = require_complete_type %iN.builtin.016 [symbolic]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make, @Make(%N.987) [symbolic]
// CHECK:STDOUT:   %OtherInt: type = class_type @OtherInt [concrete]
// CHECK:STDOUT:   %ToLiteral.type.67d: type = fn_type @ToLiteral.2 [concrete]
// CHECK:STDOUT:   %ToLiteral.ec2: %ToLiteral.type.67d = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %N.335: %OtherInt = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.59d: %OtherInt = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %ToLiteral.bound: <bound method> = bound_method %N.335, %ToLiteral.ec2 [symbolic]
// CHECK:STDOUT:   %int.convert_checked.b6b: init Core.IntLiteral = call %ToLiteral.bound(%N.335) [symbolic]
// CHECK:STDOUT:   %iN.builtin.9ef: type = int_type signed, %int.convert_checked.b6b [symbolic]
// CHECK:STDOUT:   %MakeFromClass.type: type = fn_type @MakeFromClass [concrete]
// CHECK:STDOUT:   %MakeFromClass: %MakeFromClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c7d: <witness> = require_complete_type %iN.builtin.9ef [symbolic]
// CHECK:STDOUT:   %MakeFromClass.specific_fn: <specific function> = specific_function %MakeFromClass, @MakeFromClass(%N.335) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .ToLiteral = %ToLiteral.decl.loc6
// CHECK:STDOUT:     .FromLiteral = %FromLiteral.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .OtherInt = %OtherInt.decl
// CHECK:STDOUT:     .MakeFromClass = %MakeFromClass.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [concrete = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [concrete = constants.%Int] {
// CHECK:STDOUT:     %n.patt: Core.IntLiteral = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: Core.IntLiteral = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %.loc5_22.1: type = splice_block %.loc5_22.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_22.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_22.3: type = converted %int_literal.make_type, %.loc5_22.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: Core.IntLiteral = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ToLiteral.decl.loc6: %ToLiteral.type.e7c = fn_decl @ToLiteral.1 [concrete = constants.%ToLiteral.cf3] {
// CHECK:STDOUT:     %n.patt: %i32.builtin = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32.builtin = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_40.1: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_40.2: type = converted %int_literal.make_type, %.loc6_40.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %n.param: %i32.builtin = value_param call_param0
// CHECK:STDOUT:     %.loc6_23.1: type = splice_block %.loc6_23.3 [concrete = constants.%i32.builtin] {
// CHECK:STDOUT:       %Int.ref: %Int.type = name_ref Int, file.%Int.decl [concrete = constants.%Int]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %int.make_type_signed: init type = call %Int.ref(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc6_23.2: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc6_23.3: type = converted %int.make_type_signed, %.loc6_23.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32.builtin = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param call_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FromLiteral.decl: %FromLiteral.type = fn_decl @FromLiteral [concrete = constants.%FromLiteral] {
// CHECK:STDOUT:     %n.patt: Core.IntLiteral = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: Core.IntLiteral = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32.builtin = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32.builtin = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Int.ref: %Int.type = name_ref Int, file.%Int.decl [concrete = constants.%Int]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call %Int.ref(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc7_42.1: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc7_42.2: type = converted %int.make_type_signed, %.loc7_42.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %n.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %.loc7_30.1: type = splice_block %.loc7_30.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc7_30.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc7_30.3: type = converted %int_literal.make_type, %.loc7_30.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: Core.IntLiteral = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %N.patt.loc9_9.1: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc9_9.2 (constants.%N.patt.36b)]
// CHECK:STDOUT:     %return.patt: @Make.%iN.builtin (%iN.builtin.016) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Make.%iN.builtin (%iN.builtin.016) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Int.ref.loc9_25: %Int.type = name_ref Int, file.%Int.decl [concrete = constants.%Int]
// CHECK:STDOUT:     %ToLiteral.ref: %ToLiteral.type.e7c = name_ref ToLiteral, file.%ToLiteral.decl.loc6 [concrete = constants.%ToLiteral.cf3]
// CHECK:STDOUT:     %N.ref.loc9_39: %i32.builtin = name_ref N, %N.loc9_9.1 [symbolic = %N.loc9_9.2 (constants.%N.987)]
// CHECK:STDOUT:     %int.convert_checked.loc9_40.1: init Core.IntLiteral = call %ToLiteral.ref(%N.ref.loc9_39) [symbolic = %int.convert_checked.loc9_40.2 (constants.%int.convert_checked.346)]
// CHECK:STDOUT:     %.loc9_40.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc9_40.1 [symbolic = %int.convert_checked.loc9_40.2 (constants.%int.convert_checked.346)]
// CHECK:STDOUT:     %.loc9_40.2: Core.IntLiteral = converted %int.convert_checked.loc9_40.1, %.loc9_40.1 [symbolic = %int.convert_checked.loc9_40.2 (constants.%int.convert_checked.346)]
// CHECK:STDOUT:     %int.make_type_signed.loc9_41: init type = call %Int.ref.loc9_25(%.loc9_40.2) [symbolic = %iN.builtin (constants.%iN.builtin.016)]
// CHECK:STDOUT:     %.loc9_41.1: type = value_of_initializer %int.make_type_signed.loc9_41 [symbolic = %iN.builtin (constants.%iN.builtin.016)]
// CHECK:STDOUT:     %.loc9_41.2: type = converted %int.make_type_signed.loc9_41, %.loc9_41.1 [symbolic = %iN.builtin (constants.%iN.builtin.016)]
// CHECK:STDOUT:     %.loc9_19.1: type = splice_block %.loc9_19.3 [concrete = constants.%i32.builtin] {
// CHECK:STDOUT:       %Int.ref.loc9_13: %Int.type = name_ref Int, file.%Int.decl [concrete = constants.%Int]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %int.make_type_signed.loc9_19: init type = call %Int.ref.loc9_13(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc9_19.2: type = value_of_initializer %int.make_type_signed.loc9_19 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc9_19.3: type = converted %int.make_type_signed.loc9_19, %.loc9_19.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc9_9.1: %i32.builtin = bind_symbolic_name N, 0 [symbolic = %N.loc9_9.2 (constants.%N.987)]
// CHECK:STDOUT:     %return.param: ref @Make.%iN.builtin (%iN.builtin.016) = out_param call_param0
// CHECK:STDOUT:     %return: ref @Make.%iN.builtin (%iN.builtin.016) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OtherInt.decl: type = class_decl @OtherInt [concrete = constants.%OtherInt] {} {}
// CHECK:STDOUT:   %ToLiteral.decl.loc16: %ToLiteral.type.67d = fn_decl @ToLiteral.2 [concrete = constants.%ToLiteral.ec2] {
// CHECK:STDOUT:     %self.patt: %OtherInt = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %OtherInt = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref.loc16: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc16: init type = call %IntLiteral.ref.loc16() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc16_51.1: type = value_of_initializer %int_literal.make_type.loc16 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc16_51.2: type = converted %int_literal.make_type.loc16, %.loc16_51.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %self.param.loc16: %OtherInt = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc16: type = name_ref Self, constants.%OtherInt [concrete = constants.%OtherInt]
// CHECK:STDOUT:     %self.loc16: %OtherInt = bind_name self, %self.param.loc16
// CHECK:STDOUT:     %return.param.loc16: ref Core.IntLiteral = out_param call_param1
// CHECK:STDOUT:     %return.loc16: ref Core.IntLiteral = return_slot %return.param.loc16
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeFromClass.decl: %MakeFromClass.type = fn_decl @MakeFromClass [concrete = constants.%MakeFromClass] {
// CHECK:STDOUT:     %N.patt.loc18_18.1: %OtherInt = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc18_18.2 (constants.%N.patt.59d)]
// CHECK:STDOUT:     %return.patt: @MakeFromClass.%iN.builtin (%iN.builtin.9ef) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @MakeFromClass.%iN.builtin (%iN.builtin.9ef) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Int.ref: %Int.type = name_ref Int, file.%Int.decl [concrete = constants.%Int]
// CHECK:STDOUT:     %N.ref.loc18_39: %OtherInt = name_ref N, %N.loc18_18.1 [symbolic = %N.loc18_18.2 (constants.%N.335)]
// CHECK:STDOUT:     %ToLiteral.ref: %ToLiteral.type.67d = name_ref ToLiteral, @OtherInt.%ToLiteral.decl [concrete = constants.%ToLiteral.ec2]
// CHECK:STDOUT:     %ToLiteral.bound.loc18_40.1: <bound method> = bound_method %N.ref.loc18_39, %ToLiteral.ref [symbolic = %ToLiteral.bound.loc18_40.2 (constants.%ToLiteral.bound)]
// CHECK:STDOUT:     %int.convert_checked.loc18_51.1: init Core.IntLiteral = call %ToLiteral.bound.loc18_40.1(%N.ref.loc18_39) [symbolic = %int.convert_checked.loc18_51.2 (constants.%int.convert_checked.b6b)]
// CHECK:STDOUT:     %.loc18_51.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc18_51.1 [symbolic = %int.convert_checked.loc18_51.2 (constants.%int.convert_checked.b6b)]
// CHECK:STDOUT:     %.loc18_51.2: Core.IntLiteral = converted %int.convert_checked.loc18_51.1, %.loc18_51.1 [symbolic = %int.convert_checked.loc18_51.2 (constants.%int.convert_checked.b6b)]
// CHECK:STDOUT:     %int.make_type_signed: init type = call %Int.ref(%.loc18_51.2) [symbolic = %iN.builtin (constants.%iN.builtin.9ef)]
// CHECK:STDOUT:     %.loc18_52.1: type = value_of_initializer %int.make_type_signed [symbolic = %iN.builtin (constants.%iN.builtin.9ef)]
// CHECK:STDOUT:     %.loc18_52.2: type = converted %int.make_type_signed, %.loc18_52.1 [symbolic = %iN.builtin (constants.%iN.builtin.9ef)]
// CHECK:STDOUT:     %OtherInt.ref: type = name_ref OtherInt, file.%OtherInt.decl [concrete = constants.%OtherInt]
// CHECK:STDOUT:     %N.loc18_18.1: %OtherInt = bind_symbolic_name N, 0 [symbolic = %N.loc18_18.2 (constants.%N.335)]
// CHECK:STDOUT:     %return.param: ref @MakeFromClass.%iN.builtin (%iN.builtin.9ef) = out_param call_param0
// CHECK:STDOUT:     %return: ref @MakeFromClass.%iN.builtin (%iN.builtin.9ef) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @OtherInt {
// CHECK:STDOUT:   %Int.ref: %Int.type = name_ref Int, file.%Int.decl [concrete = constants.%Int]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call %Int.ref(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc12_16.1: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc12_16.2: type = converted %int.make_type_signed, %.loc12_16.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   adapt_decl %.loc12_16.2 [concrete]
// CHECK:STDOUT:   %ToLiteral.decl: %ToLiteral.type.67d = fn_decl @ToLiteral.2 [concrete = constants.%ToLiteral.ec2] {
// CHECK:STDOUT:     %self.patt: %OtherInt = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %OtherInt = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref.loc13: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc13: init type = call %IntLiteral.ref.loc13() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc13_44.1: type = value_of_initializer %int_literal.make_type.loc13 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc13_44.2: type = converted %int_literal.make_type.loc13, %.loc13_44.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %self.param.loc13: %OtherInt = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc13: type = name_ref Self, constants.%OtherInt [concrete = constants.%OtherInt]
// CHECK:STDOUT:     %self.loc13: %OtherInt = bind_name self, %self.param.loc13
// CHECK:STDOUT:     %return.param.loc13: ref Core.IntLiteral = out_param call_param1
// CHECK:STDOUT:     %return.loc13: ref Core.IntLiteral = return_slot %return.param.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %i32.builtin [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%OtherInt
// CHECK:STDOUT:   .Int = <poisoned>
// CHECK:STDOUT:   .IntLiteral = <poisoned>
// CHECK:STDOUT:   .ToLiteral = %ToLiteral.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ToLiteral.1(%n.param_patt: %i32.builtin) -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FromLiteral(%n.param_patt: Core.IntLiteral) -> %i32.builtin = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(%N.loc9_9.1: %i32.builtin) {
// CHECK:STDOUT:   %N.loc9_9.2: %i32.builtin = bind_symbolic_name N, 0 [symbolic = %N.loc9_9.2 (constants.%N.987)]
// CHECK:STDOUT:   %N.patt.loc9_9.2: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc9_9.2 (constants.%N.patt.36b)]
// CHECK:STDOUT:   %int.convert_checked.loc9_40.2: init Core.IntLiteral = call constants.%ToLiteral.cf3(%N.loc9_9.2) [symbolic = %int.convert_checked.loc9_40.2 (constants.%int.convert_checked.346)]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %int.convert_checked.loc9_40.2 [symbolic = %iN.builtin (constants.%iN.builtin.016)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Make.%iN.builtin (%iN.builtin.016) [symbolic = %require_complete (constants.%require_complete.88f)]
// CHECK:STDOUT:   %Make.specific_fn.loc9_52.2: <specific function> = specific_function constants.%Make, @Make(%N.loc9_9.2) [symbolic = %Make.specific_fn.loc9_52.2 (constants.%Make.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.patt.loc9_9.1: %i32.builtin) -> @Make.%iN.builtin (%iN.builtin.016) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:     %N.ref.loc9_57: %i32.builtin = name_ref N, %N.loc9_9.1 [symbolic = %N.loc9_9.2 (constants.%N.987)]
// CHECK:STDOUT:     %Make.specific_fn.loc9_52.1: <specific function> = specific_function %Make.ref, @Make(constants.%N.987) [symbolic = %Make.specific_fn.loc9_52.2 (constants.%Make.specific_fn)]
// CHECK:STDOUT:     %Make.call: init @Make.%iN.builtin (%iN.builtin.016) = call %Make.specific_fn.loc9_52.1()
// CHECK:STDOUT:     %.loc9_59.1: @Make.%iN.builtin (%iN.builtin.016) = value_of_initializer %Make.call
// CHECK:STDOUT:     %.loc9_59.2: @Make.%iN.builtin (%iN.builtin.016) = converted %Make.call, %.loc9_59.1
// CHECK:STDOUT:     return %.loc9_59.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ToLiteral.2[%self.param_patt: %OtherInt]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MakeFromClass(%N.loc18_18.1: %OtherInt) {
// CHECK:STDOUT:   %N.loc18_18.2: %OtherInt = bind_symbolic_name N, 0 [symbolic = %N.loc18_18.2 (constants.%N.335)]
// CHECK:STDOUT:   %N.patt.loc18_18.2: %OtherInt = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc18_18.2 (constants.%N.patt.59d)]
// CHECK:STDOUT:   %ToLiteral.bound.loc18_40.2: <bound method> = bound_method %N.loc18_18.2, constants.%ToLiteral.ec2 [symbolic = %ToLiteral.bound.loc18_40.2 (constants.%ToLiteral.bound)]
// CHECK:STDOUT:   %int.convert_checked.loc18_51.2: init Core.IntLiteral = call %ToLiteral.bound.loc18_40.2(%N.loc18_18.2) [symbolic = %int.convert_checked.loc18_51.2 (constants.%int.convert_checked.b6b)]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %int.convert_checked.loc18_51.2 [symbolic = %iN.builtin (constants.%iN.builtin.9ef)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @MakeFromClass.%iN.builtin (%iN.builtin.9ef) [symbolic = %require_complete (constants.%require_complete.c7d)]
// CHECK:STDOUT:   %MakeFromClass.specific_fn.loc18_63.2: <specific function> = specific_function constants.%MakeFromClass, @MakeFromClass(%N.loc18_18.2) [symbolic = %MakeFromClass.specific_fn.loc18_63.2 (constants.%MakeFromClass.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.patt.loc18_18.1: %OtherInt) -> @MakeFromClass.%iN.builtin (%iN.builtin.9ef) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %MakeFromClass.ref: %MakeFromClass.type = name_ref MakeFromClass, file.%MakeFromClass.decl [concrete = constants.%MakeFromClass]
// CHECK:STDOUT:     %N.ref.loc18_77: %OtherInt = name_ref N, %N.loc18_18.1 [symbolic = %N.loc18_18.2 (constants.%N.335)]
// CHECK:STDOUT:     %MakeFromClass.specific_fn.loc18_63.1: <specific function> = specific_function %MakeFromClass.ref, @MakeFromClass(constants.%N.335) [symbolic = %MakeFromClass.specific_fn.loc18_63.2 (constants.%MakeFromClass.specific_fn)]
// CHECK:STDOUT:     %MakeFromClass.call: init @MakeFromClass.%iN.builtin (%iN.builtin.9ef) = call %MakeFromClass.specific_fn.loc18_63.1()
// CHECK:STDOUT:     %.loc18_79.1: @MakeFromClass.%iN.builtin (%iN.builtin.9ef) = value_of_initializer %MakeFromClass.call
// CHECK:STDOUT:     %.loc18_79.2: @MakeFromClass.%iN.builtin (%iN.builtin.9ef) = converted %MakeFromClass.call, %.loc18_79.1
// CHECK:STDOUT:     return %.loc18_79.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%N.987) {
// CHECK:STDOUT:   %N.loc9_9.2 => constants.%N.987
// CHECK:STDOUT:   %N.patt.loc9_9.2 => constants.%N.987
// CHECK:STDOUT:   %int.convert_checked.loc9_40.2 => constants.%int.convert_checked.346
// CHECK:STDOUT:   %iN.builtin => constants.%iN.builtin.016
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.88f
// CHECK:STDOUT:   %Make.specific_fn.loc9_52.2 => constants.%Make.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(%N.loc9_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeFromClass(constants.%N.335) {
// CHECK:STDOUT:   %N.loc18_18.2 => constants.%N.335
// CHECK:STDOUT:   %N.patt.loc18_18.2 => constants.%N.335
// CHECK:STDOUT:   %ToLiteral.bound.loc18_40.2 => constants.%ToLiteral.bound
// CHECK:STDOUT:   %int.convert_checked.loc18_51.2 => constants.%int.convert_checked.b6b
// CHECK:STDOUT:   %iN.builtin => constants.%iN.builtin.9ef
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.c7d
// CHECK:STDOUT:   %MakeFromClass.specific_fn.loc18_63.2 => constants.%MakeFromClass.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeFromClass(%N.loc18_18.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_convert_symbolic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64.fab: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64.fab [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %N.patt.36b: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N.987: %i32.builtin = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %ToLiteral.type.e7c: type = fn_type @ToLiteral.1 [concrete]
// CHECK:STDOUT:   %ToLiteral.cf3: %ToLiteral.type.e7c = struct_value () [concrete]
// CHECK:STDOUT:   %int.convert_checked.346: init Core.IntLiteral = call %ToLiteral.cf3(%N.987) [symbolic]
// CHECK:STDOUT:   %iN.builtin.016: type = int_type signed, %int.convert_checked.346 [symbolic]
// CHECK:STDOUT:   %require_complete.88f: <witness> = require_complete_type %iN.builtin.016 [symbolic]
// CHECK:STDOUT:   %Make.specific_fn.8ec: <specific function> = specific_function %Make, @Make(%N.987) [symbolic]
// CHECK:STDOUT:   %FromLiteral.type: type = fn_type @FromLiteral [concrete]
// CHECK:STDOUT:   %FromLiteral: %FromLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64.f82: %i32.builtin = int_value 64 [concrete]
// CHECK:STDOUT:   %Make.specific_fn.02d: <specific function> = specific_function %Make, @Make(%int_64.f82) [concrete]
// CHECK:STDOUT:   %MakeFromClass.type: type = fn_type @MakeFromClass [concrete]
// CHECK:STDOUT:   %MakeFromClass: %MakeFromClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %OtherInt: type = class_type @OtherInt [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %N.patt.59d: %OtherInt = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N.335: %OtherInt = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %ToLiteral.type.67d: type = fn_type @ToLiteral.2 [concrete]
// CHECK:STDOUT:   %ToLiteral.ec2: %ToLiteral.type.67d = struct_value () [concrete]
// CHECK:STDOUT:   %ToLiteral.bound.8e3: <bound method> = bound_method %N.335, %ToLiteral.ec2 [symbolic]
// CHECK:STDOUT:   %int.convert_checked.b6b: init Core.IntLiteral = call %ToLiteral.bound.8e3(%N.335) [symbolic]
// CHECK:STDOUT:   %iN.builtin.9ef: type = int_type signed, %int.convert_checked.b6b [symbolic]
// CHECK:STDOUT:   %require_complete.c7d: <witness> = require_complete_type %iN.builtin.9ef [symbolic]
// CHECK:STDOUT:   %MakeFromClass.specific_fn.004: <specific function> = specific_function %MakeFromClass, @MakeFromClass(%N.335) [symbolic]
// CHECK:STDOUT:   %int_64.06b: %OtherInt = int_value 64 [concrete]
// CHECK:STDOUT:   %ToLiteral.bound.735: <bound method> = bound_method %int_64.06b, %ToLiteral.ec2 [concrete]
// CHECK:STDOUT:   %MakeFromClass.specific_fn.61b: <specific function> = specific_function %MakeFromClass, @MakeFromClass(%int_64.06b) [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.85e: %i32.builtin = import_ref Main//convert_symbolic, loc9_9, loaded [symbolic = @Make.%N (constants.%N.987)]
// CHECK:STDOUT:   %Main.import_ref.b03: <witness> = import_ref Main//convert_symbolic, loc14_1, loaded [concrete = constants.%complete_type.f8a]
// CHECK:STDOUT:   %Main.import_ref.d11 = import_ref Main//convert_symbolic, inst128 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.8f7 = import_ref Main//convert_symbolic, loc13_45, unloaded
// CHECK:STDOUT:   %Main.import_ref.77d: %OtherInt = import_ref Main//convert_symbolic, loc18_18, loaded [symbolic = @MakeFromClass.%N (constants.%N.335)]
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
// CHECK:STDOUT:     %m.patt: %i64.builtin = binding_pattern m
// CHECK:STDOUT:     %.loc6_1: %i64.builtin = var_pattern %m.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.var: ref %i64.builtin = var m
// CHECK:STDOUT:   %.loc6_14.1: type = splice_block %.loc6_14.3 [concrete = constants.%i64.builtin] {
// CHECK:STDOUT:     %Int.ref.loc6: %Int.type = name_ref Int, imports.%Main.Int [concrete = constants.%Int]
// CHECK:STDOUT:     %int_64.loc6: Core.IntLiteral = int_value 64 [concrete = constants.%int_64.fab]
// CHECK:STDOUT:     %int.make_type_signed.loc6: init type = call %Int.ref.loc6(%int_64.loc6) [concrete = constants.%i64.builtin]
// CHECK:STDOUT:     %.loc6_14.2: type = value_of_initializer %int.make_type_signed.loc6 [concrete = constants.%i64.builtin]
// CHECK:STDOUT:     %.loc6_14.3: type = converted %int.make_type_signed.loc6, %.loc6_14.2 [concrete = constants.%i64.builtin]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m: ref %i64.builtin = bind_name m, %m.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %i64.builtin = binding_pattern n
// CHECK:STDOUT:     %.loc7_1: %i64.builtin = var_pattern %n.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %i64.builtin = var n
// CHECK:STDOUT:   %.loc7_14.1: type = splice_block %.loc7_14.3 [concrete = constants.%i64.builtin] {
// CHECK:STDOUT:     %Int.ref.loc7: %Int.type = name_ref Int, imports.%Main.Int [concrete = constants.%Int]
// CHECK:STDOUT:     %int_64.loc7: Core.IntLiteral = int_value 64 [concrete = constants.%int_64.fab]
// CHECK:STDOUT:     %int.make_type_signed.loc7: init type = call %Int.ref.loc7(%int_64.loc7) [concrete = constants.%i64.builtin]
// CHECK:STDOUT:     %.loc7_14.2: type = value_of_initializer %int.make_type_signed.loc7 [concrete = constants.%i64.builtin]
// CHECK:STDOUT:     %.loc7_14.3: type = converted %int.make_type_signed.loc7, %.loc7_14.2 [concrete = constants.%i64.builtin]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %i64.builtin = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @OtherInt [from "convert_symbolic.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.b03
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.d11
// CHECK:STDOUT:   .ToLiteral = imports.%Main.import_ref.8f7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n.param_patt: Core.IntLiteral) -> type = "int.make_type_signed" [from "convert_symbolic.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(imports.%Main.import_ref.85e: %i32.builtin) [from "convert_symbolic.carbon"] {
// CHECK:STDOUT:   %N: %i32.builtin = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.987)]
// CHECK:STDOUT:   %N.patt.2: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic = %N.patt.2 (constants.%N.patt.36b)]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call constants.%ToLiteral.cf3(%N) [symbolic = %int.convert_checked (constants.%int.convert_checked.346)]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %int.convert_checked [symbolic = %iN.builtin (constants.%iN.builtin.016)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Make.%iN.builtin (%iN.builtin.016) [symbolic = %require_complete (constants.%require_complete.88f)]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function constants.%Make, @Make(%N) [symbolic = %Make.specific_fn (constants.%Make.specific_fn.8ec)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.patt.1: %i32.builtin) -> @Make.%iN.builtin (%iN.builtin.016);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ToLiteral.1(%n.param_patt: %i32.builtin) -> Core.IntLiteral = "int.convert_checked" [from "convert_symbolic.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FromLiteral(%n.param_patt: Core.IntLiteral) -> %i32.builtin = "int.convert_checked" [from "convert_symbolic.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MakeFromClass(imports.%Main.import_ref.77d: %OtherInt) [from "convert_symbolic.carbon"] {
// CHECK:STDOUT:   %N: %OtherInt = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.335)]
// CHECK:STDOUT:   %N.patt.2: %OtherInt = symbolic_binding_pattern N, 0 [symbolic = %N.patt.2 (constants.%N.patt.59d)]
// CHECK:STDOUT:   %ToLiteral.bound: <bound method> = bound_method %N, constants.%ToLiteral.ec2 [symbolic = %ToLiteral.bound (constants.%ToLiteral.bound.8e3)]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %ToLiteral.bound(%N) [symbolic = %int.convert_checked (constants.%int.convert_checked.b6b)]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %int.convert_checked [symbolic = %iN.builtin (constants.%iN.builtin.9ef)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @MakeFromClass.%iN.builtin (%iN.builtin.9ef) [symbolic = %require_complete (constants.%require_complete.c7d)]
// CHECK:STDOUT:   %MakeFromClass.specific_fn: <specific function> = specific_function constants.%MakeFromClass, @MakeFromClass(%N) [symbolic = %MakeFromClass.specific_fn (constants.%MakeFromClass.specific_fn.004)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.patt.1: %OtherInt) -> @MakeFromClass.%iN.builtin (%iN.builtin.9ef);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ToLiteral.2[%self.param_patt: %OtherInt]() -> Core.IntLiteral = "int.convert_checked" [from "convert_symbolic.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, imports.%Main.Make [concrete = constants.%Make]
// CHECK:STDOUT:   %FromLiteral.ref.loc6: %FromLiteral.type = name_ref FromLiteral, imports.%Main.FromLiteral [concrete = constants.%FromLiteral]
// CHECK:STDOUT:   %int_64.loc6: Core.IntLiteral = int_value 64 [concrete = constants.%int_64.fab]
// CHECK:STDOUT:   %int.convert_checked.loc6: init %i32.builtin = call %FromLiteral.ref.loc6(%int_64.loc6) [concrete = constants.%int_64.f82]
// CHECK:STDOUT:   %.loc6_38.1: %i32.builtin = value_of_initializer %int.convert_checked.loc6 [concrete = constants.%int_64.f82]
// CHECK:STDOUT:   %.loc6_38.2: %i32.builtin = converted %int.convert_checked.loc6, %.loc6_38.1 [concrete = constants.%int_64.f82]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make.ref, @Make(constants.%int_64.f82) [concrete = constants.%Make.specific_fn.02d]
// CHECK:STDOUT:   %Make.call: init %i64.builtin = call %Make.specific_fn()
// CHECK:STDOUT:   assign file.%m.var, %Make.call
// CHECK:STDOUT:   %MakeFromClass.ref: %MakeFromClass.type = name_ref MakeFromClass, imports.%Main.MakeFromClass [concrete = constants.%MakeFromClass]
// CHECK:STDOUT:   %FromLiteral.ref.loc7: %FromLiteral.type = name_ref FromLiteral, imports.%Main.FromLiteral [concrete = constants.%FromLiteral]
// CHECK:STDOUT:   %int_64.loc7: Core.IntLiteral = int_value 64 [concrete = constants.%int_64.fab]
// CHECK:STDOUT:   %int.convert_checked.loc7: init %i32.builtin = call %FromLiteral.ref.loc7(%int_64.loc7) [concrete = constants.%int_64.f82]
// CHECK:STDOUT:   %OtherInt.ref: type = name_ref OtherInt, imports.%Main.OtherInt [concrete = constants.%OtherInt]
// CHECK:STDOUT:   %.loc7_48.1: init %OtherInt = as_compatible %int.convert_checked.loc7 [concrete = constants.%int_64.06b]
// CHECK:STDOUT:   %.loc7_48.2: init %OtherInt = converted %int.convert_checked.loc7, %.loc7_48.1 [concrete = constants.%int_64.06b]
// CHECK:STDOUT:   %.loc7_59.1: %OtherInt = value_of_initializer %.loc7_48.2 [concrete = constants.%int_64.06b]
// CHECK:STDOUT:   %.loc7_59.2: %OtherInt = converted %.loc7_48.2, %.loc7_59.1 [concrete = constants.%int_64.06b]
// CHECK:STDOUT:   %MakeFromClass.specific_fn: <specific function> = specific_function %MakeFromClass.ref, @MakeFromClass(constants.%int_64.06b) [concrete = constants.%MakeFromClass.specific_fn.61b]
// CHECK:STDOUT:   %MakeFromClass.call: init %i64.builtin = call %MakeFromClass.specific_fn()
// CHECK:STDOUT:   assign file.%n.var, %MakeFromClass.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%N.987) {
// CHECK:STDOUT:   %N => constants.%N.987
// CHECK:STDOUT:   %N.patt.2 => constants.%N.987
// CHECK:STDOUT:   %int.convert_checked => constants.%int.convert_checked.346
// CHECK:STDOUT:   %iN.builtin => constants.%iN.builtin.016
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.88f
// CHECK:STDOUT:   %Make.specific_fn => constants.%Make.specific_fn.8ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(%N) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%int_64.f82) {
// CHECK:STDOUT:   %N => constants.%int_64.f82
// CHECK:STDOUT:   %N.patt.2 => constants.%int_64.f82
// CHECK:STDOUT:   %int.convert_checked => constants.%int_64.fab
// CHECK:STDOUT:   %iN.builtin => constants.%i64.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.4a1
// CHECK:STDOUT:   %Make.specific_fn => constants.%Make.specific_fn.02d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeFromClass(constants.%N.335) {
// CHECK:STDOUT:   %N => constants.%N.335
// CHECK:STDOUT:   %N.patt.2 => constants.%N.335
// CHECK:STDOUT:   %ToLiteral.bound => constants.%ToLiteral.bound.8e3
// CHECK:STDOUT:   %int.convert_checked => constants.%int.convert_checked.b6b
// CHECK:STDOUT:   %iN.builtin => constants.%iN.builtin.9ef
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.c7d
// CHECK:STDOUT:   %MakeFromClass.specific_fn => constants.%MakeFromClass.specific_fn.004
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeFromClass(%N) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeFromClass(constants.%int_64.06b) {
// CHECK:STDOUT:   %N => constants.%int_64.06b
// CHECK:STDOUT:   %N.patt.2 => constants.%int_64.06b
// CHECK:STDOUT:   %ToLiteral.bound => constants.%ToLiteral.bound.735
// CHECK:STDOUT:   %int.convert_checked => constants.%int_64.fab
// CHECK:STDOUT:   %iN.builtin => constants.%i64.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.4a1
// CHECK:STDOUT:   %MakeFromClass.specific_fn => constants.%MakeFromClass.specific_fn.61b
// CHECK:STDOUT: }
// CHECK:STDOUT:
