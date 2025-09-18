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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/return_slot.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/return_slot.carbon

class Wrap(T:! type) {
  fn Make() -> T { return Make(); }
}

class C { var arr: array(i32, 100); }

fn G() {
  var a: i32 = Wrap(i32).Make();
  var b: () = Wrap(()).Make();
  var c: C = Wrap(C).Make();
}

// CHECK:STDOUT: --- return_slot.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Wrap.type: type = generic_class_type @Wrap [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Wrap.generic: %Wrap.type = struct_value () [concrete]
// CHECK:STDOUT:   %Wrap.af6: type = class_type @Wrap, @Wrap(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Wrap.Make.type.652: type = fn_type @Wrap.Make, @Wrap(%T) [symbolic]
// CHECK:STDOUT:   %Wrap.Make.eb2: %Wrap.Make.type.652 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.a83: <witness> = impl_witness @Wrap.%Destroy.impl_witness_table, @Wrap.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.b45: type = ptr_type %Wrap.af6 [symbolic]
// CHECK:STDOUT:   %pattern_type.3b2: type = pattern_type %ptr.b45 [symbolic]
// CHECK:STDOUT:   %Wrap.as.Destroy.impl.Op.type: type = fn_type @Wrap.as.Destroy.impl.Op, @Wrap.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Wrap.as.Destroy.impl.Op: %Wrap.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.bf1: <specific function> = specific_function %Wrap.Make.eb2, @Wrap.Make(%T) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_100: Core.IntLiteral = int_value 100 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_100, %i32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %array_type [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.071: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.arr.5f2: type = struct_type {.arr: %array_type} [concrete]
// CHECK:STDOUT:   %complete_type.22a: <witness> = complete_type_witness %struct_type.arr.5f2 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Wrap.f84: type = class_type @Wrap, @Wrap(%i32) [concrete]
// CHECK:STDOUT:   %Wrap.Make.type.572: type = fn_type @Wrap.Make, @Wrap(%i32) [concrete]
// CHECK:STDOUT:   %Wrap.Make.2c5: %Wrap.Make.type.572 = struct_value () [concrete]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.f29: <specific function> = specific_function %Wrap.Make.2c5, @Wrap.Make(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Wrap.1aa: type = class_type @Wrap, @Wrap(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Wrap.Make.type.20e: type = fn_type @Wrap.Make, @Wrap(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Wrap.Make.de0: %Wrap.Make.type.20e = struct_value () [concrete]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.e43: <specific function> = specific_function %Wrap.Make.de0, @Wrap.Make(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Wrap.2eb: type = class_type @Wrap, @Wrap(%C) [concrete]
// CHECK:STDOUT:   %Wrap.Make.type.708: type = fn_type @Wrap.Make, @Wrap(%C) [concrete]
// CHECK:STDOUT:   %Wrap.Make.e4b: %Wrap.Make.type.708 = struct_value () [concrete]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.834: <specific function> = specific_function %Wrap.Make.e4b, @Wrap.Make(%C) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.08a: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.daf: %T.as.Destroy.impl.Op.type.08a = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.843: type = ptr_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %T.as.Destroy.impl.Op.daf, @T.as.Destroy.impl.Op(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.type.4f9: type = fn_type @Int.as.Destroy.impl.Op, @Int.as.Destroy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.796: %Int.as.Destroy.impl.Op.type.4f9 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Destroy.impl.Op.796, @Int.as.Destroy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %complete_type.782: <witness> = complete_type_witness %empty_tuple.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Wrap = %Wrap.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Wrap.decl: %Wrap.type = class_decl @Wrap [concrete = constants.%Wrap.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc15_12.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_12.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Wrap.as.Destroy.impl(@Wrap.%T.loc15_12.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Wrap: type = class_type @Wrap, @Wrap(%T) [symbolic = %Wrap (constants.%Wrap.af6)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Wrap.%Destroy.impl_witness_table, @Wrap.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.a83)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.as.Destroy.impl.Op.type: type = fn_type @Wrap.as.Destroy.impl.Op, @Wrap.as.Destroy.impl(%T) [symbolic = %Wrap.as.Destroy.impl.Op.type (constants.%Wrap.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Wrap.as.Destroy.impl.Op: @Wrap.as.Destroy.impl.%Wrap.as.Destroy.impl.Op.type (%Wrap.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Wrap.as.Destroy.impl.Op (constants.%Wrap.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Wrap.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %Wrap.as.Destroy.impl.Op.decl: @Wrap.as.Destroy.impl.%Wrap.as.Destroy.impl.Op.type (%Wrap.as.Destroy.impl.Op.type) = fn_decl @Wrap.as.Destroy.impl.Op [symbolic = @Wrap.as.Destroy.impl.%Wrap.as.Destroy.impl.Op (constants.%Wrap.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Wrap.as.Destroy.impl.Op.%pattern_type (%pattern_type.3b2) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Wrap.as.Destroy.impl.Op.%pattern_type (%pattern_type.3b2) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc15_22.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Wrap.as.Destroy.impl.Op.%ptr (%ptr.b45) = value_param call_param0
// CHECK:STDOUT:       %.loc15_22.2: type = splice_block %Self.ref [symbolic = %Wrap (constants.%Wrap.af6)] {
// CHECK:STDOUT:         %.loc15_22.3: type = specific_constant constants.%Wrap.af6, @Wrap(constants.%T) [symbolic = %Wrap (constants.%Wrap.af6)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc15_22.3 [symbolic = %Wrap (constants.%Wrap.af6)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Wrap.as.Destroy.impl.Op.%ptr (%ptr.b45) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Wrap.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Wrap.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc19: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Wrap(%T.loc15_12.2: type) {
// CHECK:STDOUT:   %T.loc15_12.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_12.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.Make.type: type = fn_type @Wrap.Make, @Wrap(%T.loc15_12.1) [symbolic = %Wrap.Make.type (constants.%Wrap.Make.type.652)]
// CHECK:STDOUT:   %Wrap.Make: @Wrap.%Wrap.Make.type (%Wrap.Make.type.652) = struct_value () [symbolic = %Wrap.Make (constants.%Wrap.Make.eb2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Wrap.Make.decl: @Wrap.%Wrap.Make.type (%Wrap.Make.type.652) = fn_decl @Wrap.Make [symbolic = @Wrap.%Wrap.Make (constants.%Wrap.Make.eb2)] {
// CHECK:STDOUT:       %return.patt: @Wrap.Make.%pattern_type (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Wrap.Make.%pattern_type (%pattern_type.7dc) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Wrap.%T.loc15_12.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %return.param: ref @Wrap.Make.%T (%T) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Wrap.Make.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Wrap.af6 [symbolic = @Wrap.as.Destroy.impl.%Wrap (constants.%Wrap.af6)]
// CHECK:STDOUT:     impl_decl @Wrap.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Wrap.as.Destroy.impl.%Wrap.as.Destroy.impl.Op.decl), @Wrap.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Wrap.as.Destroy.impl(constants.%T) [symbolic = @Wrap.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.a83)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Wrap.af6
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Make = %Wrap.Make.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_100: Core.IntLiteral = int_value 100 [concrete = constants.%int_100]
// CHECK:STDOUT:   %array_type: type = array_type %int_100, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:   %.loc19: %C.elem = field_decl arr, element0 [concrete]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.071]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.arr.5f2 [concrete = constants.%complete_type.22a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .arr = %.loc19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Wrap.Make(@Wrap.%T.loc15_12.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Wrap.Make.type: type = fn_type @Wrap.Make, @Wrap(%T) [symbolic = %Wrap.Make.type (constants.%Wrap.Make.type.652)]
// CHECK:STDOUT:   %Wrap.Make: @Wrap.Make.%Wrap.Make.type (%Wrap.Make.type.652) = struct_value () [symbolic = %Wrap.Make (constants.%Wrap.Make.eb2)]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2: <specific function> = specific_function %Wrap.Make, @Wrap.Make(%T) [symbolic = %Wrap.Make.specific_fn.loc16_27.2 (constants.%Wrap.Make.specific_fn.bf1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param: @Wrap.Make.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc16_27: @Wrap.Make.%Wrap.Make.type (%Wrap.Make.type.652) = specific_constant @Wrap.%Wrap.Make.decl, @Wrap(constants.%T) [symbolic = %Wrap.Make (constants.%Wrap.Make.eb2)]
// CHECK:STDOUT:     %Make.ref: @Wrap.Make.%Wrap.Make.type (%Wrap.Make.type.652) = name_ref Make, %.loc16_27 [symbolic = %Wrap.Make (constants.%Wrap.Make.eb2)]
// CHECK:STDOUT:     %Wrap.Make.specific_fn.loc16_27.1: <specific function> = specific_function %Make.ref, @Wrap.Make(constants.%T) [symbolic = %Wrap.Make.specific_fn.loc16_27.2 (constants.%Wrap.Make.specific_fn.bf1)]
// CHECK:STDOUT:     %.loc16_13: ref @Wrap.Make.%T (%T) = splice_block %return {}
// CHECK:STDOUT:     %Wrap.Make.call: init @Wrap.Make.%T (%T) = call %Wrap.Make.specific_fn.loc16_27.1() to %.loc16_13
// CHECK:STDOUT:     return %Wrap.Make.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Wrap.as.Destroy.impl.Op(@Wrap.%T.loc15_12.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Wrap: type = class_type @Wrap, @Wrap(%T) [symbolic = %Wrap (constants.%Wrap.af6)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Wrap [symbolic = %ptr (constants.%ptr.b45)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.3b2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Wrap.as.Destroy.impl.Op.%ptr (%ptr.b45)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.7ce = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %i32 = var %a.var_patt
// CHECK:STDOUT:   %Wrap.ref.loc22: %Wrap.type = name_ref Wrap, file.%Wrap.decl [concrete = constants.%Wrap.generic]
// CHECK:STDOUT:   %int_32.loc22_21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc22_21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Wrap.loc22: type = class_type @Wrap, @Wrap(constants.%i32) [concrete = constants.%Wrap.f84]
// CHECK:STDOUT:   %.loc22_25: %Wrap.Make.type.572 = specific_constant @Wrap.%Wrap.Make.decl, @Wrap(constants.%i32) [concrete = constants.%Wrap.Make.2c5]
// CHECK:STDOUT:   %Make.ref.loc22: %Wrap.Make.type.572 = name_ref Make, %.loc22_25 [concrete = constants.%Wrap.Make.2c5]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc22: <specific function> = specific_function %Make.ref.loc22, @Wrap.Make(constants.%i32) [concrete = constants.%Wrap.Make.specific_fn.f29]
// CHECK:STDOUT:   %Wrap.Make.call.loc22: init %i32 = call %Wrap.Make.specific_fn.loc22()
// CHECK:STDOUT:   assign %a.var, %Wrap.Make.call.loc22
// CHECK:STDOUT:   %.loc22_10: type = splice_block %i32.loc22_10 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc22_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %i32 = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.cb1 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %empty_tuple.type = var %b.var_patt
// CHECK:STDOUT:   %Wrap.ref.loc23: %Wrap.type = name_ref Wrap, file.%Wrap.decl [concrete = constants.%Wrap.generic]
// CHECK:STDOUT:   %.loc23_21: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc23_22: type = converted %.loc23_21, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Wrap.loc23: type = class_type @Wrap, @Wrap(constants.%empty_tuple.type) [concrete = constants.%Wrap.1aa]
// CHECK:STDOUT:   %.loc23_23: %Wrap.Make.type.20e = specific_constant @Wrap.%Wrap.Make.decl, @Wrap(constants.%empty_tuple.type) [concrete = constants.%Wrap.Make.de0]
// CHECK:STDOUT:   %Make.ref.loc23: %Wrap.Make.type.20e = name_ref Make, %.loc23_23 [concrete = constants.%Wrap.Make.de0]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc23: <specific function> = specific_function %Make.ref.loc23, @Wrap.Make(constants.%empty_tuple.type) [concrete = constants.%Wrap.Make.specific_fn.e43]
// CHECK:STDOUT:   %Wrap.Make.call.loc23: init %empty_tuple.type = call %Wrap.Make.specific_fn.loc23()
// CHECK:STDOUT:   assign %b.var, %Wrap.Make.call.loc23
// CHECK:STDOUT:   %.loc23_11.1: type = splice_block %.loc23_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc23_11.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc23_11.3: type = converted %.loc23_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %empty_tuple.type = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.c48 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %Wrap.ref.loc24: %Wrap.type = name_ref Wrap, file.%Wrap.decl [concrete = constants.%Wrap.generic]
// CHECK:STDOUT:   %C.ref.loc24_19: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Wrap.loc24: type = class_type @Wrap, @Wrap(constants.%C) [concrete = constants.%Wrap.2eb]
// CHECK:STDOUT:   %.loc24_21: %Wrap.Make.type.708 = specific_constant @Wrap.%Wrap.Make.decl, @Wrap(constants.%C) [concrete = constants.%Wrap.Make.e4b]
// CHECK:STDOUT:   %Make.ref.loc24: %Wrap.Make.type.708 = name_ref Make, %.loc24_21 [concrete = constants.%Wrap.Make.e4b]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc24: <specific function> = specific_function %Make.ref.loc24, @Wrap.Make(constants.%C) [concrete = constants.%Wrap.Make.specific_fn.834]
// CHECK:STDOUT:   %.loc24_3: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %Wrap.Make.call.loc24: init %C = call %Wrap.Make.specific_fn.loc24() to %.loc24_3
// CHECK:STDOUT:   assign %c.var, %Wrap.Make.call.loc24
// CHECK:STDOUT:   %C.ref.loc24_10: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound: <bound method> = bound_method %c.var, constants.%C.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc24: %ptr.019 = addr_of %c.var
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call: init %empty_tuple.type = call %C.as.Destroy.impl.Op.bound(%addr.loc24)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %b.var, constants.%T.as.Destroy.impl.Op.daf
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.daf, @T.as.Destroy.impl.Op(constants.%empty_tuple.type) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23: <bound method> = bound_method %b.var, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc23: %ptr.843 = addr_of %b.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc23(%addr.loc23)
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.bound: <bound method> = bound_method %a.var, constants.%Int.as.Destroy.impl.Op.796
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%Int.as.Destroy.impl.Op.796, @Int.as.Destroy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22: <bound method> = bound_method %a.var, %Int.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc22: %ptr.235 = addr_of %a.var
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc22(%addr.loc22)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%T) {
// CHECK:STDOUT:   %T.loc15_12.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.652
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.eb2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap.Make(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.652
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.eb2
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2 => constants.%Wrap.Make.specific_fn.bf1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Wrap => constants.%Wrap.af6
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.a83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Wrap => constants.%Wrap.af6
// CHECK:STDOUT:   %ptr => constants.%ptr.b45
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3b2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%i32) {
// CHECK:STDOUT:   %T.loc15_12.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.572
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.2c5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap.Make(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.572
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.2c5
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2 => constants.%Wrap.Make.specific_fn.f29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T.loc15_12.1 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.20e
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.de0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap.Make(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cb1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.782
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.20e
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.de0
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2 => constants.%Wrap.Make.specific_fn.e43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%C) {
// CHECK:STDOUT:   %T.loc15_12.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.708
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.e4b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap.Make(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c48
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.22a
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.708
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.e4b
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2 => constants.%Wrap.Make.specific_fn.834
// CHECK:STDOUT: }
// CHECK:STDOUT:
