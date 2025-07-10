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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Wrap.type: type = generic_class_type @Wrap [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Wrap.generic: %Wrap.type = struct_value () [concrete]
// CHECK:STDOUT:   %Wrap.af6: type = class_type @Wrap, @Wrap(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Make.type.652: type = fn_type @Make, @Wrap(%T) [symbolic]
// CHECK:STDOUT:   %Make.eb2: %Make.type.652 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Make.specific_fn.bf1: <specific function> = specific_function %Make.eb2, @Make(%T) [symbolic]
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
// CHECK:STDOUT:   %struct_type.arr.5f2: type = struct_type {.arr: %array_type} [concrete]
// CHECK:STDOUT:   %complete_type.22a: <witness> = complete_type_witness %struct_type.arr.5f2 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Wrap.f84: type = class_type @Wrap, @Wrap(%i32) [concrete]
// CHECK:STDOUT:   %Make.type.572: type = fn_type @Make, @Wrap(%i32) [concrete]
// CHECK:STDOUT:   %Make.2c5: %Make.type.572 = struct_value () [concrete]
// CHECK:STDOUT:   %Make.specific_fn.f29: <specific function> = specific_function %Make.2c5, @Make(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Wrap.1aa: type = class_type @Wrap, @Wrap(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Make.type.20e: type = fn_type @Make, @Wrap(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Make.de0: %Make.type.20e = struct_value () [concrete]
// CHECK:STDOUT:   %Make.specific_fn.e43: <specific function> = specific_function %Make.de0, @Make(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Wrap.2eb: type = class_type @Wrap, @Wrap(%C) [concrete]
// CHECK:STDOUT:   %Make.type.708: type = fn_type @Make, @Wrap(%C) [concrete]
// CHECK:STDOUT:   %Make.e4b: %Make.type.708 = struct_value () [concrete]
// CHECK:STDOUT:   %Make.specific_fn.834: <specific function> = specific_function %Make.e4b, @Make(%C) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.153: type = fn_type @Op.2, @impl.49c(%C) [concrete]
// CHECK:STDOUT:   %Op.5d7: %Op.type.153 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Op.specific_fn.6eb: <specific function> = specific_function %Op.5d7, @Op.2(%C) [concrete]
// CHECK:STDOUT:   %Op.type.a63: type = fn_type @Op.2, @impl.49c(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Op.ea3: %Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.843: type = ptr_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Op.specific_fn.393: <specific function> = specific_function %Op.ea3, @Op.2(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Op.type.a17: type = fn_type @Op.2, @impl.49c(%i32) [concrete]
// CHECK:STDOUT:   %Op.e6a: %Op.type.a17 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.014: <specific function> = specific_function %Op.e6a, @Op.2(%i32) [concrete]
// CHECK:STDOUT:   %complete_type.782: <witness> = complete_type_witness %empty_tuple.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %T.loc15_12.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_12.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Wrap(%T.loc15_12.1: type) {
// CHECK:STDOUT:   %T.loc15_12.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_12.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Wrap(%T.loc15_12.2) [symbolic = %Make.type (constants.%Make.type.652)]
// CHECK:STDOUT:   %Make: @Wrap.%Make.type (%Make.type.652) = struct_value () [symbolic = %Make (constants.%Make.eb2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Make.decl: @Wrap.%Make.type (%Make.type.652) = fn_decl @Make [symbolic = @Wrap.%Make (constants.%Make.eb2)] {
// CHECK:STDOUT:       %return.patt: @Make.%pattern_type (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Make.%pattern_type (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Wrap.%T.loc15_12.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %return.param: ref @Make.%T (%T) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Make.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Wrap.af6
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_100: Core.IntLiteral = int_value 100 [concrete = constants.%int_100]
// CHECK:STDOUT:   %array_type: type = array_type %int_100, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:   %.loc19: %C.elem = field_decl arr, element0 [concrete]
// CHECK:STDOUT:   %struct_type.arr: type = struct_type {.arr: %array_type} [concrete = constants.%struct_type.arr.5f2]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.arr [concrete = constants.%complete_type.22a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .arr = %.loc19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(@Wrap.%T.loc15_12.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Wrap(%T) [symbolic = %Make.type (constants.%Make.type.652)]
// CHECK:STDOUT:   %Make: @Make.%Make.type (%Make.type.652) = struct_value () [symbolic = %Make (constants.%Make.eb2)]
// CHECK:STDOUT:   %Make.specific_fn.loc16_27.2: <specific function> = specific_function %Make, @Make(%T) [symbolic = %Make.specific_fn.loc16_27.2 (constants.%Make.specific_fn.bf1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc16_27: @Make.%Make.type (%Make.type.652) = specific_constant @Wrap.%Make.decl, @Wrap(constants.%T) [symbolic = %Make (constants.%Make.eb2)]
// CHECK:STDOUT:     %Make.ref: @Make.%Make.type (%Make.type.652) = name_ref Make, %.loc16_27 [symbolic = %Make (constants.%Make.eb2)]
// CHECK:STDOUT:     %Make.specific_fn.loc16_27.1: <specific function> = specific_function %Make.ref, @Make(constants.%T) [symbolic = %Make.specific_fn.loc16_27.2 (constants.%Make.specific_fn.bf1)]
// CHECK:STDOUT:     %Make.call: init @Make.%T (%T) = call %Make.specific_fn.loc16_27.1()
// CHECK:STDOUT:     %.loc16_33.1: @Make.%T (%T) = value_of_initializer %Make.call
// CHECK:STDOUT:     %.loc16_33.2: @Make.%T (%T) = converted %Make.call, %.loc16_33.1
// CHECK:STDOUT:     return %.loc16_33.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %.loc22_25: %Make.type.572 = specific_constant @Wrap.%Make.decl, @Wrap(constants.%i32) [concrete = constants.%Make.2c5]
// CHECK:STDOUT:   %Make.ref.loc22: %Make.type.572 = name_ref Make, %.loc22_25 [concrete = constants.%Make.2c5]
// CHECK:STDOUT:   %Make.specific_fn.loc22: <specific function> = specific_function %Make.ref.loc22, @Make(constants.%i32) [concrete = constants.%Make.specific_fn.f29]
// CHECK:STDOUT:   %Make.call.loc22: init %i32 = call %Make.specific_fn.loc22()
// CHECK:STDOUT:   assign %a.var, %Make.call.loc22
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
// CHECK:STDOUT:   %.loc23_23: %Make.type.20e = specific_constant @Wrap.%Make.decl, @Wrap(constants.%empty_tuple.type) [concrete = constants.%Make.de0]
// CHECK:STDOUT:   %Make.ref.loc23: %Make.type.20e = name_ref Make, %.loc23_23 [concrete = constants.%Make.de0]
// CHECK:STDOUT:   %Make.specific_fn.loc23: <specific function> = specific_function %Make.ref.loc23, @Make(constants.%empty_tuple.type) [concrete = constants.%Make.specific_fn.e43]
// CHECK:STDOUT:   %Make.call.loc23: init %empty_tuple.type = call %Make.specific_fn.loc23()
// CHECK:STDOUT:   assign %b.var, %Make.call.loc23
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
// CHECK:STDOUT:   %.loc24_21: %Make.type.708 = specific_constant @Wrap.%Make.decl, @Wrap(constants.%C) [concrete = constants.%Make.e4b]
// CHECK:STDOUT:   %Make.ref.loc24: %Make.type.708 = name_ref Make, %.loc24_21 [concrete = constants.%Make.e4b]
// CHECK:STDOUT:   %Make.specific_fn.loc24: <specific function> = specific_function %Make.ref.loc24, @Make(constants.%C) [concrete = constants.%Make.specific_fn.834]
// CHECK:STDOUT:   %.loc24_3: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %Make.call.loc24: init %C = call %Make.specific_fn.loc24() to %.loc24_3
// CHECK:STDOUT:   assign %c.var, %Make.call.loc24
// CHECK:STDOUT:   %C.ref.loc24_10: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   %Op.bound.loc24_3.1: <bound method> = bound_method %.loc24_3, constants.%Op.5d7
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.5d7, @Op.2(constants.%C) [concrete = constants.%Op.specific_fn.6eb]
// CHECK:STDOUT:   %bound_method.loc24_3.1: <bound method> = bound_method %.loc24_3, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc24_3.1: %ptr.019 = addr_of %.loc24_3
// CHECK:STDOUT:   %no_op.loc24_3.1: init %empty_tuple.type = call %bound_method.loc24_3.1(%addr.loc24_3.1)
// CHECK:STDOUT:   %Op.bound.loc24_3.2: <bound method> = bound_method %c.var, constants.%Op.5d7
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.5d7, @Op.2(constants.%C) [concrete = constants.%Op.specific_fn.6eb]
// CHECK:STDOUT:   %bound_method.loc24_3.2: <bound method> = bound_method %c.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc24_3.2: %ptr.019 = addr_of %c.var
// CHECK:STDOUT:   %no_op.loc24_3.2: init %empty_tuple.type = call %bound_method.loc24_3.2(%addr.loc24_3.2)
// CHECK:STDOUT:   %Op.bound.loc23: <bound method> = bound_method %b.var, constants.%Op.ea3
// CHECK:STDOUT:   %Op.specific_fn.3: <specific function> = specific_function constants.%Op.ea3, @Op.2(constants.%empty_tuple.type) [concrete = constants.%Op.specific_fn.393]
// CHECK:STDOUT:   %bound_method.loc23: <bound method> = bound_method %b.var, %Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc23: %ptr.843 = addr_of %b.var
// CHECK:STDOUT:   %no_op.loc23: init %empty_tuple.type = call %bound_method.loc23(%addr.loc23)
// CHECK:STDOUT:   %Op.bound.loc22: <bound method> = bound_method %a.var, constants.%Op.e6a
// CHECK:STDOUT:   %Op.specific_fn.4: <specific function> = specific_function constants.%Op.e6a, @Op.2(constants.%i32) [concrete = constants.%Op.specific_fn.014]
// CHECK:STDOUT:   %bound_method.loc22: <bound method> = bound_method %a.var, %Op.specific_fn.4
// CHECK:STDOUT:   %addr.loc22: %ptr.235 = addr_of %a.var
// CHECK:STDOUT:   %no_op.loc22: init %empty_tuple.type = call %bound_method.loc22(%addr.loc22)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%T) {
// CHECK:STDOUT:   %T.loc15_12.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.652
// CHECK:STDOUT:   %Make => constants.%Make.eb2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %Make.type => constants.%Make.type.652
// CHECK:STDOUT:   %Make => constants.%Make.eb2
// CHECK:STDOUT:   %Make.specific_fn.loc16_27.2 => constants.%Make.specific_fn.bf1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%i32) {
// CHECK:STDOUT:   %T.loc15_12.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.572
// CHECK:STDOUT:   %Make => constants.%Make.2c5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Make.type => constants.%Make.type.572
// CHECK:STDOUT:   %Make => constants.%Make.2c5
// CHECK:STDOUT:   %Make.specific_fn.loc16_27.2 => constants.%Make.specific_fn.f29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T.loc15_12.2 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.20e
// CHECK:STDOUT:   %Make => constants.%Make.de0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cb1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.782
// CHECK:STDOUT:   %Make.type => constants.%Make.type.20e
// CHECK:STDOUT:   %Make => constants.%Make.de0
// CHECK:STDOUT:   %Make.specific_fn.loc16_27.2 => constants.%Make.specific_fn.e43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%C) {
// CHECK:STDOUT:   %T.loc15_12.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.708
// CHECK:STDOUT:   %Make => constants.%Make.e4b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c48
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.22a
// CHECK:STDOUT:   %Make.type => constants.%Make.type.708
// CHECK:STDOUT:   %Make => constants.%Make.e4b
// CHECK:STDOUT:   %Make.specific_fn.loc16_27.2 => constants.%Make.specific_fn.834
// CHECK:STDOUT: }
// CHECK:STDOUT:
