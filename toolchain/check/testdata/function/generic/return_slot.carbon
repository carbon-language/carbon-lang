// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Wrap.type: type = generic_class_type @Wrap [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Wrap.generic: %Wrap.type = struct_value () [concrete]
// CHECK:STDOUT:   %Wrap.af6: type = class_type @Wrap, @Wrap(%T) [symbolic]
// CHECK:STDOUT:   %Make.type.652: type = fn_type @Make, @Wrap(%T) [symbolic]
// CHECK:STDOUT:   %Make.eb2: %Make.type.652 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Make.specific_fn.bf1: <specific function> = specific_function %Make.eb2, @Make(%T) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
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
// CHECK:STDOUT:   %Wrap.f84: type = class_type @Wrap, @Wrap(%i32) [concrete]
// CHECK:STDOUT:   %Make.type.572: type = fn_type @Make, @Wrap(%i32) [concrete]
// CHECK:STDOUT:   %Make.2c5: %Make.type.572 = struct_value () [concrete]
// CHECK:STDOUT:   %Make.specific_fn.f29: <specific function> = specific_function %Make.2c5, @Make(%i32) [concrete]
// CHECK:STDOUT:   %Wrap.1aa: type = class_type @Wrap, @Wrap(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Make.type.20e: type = fn_type @Make, @Wrap(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Make.de0: %Make.type.20e = struct_value () [concrete]
// CHECK:STDOUT:   %Make.specific_fn.e43: <specific function> = specific_function %Make.de0, @Make(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Wrap.2eb: type = class_type @Wrap, @Wrap(%C) [concrete]
// CHECK:STDOUT:   %Make.type.708: type = fn_type @Make, @Wrap(%C) [concrete]
// CHECK:STDOUT:   %Make.e4b: %Make.type.708 = struct_value () [concrete]
// CHECK:STDOUT:   %Make.specific_fn.834: <specific function> = specific_function %Make.e4b, @Make(%C) [concrete]
// CHECK:STDOUT:   %complete_type.782: <witness> = complete_type_witness %empty_tuple.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %T.patt.loc11_12.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_12.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_12.1, runtime_param<none> [symbolic = %T.patt.loc11_12.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc11_12.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_12.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Wrap(%T.loc11_12.1: type) {
// CHECK:STDOUT:   %T.loc11_12.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_12.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_12.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_12.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Wrap(%T.loc11_12.2) [symbolic = %Make.type (constants.%Make.type.652)]
// CHECK:STDOUT:   %Make: @Wrap.%Make.type (%Make.type.652) = struct_value () [symbolic = %Make (constants.%Make.eb2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Make.decl: @Wrap.%Make.type (%Make.type.652) = fn_decl @Make [symbolic = @Wrap.%Make (constants.%Make.eb2)] {
// CHECK:STDOUT:       %return.patt: @Make.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Make.%T (%T) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Wrap.%T.loc11_12.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %return.param: ref @Make.%T (%T) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @Make.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
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
// CHECK:STDOUT:   %.loc15_18: %C.elem = field_decl arr, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc15_11: %C.elem = var_pattern %.loc15_18
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %C.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.arr.5f2 [concrete = constants.%complete_type.22a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .arr = %.loc15_18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(@Wrap.%T.loc11_12.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Make.%T (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Wrap(%T) [symbolic = %Make.type (constants.%Make.type.652)]
// CHECK:STDOUT:   %Make: @Make.%Make.type (%Make.type.652) = struct_value () [symbolic = %Make (constants.%Make.eb2)]
// CHECK:STDOUT:   %Make.specific_fn.loc12_27.2: <specific function> = specific_function %Make, @Make(%T) [symbolic = %Make.specific_fn.loc12_27.2 (constants.%Make.specific_fn.bf1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc12_27: @Make.%Make.type (%Make.type.652) = specific_constant @Wrap.%Make.decl, @Wrap(constants.%T) [symbolic = %Make (constants.%Make.eb2)]
// CHECK:STDOUT:     %Make.ref: @Make.%Make.type (%Make.type.652) = name_ref Make, %.loc12_27 [symbolic = %Make (constants.%Make.eb2)]
// CHECK:STDOUT:     %Make.specific_fn.loc12_27.1: <specific function> = specific_function %Make.ref, @Make(constants.%T) [symbolic = %Make.specific_fn.loc12_27.2 (constants.%Make.specific_fn.bf1)]
// CHECK:STDOUT:     %Make.call: init @Make.%T (%T) = call %Make.specific_fn.loc12_27.1()
// CHECK:STDOUT:     %.loc12_33.1: @Make.%T (%T) = value_of_initializer %Make.call
// CHECK:STDOUT:     %.loc12_33.2: @Make.%T (%T) = converted %Make.call, %.loc12_33.1
// CHECK:STDOUT:     return %.loc12_33.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %.loc18_3: %i32 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %i32 = var a
// CHECK:STDOUT:   %Wrap.ref.loc18: %Wrap.type = name_ref Wrap, file.%Wrap.decl [concrete = constants.%Wrap.generic]
// CHECK:STDOUT:   %int_32.loc18_21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc18_21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Wrap.loc18: type = class_type @Wrap, @Wrap(constants.%i32) [concrete = constants.%Wrap.f84]
// CHECK:STDOUT:   %.loc18_25: %Make.type.572 = specific_constant @Wrap.%Make.decl, @Wrap(constants.%i32) [concrete = constants.%Make.2c5]
// CHECK:STDOUT:   %Make.ref.loc18: %Make.type.572 = name_ref Make, %.loc18_25 [concrete = constants.%Make.2c5]
// CHECK:STDOUT:   %Make.specific_fn.loc18: <specific function> = specific_function %Make.ref.loc18, @Make(constants.%i32) [concrete = constants.%Make.specific_fn.f29]
// CHECK:STDOUT:   %Make.call.loc18: init %i32 = call %Make.specific_fn.loc18()
// CHECK:STDOUT:   assign %a.var, %Make.call.loc18
// CHECK:STDOUT:   %.loc18_10: type = splice_block %i32.loc18_10 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc18_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %i32 = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %empty_tuple.type = binding_pattern b
// CHECK:STDOUT:     %.loc19_3: %empty_tuple.type = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %empty_tuple.type = var b
// CHECK:STDOUT:   %Wrap.ref.loc19: %Wrap.type = name_ref Wrap, file.%Wrap.decl [concrete = constants.%Wrap.generic]
// CHECK:STDOUT:   %.loc19_21: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc19_22: type = converted %.loc19_21, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Wrap.loc19: type = class_type @Wrap, @Wrap(constants.%empty_tuple.type) [concrete = constants.%Wrap.1aa]
// CHECK:STDOUT:   %.loc19_23: %Make.type.20e = specific_constant @Wrap.%Make.decl, @Wrap(constants.%empty_tuple.type) [concrete = constants.%Make.de0]
// CHECK:STDOUT:   %Make.ref.loc19: %Make.type.20e = name_ref Make, %.loc19_23 [concrete = constants.%Make.de0]
// CHECK:STDOUT:   %Make.specific_fn.loc19: <specific function> = specific_function %Make.ref.loc19, @Make(constants.%empty_tuple.type) [concrete = constants.%Make.specific_fn.e43]
// CHECK:STDOUT:   %Make.call.loc19: init %empty_tuple.type = call %Make.specific_fn.loc19()
// CHECK:STDOUT:   assign %b.var, %Make.call.loc19
// CHECK:STDOUT:   %.loc19_11.1: type = splice_block %.loc19_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc19_11.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc19_11.3: type = converted %.loc19_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %empty_tuple.type = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %.loc20_3.1: %C = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %Wrap.ref.loc20: %Wrap.type = name_ref Wrap, file.%Wrap.decl [concrete = constants.%Wrap.generic]
// CHECK:STDOUT:   %C.ref.loc20_19: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Wrap.loc20: type = class_type @Wrap, @Wrap(constants.%C) [concrete = constants.%Wrap.2eb]
// CHECK:STDOUT:   %.loc20_21: %Make.type.708 = specific_constant @Wrap.%Make.decl, @Wrap(constants.%C) [concrete = constants.%Make.e4b]
// CHECK:STDOUT:   %Make.ref.loc20: %Make.type.708 = name_ref Make, %.loc20_21 [concrete = constants.%Make.e4b]
// CHECK:STDOUT:   %Make.specific_fn.loc20: <specific function> = specific_function %Make.ref.loc20, @Make(constants.%C) [concrete = constants.%Make.specific_fn.834]
// CHECK:STDOUT:   %.loc20_3.2: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %Make.call.loc20: init %C = call %Make.specific_fn.loc20() to %.loc20_3.2
// CHECK:STDOUT:   assign %c.var, %Make.call.loc20
// CHECK:STDOUT:   %C.ref.loc20_10: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%T) {
// CHECK:STDOUT:   %T.loc11_12.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_12.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.652
// CHECK:STDOUT:   %Make => constants.%Make.eb2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %Make.type => constants.%Make.type.652
// CHECK:STDOUT:   %Make => constants.%Make.eb2
// CHECK:STDOUT:   %Make.specific_fn.loc12_27.2 => constants.%Make.specific_fn.bf1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(%T.loc11_12.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(@Make.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%i32) {
// CHECK:STDOUT:   %T.loc11_12.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc11_12.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.572
// CHECK:STDOUT:   %Make => constants.%Make.2c5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Make.type => constants.%Make.type.572
// CHECK:STDOUT:   %Make => constants.%Make.2c5
// CHECK:STDOUT:   %Make.specific_fn.loc12_27.2 => constants.%Make.specific_fn.f29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T.loc11_12.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %T.patt.loc11_12.2 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.20e
// CHECK:STDOUT:   %Make => constants.%Make.de0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.782
// CHECK:STDOUT:   %Make.type => constants.%Make.type.20e
// CHECK:STDOUT:   %Make => constants.%Make.de0
// CHECK:STDOUT:   %Make.specific_fn.loc12_27.2 => constants.%Make.specific_fn.e43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%C) {
// CHECK:STDOUT:   %T.loc11_12.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc11_12.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.708
// CHECK:STDOUT:   %Make => constants.%Make.e4b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.22a
// CHECK:STDOUT:   %Make.type => constants.%Make.type.708
// CHECK:STDOUT:   %Make => constants.%Make.e4b
// CHECK:STDOUT:   %Make.specific_fn.loc12_27.2 => constants.%Make.specific_fn.834
// CHECK:STDOUT: }
// CHECK:STDOUT:
