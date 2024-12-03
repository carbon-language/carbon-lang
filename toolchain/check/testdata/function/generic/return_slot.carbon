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

class C { var arr: [i32; 100]; }

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
// CHECK:STDOUT:   %Wrap.type: type = generic_class_type @Wrap [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Wrap.generic: %Wrap.type = struct_value () [template]
// CHECK:STDOUT:   %Wrap.1: type = class_type @Wrap, @Wrap(%T) [symbolic]
// CHECK:STDOUT:   %Make.type.1: type = fn_type @Make, @Wrap(%T) [symbolic]
// CHECK:STDOUT:   %Make.1: %Make.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Make.specific_fn.1: <specific function> = specific_function %Make.1, @Make(%T) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_100: Core.IntLiteral = int_value 100 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_100, %i32 [template]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %array_type [template]
// CHECK:STDOUT:   %struct_type.arr.1: type = struct_type {.arr: %array_type} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.arr.1 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %Wrap.2: type = class_type @Wrap, @Wrap(%i32) [template]
// CHECK:STDOUT:   %Make.type.2: type = fn_type @Make, @Wrap(%i32) [template]
// CHECK:STDOUT:   %Make.2: %Make.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Make.specific_fn.2: <specific function> = specific_function %Make.2, @Make(%i32) [template]
// CHECK:STDOUT:   %Wrap.3: type = class_type @Wrap, @Wrap(%empty_tuple.type) [template]
// CHECK:STDOUT:   %Make.type.3: type = fn_type @Make, @Wrap(%empty_tuple.type) [template]
// CHECK:STDOUT:   %Make.3: %Make.type.3 = struct_value () [template]
// CHECK:STDOUT:   %Make.specific_fn.3: <specific function> = specific_function %Make.3, @Make(%empty_tuple.type) [template]
// CHECK:STDOUT:   %Wrap.4: type = class_type @Wrap, @Wrap(%C) [template]
// CHECK:STDOUT:   %Make.type.4: type = fn_type @Make, @Wrap(%C) [template]
// CHECK:STDOUT:   %Make.4: %Make.type.4 = struct_value () [template]
// CHECK:STDOUT:   %Make.specific_fn.4: <specific function> = specific_function %Make.4, @Make(%C) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Wrap = %Wrap.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Wrap.decl: %Wrap.type = class_decl @Wrap [template = constants.%Wrap.generic] {
// CHECK:STDOUT:     %T.patt.loc11_12.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_12.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_12.1, runtime_param<invalid> [symbolic = %T.patt.loc11_12.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_12.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_12.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Wrap(%T.loc11_12.1: type) {
// CHECK:STDOUT:   %T.loc11_12.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_12.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_12.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_12.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Wrap(%T.loc11_12.2) [symbolic = %Make.type (constants.%Make.type.1)]
// CHECK:STDOUT:   %Make: @Wrap.%Make.type (%Make.type.1) = struct_value () [symbolic = %Make (constants.%Make.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Make.decl: @Wrap.%Make.type (%Make.type.1) = fn_decl @Make [symbolic = @Wrap.%Make (constants.%Make.1)] {
// CHECK:STDOUT:       %return.patt: @Make.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Make.%T (%T) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Wrap.%T.loc11_12.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %return.param: ref @Make.%T (%T) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @Make.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Wrap.1
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_100: Core.IntLiteral = int_value 100 [template = constants.%int_100]
// CHECK:STDOUT:   %.loc15_21.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_21.2: type = converted %int.make_type_signed, %.loc15_21.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type: type = array_type %int_100, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %.loc15_18: %C.elem = field_decl arr, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.arr.1 [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .arr = %.loc15_18
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(@Wrap.%T.loc11_12.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Wrap(%T) [symbolic = %Make.type (constants.%Make.type.1)]
// CHECK:STDOUT:   %Make: @Make.%Make.type (%Make.type.1) = struct_value () [symbolic = %Make (constants.%Make.1)]
// CHECK:STDOUT:   %Make.specific_fn.loc12_27.2: <specific function> = specific_function %Make, @Make(%T) [symbolic = %Make.specific_fn.loc12_27.2 (constants.%Make.specific_fn.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc12_27: @Make.%Make.type (%Make.type.1) = specific_constant @Wrap.%Make.decl, @Wrap(constants.%T) [symbolic = %Make (constants.%Make.1)]
// CHECK:STDOUT:     %Make.ref: @Make.%Make.type (%Make.type.1) = name_ref Make, %.loc12_27 [symbolic = %Make (constants.%Make.1)]
// CHECK:STDOUT:     %Make.specific_fn.loc12_27.1: <specific function> = specific_function %Make.ref, @Make(constants.%T) [symbolic = %Make.specific_fn.loc12_27.2 (constants.%Make.specific_fn.1)]
// CHECK:STDOUT:     %Make.call: init @Make.%T (%T) = call %Make.specific_fn.loc12_27.1()
// CHECK:STDOUT:     %.loc12_33.1: @Make.%T (%T) = value_of_initializer %Make.call
// CHECK:STDOUT:     %.loc12_33.2: @Make.%T (%T) = converted %Make.call, %.loc12_33.1
// CHECK:STDOUT:     return %.loc12_33.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32.loc18_10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc18_10: init type = call constants.%Int(%int_32.loc18_10) [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_10.1: type = value_of_initializer %int.make_type_signed.loc18_10 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_10.2: type = converted %int.make_type_signed.loc18_10, %.loc18_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %a.var: ref %i32 = var a
// CHECK:STDOUT:   %a: ref %i32 = bind_name a, %a.var
// CHECK:STDOUT:   %Wrap.ref.loc18: %Wrap.type = name_ref Wrap, file.%Wrap.decl [template = constants.%Wrap.generic]
// CHECK:STDOUT:   %int_32.loc18_21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc18_21: init type = call constants.%Int(%int_32.loc18_21) [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_24.1: type = value_of_initializer %int.make_type_signed.loc18_21 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_24.2: type = converted %int.make_type_signed.loc18_21, %.loc18_24.1 [template = constants.%i32]
// CHECK:STDOUT:   %Wrap.loc18: type = class_type @Wrap, @Wrap(constants.%i32) [template = constants.%Wrap.2]
// CHECK:STDOUT:   %.loc18_25: %Make.type.2 = specific_constant @Wrap.%Make.decl, @Wrap(constants.%i32) [template = constants.%Make.2]
// CHECK:STDOUT:   %Make.ref.loc18: %Make.type.2 = name_ref Make, %.loc18_25 [template = constants.%Make.2]
// CHECK:STDOUT:   %Make.specific_fn.loc18: <specific function> = specific_function %Make.ref.loc18, @Make(constants.%i32) [template = constants.%Make.specific_fn.2]
// CHECK:STDOUT:   %Make.call.loc18: init %i32 = call %Make.specific_fn.loc18()
// CHECK:STDOUT:   assign %a.var, %Make.call.loc18
// CHECK:STDOUT:   %.loc19_11.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc19_11.2: type = converted %.loc19_11.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %b.var: ref %empty_tuple.type = var b
// CHECK:STDOUT:   %b: ref %empty_tuple.type = bind_name b, %b.var
// CHECK:STDOUT:   %Wrap.ref.loc19: %Wrap.type = name_ref Wrap, file.%Wrap.decl [template = constants.%Wrap.generic]
// CHECK:STDOUT:   %.loc19_21: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc19_22: type = converted %.loc19_21, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Wrap.loc19: type = class_type @Wrap, @Wrap(constants.%empty_tuple.type) [template = constants.%Wrap.3]
// CHECK:STDOUT:   %.loc19_23: %Make.type.3 = specific_constant @Wrap.%Make.decl, @Wrap(constants.%empty_tuple.type) [template = constants.%Make.3]
// CHECK:STDOUT:   %Make.ref.loc19: %Make.type.3 = name_ref Make, %.loc19_23 [template = constants.%Make.3]
// CHECK:STDOUT:   %Make.specific_fn.loc19: <specific function> = specific_function %Make.ref.loc19, @Make(constants.%empty_tuple.type) [template = constants.%Make.specific_fn.3]
// CHECK:STDOUT:   %Make.call.loc19: init %empty_tuple.type = call %Make.specific_fn.loc19()
// CHECK:STDOUT:   assign %b.var, %Make.call.loc19
// CHECK:STDOUT:   %C.ref.loc20_10: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   %Wrap.ref.loc20: %Wrap.type = name_ref Wrap, file.%Wrap.decl [template = constants.%Wrap.generic]
// CHECK:STDOUT:   %C.ref.loc20_19: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %Wrap.loc20: type = class_type @Wrap, @Wrap(constants.%C) [template = constants.%Wrap.4]
// CHECK:STDOUT:   %.loc20_21: %Make.type.4 = specific_constant @Wrap.%Make.decl, @Wrap(constants.%C) [template = constants.%Make.4]
// CHECK:STDOUT:   %Make.ref.loc20: %Make.type.4 = name_ref Make, %.loc20_21 [template = constants.%Make.4]
// CHECK:STDOUT:   %Make.specific_fn.loc20: <specific function> = specific_function %Make.ref.loc20, @Make(constants.%C) [template = constants.%Make.specific_fn.4]
// CHECK:STDOUT:   %.loc20_7: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %Make.call.loc20: init %C = call %Make.specific_fn.loc20() to %.loc20_7
// CHECK:STDOUT:   assign %c.var, %Make.call.loc20
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%T) {
// CHECK:STDOUT:   %T.loc11_12.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_12.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.1
// CHECK:STDOUT:   %Make => constants.%Make.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.1
// CHECK:STDOUT:   %Make => constants.%Make.1
// CHECK:STDOUT:   %Make.specific_fn.loc12_27.2 => constants.%Make.specific_fn.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(%T.loc11_12.2) {
// CHECK:STDOUT:   %T.loc11_12.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_12.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(@Make.%T) {
// CHECK:STDOUT:   %T.loc11_12.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_12.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%i32) {
// CHECK:STDOUT:   %T.loc11_12.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc11_12.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.2
// CHECK:STDOUT:   %Make => constants.%Make.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.2
// CHECK:STDOUT:   %Make => constants.%Make.2
// CHECK:STDOUT:   %Make.specific_fn.loc12_27.2 => constants.%Make.specific_fn.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T.loc11_12.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %T.patt.loc11_12.2 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.3
// CHECK:STDOUT:   %Make => constants.%Make.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.3
// CHECK:STDOUT:   %Make => constants.%Make.3
// CHECK:STDOUT:   %Make.specific_fn.loc12_27.2 => constants.%Make.specific_fn.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%C) {
// CHECK:STDOUT:   %T.loc11_12.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc11_12.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.4
// CHECK:STDOUT:   %Make => constants.%Make.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type.4
// CHECK:STDOUT:   %Make => constants.%Make.4
// CHECK:STDOUT:   %Make.specific_fn.loc12_27.2 => constants.%Make.specific_fn.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
