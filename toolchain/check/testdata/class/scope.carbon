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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/scope.carbon

class Class {
  fn F() -> i32 {
    return 1;
  }

  fn G() -> i32 {
    return F();
  }
}

fn F() -> i32 {
  return 2;
}

fn Run() {
  var a: i32 = F();
  var b: i32 = Class.F();
}

// CHECK:STDOUT: --- scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type.f1b: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.1f2: %F.type.f1b = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.3, @impl.49c(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.64e: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%i32) [concrete]
// CHECK:STDOUT:   %Op.type.a17: type = fn_type @Op.3, @impl.49c(%i32) [concrete]
// CHECK:STDOUT:   %Op.e6a: %Op.type.a17 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %i32, (%Destroy.impl_witness.64e) [concrete]
// CHECK:STDOUT:   %.ab7: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.e6a, @Op.3(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.49c.%Op.type (%Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_29, loaded [symbolic = @impl.49c.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl.49c [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.b25 = fn_decl @F.2 [concrete = constants.%F.c41] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type.f1b = fn_decl @F.1 [concrete = constants.%F.1f2] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc17_13.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_13.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc17_13.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc17_13.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc17_13.2: %i32 = converted %int_1, %.loc17_13.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   return %.loc17_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type.f1b = name_ref F, @Class.%F.decl [concrete = constants.%F.1f2]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref()
// CHECK:STDOUT:   %.loc21_15.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc21_15.2: %i32 = converted %F.call, %.loc21_15.1
// CHECK:STDOUT:   return %.loc21_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc26_11.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_11.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc26_11.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc26_11.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc26_11.2: %i32 = converted %int_2, %.loc26_11.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   return %.loc26_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.7ce = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %i32 = var %a.var_patt
// CHECK:STDOUT:   %F.ref.loc30: %F.type.b25 = name_ref F, file.%F.decl [concrete = constants.%F.c41]
// CHECK:STDOUT:   %F.call.loc30: init %i32 = call %F.ref.loc30()
// CHECK:STDOUT:   assign %a.var, %F.call.loc30
// CHECK:STDOUT:   %.loc30: type = splice_block %i32.loc30 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc30: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc30: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %i32 = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.7ce = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var %b.var_patt
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %F.ref.loc31: %F.type.f1b = name_ref F, @Class.%F.decl [concrete = constants.%F.1f2]
// CHECK:STDOUT:   %F.call.loc31: init %i32 = call %F.ref.loc31()
// CHECK:STDOUT:   assign %b.var, %F.call.loc31
// CHECK:STDOUT:   %.loc31: type = splice_block %i32.loc31 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc31: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc31: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   %impl.elem0.loc31: %.ab7 = impl_witness_access constants.%Destroy.impl_witness.64e, element0 [concrete = constants.%Op.e6a]
// CHECK:STDOUT:   %bound_method.loc31_3.1: <bound method> = bound_method %b.var, %impl.elem0.loc31
// CHECK:STDOUT:   %specific_fn.loc31: <specific function> = specific_function %impl.elem0.loc31, @Op.3(constants.%i32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc31_3.2: <bound method> = bound_method %b.var, %specific_fn.loc31
// CHECK:STDOUT:   %addr.loc31: %ptr.235 = addr_of %b.var
// CHECK:STDOUT:   %no_op.loc31: init %empty_tuple.type = call %bound_method.loc31_3.2(%addr.loc31)
// CHECK:STDOUT:   %impl.elem0.loc30: %.ab7 = impl_witness_access constants.%Destroy.impl_witness.64e, element0 [concrete = constants.%Op.e6a]
// CHECK:STDOUT:   %bound_method.loc30_3.1: <bound method> = bound_method %a.var, %impl.elem0.loc30
// CHECK:STDOUT:   %specific_fn.loc30: <specific function> = specific_function %impl.elem0.loc30, @Op.3(constants.%i32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc30_3.2: <bound method> = bound_method %a.var, %specific_fn.loc30
// CHECK:STDOUT:   %addr.loc30: %ptr.235 = addr_of %a.var
// CHECK:STDOUT:   %no_op.loc30: init %empty_tuple.type = call %bound_method.loc30_3.2(%addr.loc30)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
