// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/returned_var_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/returned_var_scope.carbon

fn UnrelatedScopes() -> i32 {
  if (true) {
    returned var v: i32 = 0;
  }
  if (true) {
    returned var w: i32 = 1;
  }
  return 0;
}

fn EnclosingButAfter(b: bool) -> i32 {
  if (b) {
    returned var v: i32 = 0;
    return var;
  }
  returned var w: i32 = 1;
  return var;
}

// CHECK:STDOUT: --- returned_var_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %UnrelatedScopes.type: type = fn_type @UnrelatedScopes [concrete]
// CHECK:STDOUT:   %UnrelatedScopes: %UnrelatedScopes.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @ImplicitAs.impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.d04: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.a17: type = fn_type @Op.3, @Destroy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Op.e6a: %Op.type.a17 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.e6a, @Op.3(%i32) [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %EnclosingButAfter.type: type = fn_type @EnclosingButAfter [concrete]
// CHECK:STDOUT:   %EnclosingButAfter: %EnclosingButAfter.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @ImplicitAs.impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @ImplicitAs.impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @ImplicitAs.impl.4f9 [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .UnrelatedScopes = %UnrelatedScopes.decl
// CHECK:STDOUT:     .EnclosingButAfter = %EnclosingButAfter.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %UnrelatedScopes.decl: %UnrelatedScopes.type = fn_decl @UnrelatedScopes [concrete = constants.%UnrelatedScopes] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EnclosingButAfter.decl: %EnclosingButAfter.type = fn_decl @EnclosingButAfter [concrete = constants.%EnclosingButAfter] {
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc25_25.1: type = splice_block %.loc25_25.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc25_25.2: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:       %.loc25_25.3: type = converted %bool.make_type, %.loc25_25.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UnrelatedScopes() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true.loc16: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true.loc16 br !if.then.loc16 else br !if.else.loc16
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc16:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.7ce = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.7ce = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %i32 = var %v.var_patt
// CHECK:STDOUT:   %int_0.loc17: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc17: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc17_14.1: <bound method> = bound_method %int_0.loc17, %impl.elem0.loc17 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_14.2: <bound method> = bound_method %int_0.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc17: init %i32 = call %bound_method.loc17_14.2(%int_0.loc17) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc17_14: init %i32 = converted %int_0.loc17, %int.convert_checked.loc17 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   assign %v.var, %.loc17_14
// CHECK:STDOUT:   %.loc17_21: type = splice_block %i32.loc17 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %i32 = bind_name v, %v.var
// CHECK:STDOUT:   br !if.else.loc16
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc16:
// CHECK:STDOUT:   %true.loc19: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true.loc19 br !if.then.loc19 else br !if.else.loc19
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc19:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type.7ce = binding_pattern w [concrete]
// CHECK:STDOUT:     %w.var_patt: %pattern_type.7ce = var_pattern %w.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.var: ref %i32 = var %w.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc20: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc20_14.1: <bound method> = bound_method %int_1, %impl.elem0.loc20 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_14.2: <bound method> = bound_method %int_1, %specific_fn.loc20 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc20: init %i32 = call %bound_method.loc20_14.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc20_14: init %i32 = converted %int_1, %int.convert_checked.loc20 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   assign %w.var, %.loc20_14
// CHECK:STDOUT:   %.loc20_21: type = splice_block %i32.loc20 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc20: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc20: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w: ref %i32 = bind_name w, %w.var
// CHECK:STDOUT:   br !if.else.loc19
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc19:
// CHECK:STDOUT:   %int_0.loc22: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc22: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc22_11.1: <bound method> = bound_method %int_0.loc22, %impl.elem0.loc22 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem0.loc22, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_11.2: <bound method> = bound_method %int_0.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc22: init %i32 = call %bound_method.loc22_11.2(%int_0.loc22) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc22_11.1: %i32 = value_of_initializer %int.convert_checked.loc22 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc22_11.2: %i32 = converted %int_0.loc22, %.loc22_11.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %Op.bound.loc20: <bound method> = bound_method %w.var, constants.%Op.e6a
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.e6a, @Op.3(constants.%i32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_14.3: <bound method> = bound_method %w.var, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc20: %ptr.235 = addr_of %w.var
// CHECK:STDOUT:   %no_op.loc20: init %empty_tuple.type = call %bound_method.loc20_14.3(%addr.loc20)
// CHECK:STDOUT:   %Op.bound.loc17: <bound method> = bound_method %v.var, constants.%Op.e6a
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.e6a, @Op.3(constants.%i32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_14.3: <bound method> = bound_method %v.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc17: %ptr.235 = addr_of %v.var
// CHECK:STDOUT:   %no_op.loc17: init %empty_tuple.type = call %bound_method.loc17_14.3(%addr.loc17)
// CHECK:STDOUT:   return %.loc22_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @EnclosingButAfter(%b.param: bool) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.7ce = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.7ce = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %i32 = var %v.var_patt
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc27: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc27_14.1: <bound method> = bound_method %int_0, %impl.elem0.loc27 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_14.2: <bound method> = bound_method %int_0, %specific_fn.loc27 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc27: init %i32 = call %bound_method.loc27_14.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc27_14: init %i32 = converted %int_0, %int.convert_checked.loc27 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   assign %v.var, %.loc27_14
// CHECK:STDOUT:   %.loc27_21: type = splice_block %i32.loc27 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc27_18: %i32 = bind_value %v
// CHECK:STDOUT:   %Op.bound.loc27_14.1: <bound method> = bound_method %v.var, constants.%Op.e6a
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.e6a, @Op.3(constants.%i32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_14.3: <bound method> = bound_method %v.var, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc27_14.1: %ptr.235 = addr_of %v.var
// CHECK:STDOUT:   %no_op.loc27_14.1: init %empty_tuple.type = call %bound_method.loc27_14.3(%addr.loc27_14.1)
// CHECK:STDOUT:   return %.loc27_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type.7ce = binding_pattern w [concrete]
// CHECK:STDOUT:     %w.var_patt: %pattern_type.7ce = var_pattern %w.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.var: ref %i32 = var %w.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc30: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc30_12.1: <bound method> = bound_method %int_1, %impl.elem0.loc30 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc30: <specific function> = specific_function %impl.elem0.loc30, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc30_12.2: <bound method> = bound_method %int_1, %specific_fn.loc30 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc30: init %i32 = call %bound_method.loc30_12.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc30_12: init %i32 = converted %int_1, %int.convert_checked.loc30 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   assign %w.var, %.loc30_12
// CHECK:STDOUT:   %.loc30_19: type = splice_block %i32.loc30 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc30: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc30: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w: ref %i32 = bind_name w, %w.var
// CHECK:STDOUT:   %.loc30_16: %i32 = bind_value %w
// CHECK:STDOUT:   %Op.bound.loc30: <bound method> = bound_method %w.var, constants.%Op.e6a
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.e6a, @Op.3(constants.%i32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc30_12.3: <bound method> = bound_method %w.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc30: %ptr.235 = addr_of %w.var
// CHECK:STDOUT:   %no_op.loc30: init %empty_tuple.type = call %bound_method.loc30_12.3(%addr.loc30)
// CHECK:STDOUT:   %Op.bound.loc27_14.2: <bound method> = bound_method %v.var, constants.%Op.e6a
// CHECK:STDOUT:   %Op.specific_fn.3: <specific function> = specific_function constants.%Op.e6a, @Op.3(constants.%i32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_14.4: <bound method> = bound_method %v.var, %Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc27_14.2: %ptr.235 = addr_of %v.var
// CHECK:STDOUT:   %no_op.loc27_14.2: init %empty_tuple.type = call %bound_method.loc27_14.4(%addr.loc27_14.2)
// CHECK:STDOUT:   return %.loc30_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
