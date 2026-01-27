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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/fail_returned_var_shadow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/fail_returned_var_shadow.carbon

fn SameScope() -> i32 {
  if (true) {
    returned var v: i32 = 0;
    // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE+11]]:18: error: cannot declare a `returned var` in the scope of another `returned var` declaration [ReturnedVarShadowed]
    // CHECK:STDERR:     returned var w: i32 = 1;
    // CHECK:STDERR:                  ^
    // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE-4]]:18: note: `returned var` was declared here [ReturnedVarHere]
    // CHECK:STDERR:     returned var v: i32 = 0;
    // CHECK:STDERR:                  ^
    // CHECK:STDERR:
    // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE+4]]:18: warning: binding `w` unused [UnusedBinding]
    // CHECK:STDERR:     returned var w: i32 = 1;
    // CHECK:STDERR:                  ^
    // CHECK:STDERR:
    returned var w: i32 = 1;
  }
  return 0;
}

fn DifferentScopes() -> i32 {
  if (true) {
    returned var v: i32 = 0;
    if (true) {
      // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE+11]]:20: error: cannot declare a `returned var` in the scope of another `returned var` declaration [ReturnedVarShadowed]
      // CHECK:STDERR:       returned var w: i32 = 1;
      // CHECK:STDERR:                    ^
      // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE-5]]:18: note: `returned var` was declared here [ReturnedVarHere]
      // CHECK:STDERR:     returned var v: i32 = 0;
      // CHECK:STDERR:                  ^
      // CHECK:STDERR:
      // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE+4]]:20: warning: binding `w` unused [UnusedBinding]
      // CHECK:STDERR:       returned var w: i32 = 1;
      // CHECK:STDERR:                    ^
      // CHECK:STDERR:
      returned var w: i32 = 1;
    }
  }
  return 0;
}

// CHECK:STDOUT: --- fail_returned_var_shadow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %SameScope.type: type = fn_type @SameScope [concrete]
// CHECK:STDOUT:   %SameScope: %SameScope.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.740: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.57f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8c3, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.57f) [concrete]
// CHECK:STDOUT:   %.dbf: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4b5: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.6f8: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.3d4: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0 [concrete]
// CHECK:STDOUT:   %bound_method.d3a: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %i32, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.9df: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.ae2: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.9df = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.ae2, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT:   %DifferentScopes.type: type = fn_type @DifferentScopes [concrete]
// CHECK:STDOUT:   %DifferentScopes: %DifferentScopes.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.1b5: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.740)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8c3 = impl_witness_table (%Core.import_ref.1b5), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .SameScope = %SameScope.decl
// CHECK:STDOUT:     .DifferentScopes = %DifferentScopes.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %SameScope.decl: %SameScope.type = fn_decl @SameScope [concrete = constants.%SameScope] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DifferentScopes.decl: %DifferentScopes.type = fn_decl @DifferentScopes [concrete = constants.%DifferentScopes] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc34: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc34: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SameScope() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.7ce = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.7ce = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %i32 = var %v.var_patt
// CHECK:STDOUT:   %int_0.loc17: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc17: %.dbf = impl_witness_access constants.%ImplicitAs.impl_witness.57f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc17_14.1: <bound method> = bound_method %int_0.loc17, %impl.elem0.loc17 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4b5]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_14.2: <bound method> = bound_method %int_0.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.6f8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %i32 = call %bound_method.loc17_14.2(%int_0.loc17) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc17_14: init %i32 = converted %int_0.loc17, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   assign %v.var, %.loc17_14
// CHECK:STDOUT:   %.loc17_21: type = splice_block %i32.loc17 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %i32 = ref_binding v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type.7ce = ref_binding_pattern w [concrete]
// CHECK:STDOUT:     %w.var_patt: %pattern_type.7ce = var_pattern %w.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.var: ref %i32 = var %w.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc29: %.dbf = impl_witness_access constants.%ImplicitAs.impl_witness.57f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc29_14.1: <bound method> = bound_method %int_1, %impl.elem0.loc29 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.3d4]
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem0.loc29, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc29_14.2: <bound method> = bound_method %int_1, %specific_fn.loc29 [concrete = constants.%bound_method.d3a]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc29: init %i32 = call %bound_method.loc29_14.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc29_14: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc29 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   assign %w.var, %.loc29_14
// CHECK:STDOUT:   %.loc29_21: type = splice_block %i32.loc29 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc29: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc29: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w: ref %i32 = ref_binding w, %w.var
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %int_0.loc31: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc31: %.dbf = impl_witness_access constants.%ImplicitAs.impl_witness.57f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc31_11.1: <bound method> = bound_method %int_0.loc31, %impl.elem0.loc31 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4b5]
// CHECK:STDOUT:   %specific_fn.loc31: <specific function> = specific_function %impl.elem0.loc31, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc31_11.2: <bound method> = bound_method %int_0.loc31, %specific_fn.loc31 [concrete = constants.%bound_method.6f8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc31: init %i32 = call %bound_method.loc31_11.2(%int_0.loc31) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc31: init %i32 = converted %int_0.loc31, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc31 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc29: <bound method> = bound_method %w.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ae2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ae2, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc29_14.3: <bound method> = bound_method %w.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc29: init %empty_tuple.type = call %bound_method.loc29_14.3(%w.var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc17: <bound method> = bound_method %v.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ae2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ae2, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_14.3: <bound method> = bound_method %v.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc17: init %empty_tuple.type = call %bound_method.loc17_14.3(%v.var)
// CHECK:STDOUT:   return %.loc31 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DifferentScopes() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true.loc35: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true.loc35 br !if.then.loc35 else br !if.else.loc35
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc35:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.7ce = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.7ce = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %i32 = var %v.var_patt
// CHECK:STDOUT:   %int_0.loc36: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc36: %.dbf = impl_witness_access constants.%ImplicitAs.impl_witness.57f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc36_14.1: <bound method> = bound_method %int_0.loc36, %impl.elem0.loc36 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4b5]
// CHECK:STDOUT:   %specific_fn.loc36: <specific function> = specific_function %impl.elem0.loc36, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc36_14.2: <bound method> = bound_method %int_0.loc36, %specific_fn.loc36 [concrete = constants.%bound_method.6f8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc36: init %i32 = call %bound_method.loc36_14.2(%int_0.loc36) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc36_14: init %i32 = converted %int_0.loc36, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc36 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   assign %v.var, %.loc36_14
// CHECK:STDOUT:   %.loc36_21: type = splice_block %i32.loc36 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc36: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc36: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %i32 = ref_binding v, %v.var
// CHECK:STDOUT:   %true.loc37: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true.loc37 br !if.then.loc37 else br !if.else.loc37
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc37:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type.7ce = ref_binding_pattern w [concrete]
// CHECK:STDOUT:     %w.var_patt: %pattern_type.7ce = var_pattern %w.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.var: ref %i32 = var %w.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc49: %.dbf = impl_witness_access constants.%ImplicitAs.impl_witness.57f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc49_16.1: <bound method> = bound_method %int_1, %impl.elem0.loc49 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.3d4]
// CHECK:STDOUT:   %specific_fn.loc49: <specific function> = specific_function %impl.elem0.loc49, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc49_16.2: <bound method> = bound_method %int_1, %specific_fn.loc49 [concrete = constants.%bound_method.d3a]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc49: init %i32 = call %bound_method.loc49_16.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc49_16: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc49 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   assign %w.var, %.loc49_16
// CHECK:STDOUT:   %.loc49_23: type = splice_block %i32.loc49 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc49: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc49: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w: ref %i32 = ref_binding w, %w.var
// CHECK:STDOUT:   br !if.else.loc37
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc37:
// CHECK:STDOUT:   br !if.else.loc35
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc35:
// CHECK:STDOUT:   %int_0.loc52: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc52: %.dbf = impl_witness_access constants.%ImplicitAs.impl_witness.57f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc52_11.1: <bound method> = bound_method %int_0.loc52, %impl.elem0.loc52 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4b5]
// CHECK:STDOUT:   %specific_fn.loc52: <specific function> = specific_function %impl.elem0.loc52, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc52_11.2: <bound method> = bound_method %int_0.loc52, %specific_fn.loc52 [concrete = constants.%bound_method.6f8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc52: init %i32 = call %bound_method.loc52_11.2(%int_0.loc52) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc52: init %i32 = converted %int_0.loc52, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc52 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc49: <bound method> = bound_method %w.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ae2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ae2, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc49_16.3: <bound method> = bound_method %w.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc49: init %empty_tuple.type = call %bound_method.loc49_16.3(%w.var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc36: <bound method> = bound_method %v.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ae2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ae2, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc36_14.3: <bound method> = bound_method %v.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc36: init %empty_tuple.type = call %bound_method.loc36_14.3(%v.var)
// CHECK:STDOUT:   return %.loc52 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
