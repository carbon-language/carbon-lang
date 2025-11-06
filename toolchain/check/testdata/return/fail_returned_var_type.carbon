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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/fail_returned_var_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/fail_returned_var_type.carbon

fn Mismatch() -> i32 {
  // CHECK:STDERR: fail_returned_var_type.carbon:[[@LINE+7]]:19: error: type `f64` of `returned var` does not match return type of enclosing function [ReturnedVarWrongType]
  // CHECK:STDERR:   returned var v: f64 = 0.0;
  // CHECK:STDERR:                   ^~~
  // CHECK:STDERR: fail_returned_var_type.carbon:[[@LINE-4]]:15: note: return type of function is `i32` [ReturnTypeHereNote]
  // CHECK:STDERR: fn Mismatch() -> i32 {
  // CHECK:STDERR:               ^~~~~~
  // CHECK:STDERR:
  returned var v: f64 = 0.0;
  return var;
}

// CHECK:STDOUT: --- fail_returned_var_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Mismatch.type: type = fn_type @Mismatch [concrete]
// CHECK:STDOUT:   %Mismatch: %Mismatch.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %f64.d77 [concrete]
// CHECK:STDOUT:   %float.1f7: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.73d: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.d77)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.726: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%f64.d77) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.528: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.98e: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.528 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9db: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.72b: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d57: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.72b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.73d = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.9db) [concrete]
// CHECK:STDOUT:   %.c02: type = fn_type_with_self_type %ImplicitAs.Convert.type.726, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d57 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d57, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.0a8: %f64.d77 = float_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %f64.d77, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d28: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.f06: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d28 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.f06, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/parts/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.f5b: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.528) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.98e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.f5b), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Mismatch = %Mismatch.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Mismatch.decl: %Mismatch.type = fn_decl @Mismatch [concrete = constants.%Mismatch] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Mismatch() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.0ae = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.0ae = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %f64.d77 = var %v.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %impl.elem0: %.c02 = impl_witness_access constants.%ImplicitAs.impl_witness.9db, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.d57]
// CHECK:STDOUT:   %bound_method.loc23_12.1: <bound method> = bound_method %float, %impl.elem0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_12.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f64.d77 = call %bound_method.loc23_12.2(%float) [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc23_12: init %f64.d77 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.0a8]
// CHECK:STDOUT:   assign %v.var, %.loc23_12
// CHECK:STDOUT:   %.loc23_19: type = splice_block %f64 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %f64.d77 = ref_binding v, %v.var
// CHECK:STDOUT:   %.loc23_16: %f64.d77 = acquire_value %v
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %v.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.f06
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.f06, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_12.3: <bound method> = bound_method %v.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc23_12.3(%v.var)
// CHECK:STDOUT:   return %.loc23_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
