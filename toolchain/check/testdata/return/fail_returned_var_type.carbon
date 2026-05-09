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
  // CHECK:STDERR: fail_returned_var_type.carbon:[[@LINE-4]]:18: note: return type of function is `i32` [ReturnTypeHereNote]
  // CHECK:STDERR: fn Mismatch() -> i32 {
  // CHECK:STDERR:                  ^~~
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
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Mismatch.type: type = fn_type @Mismatch [concrete]
// CHECK:STDOUT:   %Mismatch: %Mismatch.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %f64.794: type = float_type %int_64, f64 [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %f64.d77 [concrete]
// CHECK:STDOUT:   %float.1f7: Core.FloatLiteral = float_literal_value 0e-1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.4a8: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.d77)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.823: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.39d: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.823 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f7c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.32a: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.6fd: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.32a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.4a8 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.f7c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e0d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f64.d77, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.9a0: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e0d, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.6fd [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.6fd, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float.1f7, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.0a8: %f64.d77 = float_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc23_12.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.1df: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.823) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.39d)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.1df), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15: Core.Form = init_form %i32 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Mismatch() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.0ae = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.0ae = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %f64.d77 = var %v.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 0e-1 [concrete = constants.%float.1f7]
// CHECK:STDOUT:   %impl.elem0: %.9a0 = impl_witness_access constants.%ImplicitAs.impl_witness.f7c, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.6fd]
// CHECK:STDOUT:   %bound_method.loc23_12.1: <bound method> = bound_method %float, %impl.elem0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_12.2: <bound method> = bound_method %float, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f64.d77 = call %bound_method.loc23_12.2(%float) [concrete = constants.%float.0a8]
// CHECK:STDOUT:   %.loc23_12: init %f64.d77 = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.0a8]
// CHECK:STDOUT:   assign %v.var, %.loc23_12
// CHECK:STDOUT:   %f64: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %v: ref %f64.d77 = ref_binding v, %v.var
// CHECK:STDOUT:   %.loc23_17: %f64.d77 = acquire_value %v
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %v.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%v.var)
// CHECK:STDOUT:   return %.loc23_17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_12.1(%self.param: ref %f64.794) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_12.2(%self.param: ref %f64.d77) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
