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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/fail_return_with_returned_var.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/fail_return_with_returned_var.carbon

fn F() -> i32 {
  returned var v: i32 = 0;
  // CHECK:STDERR: fail_return_with_returned_var.carbon:[[@LINE+7]]:3: error: can only `return var;` in the scope of a `returned var` [ReturnExprWithReturnedVar]
  // CHECK:STDERR:   return 1;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_return_with_returned_var.carbon:[[@LINE-4]]:16: note: `returned var` was declared here [ReturnedVarHere]
  // CHECK:STDERR:   returned var v: i32 = 0;
  // CHECK:STDERR:                ^~~~~~
  // CHECK:STDERR:
  return 1;
}

class C { var a: i32; var b: i32; }
fn G() -> C {
  returned var c: C = {.a = 1, .b = 2};
  // CHECK:STDERR: fail_return_with_returned_var.carbon:[[@LINE+7]]:3: error: can only `return var;` in the scope of a `returned var` [ReturnExprWithReturnedVar]
  // CHECK:STDERR:   return c;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_return_with_returned_var.carbon:[[@LINE-4]]:16: note: `returned var` was declared here [ReturnedVarHere]
  // CHECK:STDERR:   returned var c: C = {.a = 1, .b = 2};
  // CHECK:STDERR:                ^~~~
  // CHECK:STDERR:
  return c;
}

// CHECK:STDOUT: --- fail_return_with_returned_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.7ea: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.104: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.754: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.754 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.360: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3d6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0bd, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.360 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3d6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.474: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da5, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.638: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.82d: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.62d: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc16_12.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.b47: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.abf: <witness> = complete_type_witness %struct_type.a.b.b47 [concrete]
// CHECK:STDOUT:   %.8f0: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.0d5: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.cfd = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c1a: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %bound_method.e5b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d9c: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.bf7: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %bound_method.7d5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.228: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%int_1.d9c, %int_2.228) [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.754 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.b42: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0bd = impl_witness_table (%Core.import_ref.b42), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.104 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.104 = return_slot_pattern %return.param_patt, %i32.loc15 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15: Core.Form = init_form %i32.loc15 [concrete = constants.%.7ea]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.0d5 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.0d5 = return_slot_pattern %return.param_patt, %C.ref.loc28 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc28: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc28: Core.Form = init_form %C.ref.loc28 [concrete = constants.%.8f0]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc27_16: %C.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %.loc27_28: %C.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b.b47 [concrete = constants.%complete_type.abf]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .a = %.loc27_16
// CHECK:STDOUT:   .b = %.loc27_28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.104 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.104 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %i32 = var %v.var_patt
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc16_12.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.638]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_12.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.82d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc16_12.2(%int_0) [concrete = constants.%int_0.62d]
// CHECK:STDOUT:   %.loc16: init %i32 = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.62d]
// CHECK:STDOUT:   assign %v.var, %.loc16
// CHECK:STDOUT:   %i32.loc16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %v: ref %i32 = ref_binding v, %v.var
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %v.var, constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%v.var)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc16_12.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc16_12.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.0d5 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.0d5 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc29_38.1: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2) [concrete = constants.%struct]
// CHECK:STDOUT:   %impl.elem0.loc29_38.1: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc29_38.1: <bound method> = bound_method %int_1, %impl.elem0.loc29_38.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c1a]
// CHECK:STDOUT:   %specific_fn.loc29_38.1: <specific function> = specific_function %impl.elem0.loc29_38.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc29_38.2: <bound method> = bound_method %int_1, %specific_fn.loc29_38.1 [concrete = constants.%bound_method.e5b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc29_38.1: init %i32 = call %bound_method.loc29_38.2(%int_1) [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %.loc29_38.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc29_38.1 [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %.loc29_38.3: ref %i32 = class_element_access %return.param, element0
// CHECK:STDOUT:   %.loc29_38.4: init %i32 to %.loc29_38.3 = in_place_init %.loc29_38.2 [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %impl.elem0.loc29_38.2: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc29_38.3: <bound method> = bound_method %int_2, %impl.elem0.loc29_38.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.bf7]
// CHECK:STDOUT:   %specific_fn.loc29_38.2: <specific function> = specific_function %impl.elem0.loc29_38.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc29_38.4: <bound method> = bound_method %int_2, %specific_fn.loc29_38.2 [concrete = constants.%bound_method.7d5]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc29_38.2: init %i32 = call %bound_method.loc29_38.4(%int_2) [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %.loc29_38.5: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc29_38.2 [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %.loc29_38.6: ref %i32 = class_element_access %return.param, element1
// CHECK:STDOUT:   %.loc29_38.7: init %i32 to %.loc29_38.6 = in_place_init %.loc29_38.5 [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %.loc29_38.8: init %C to %return.param = class_init (%.loc29_38.4, %.loc29_38.7) [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc29_12: init %C = converted %.loc29_38.1, %.loc29_38.8 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %return.param, %.loc29_12
// CHECK:STDOUT:   %C.ref.loc29: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %return.param
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
