// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/fail_expr_category.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/fail_expr_category.carbon

fn F() -> array(i32, 3);

fn G(b: array(i32, 3)) {
  // Indexing an array value gives a value.
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+4]]:18: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   var pb: i32* = &b[0];
  // CHECK:STDERR:                  ^
  // CHECK:STDERR:
  var pb: i32* = &b[0];
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   b[0] = 4;
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  b[0] = 4;

  // Indexing an ephemeral reference (materialized from an initializing
  // expression) gives a value.
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+4]]:18: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   var pf: i32* = &F()[0];
  // CHECK:STDERR:                  ^
  // CHECK:STDERR:
  var pf: i32* = &F()[0];
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   F()[0] = 4;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  F()[0] = 4;
}

// CHECK:STDOUT: --- fail_expr_category.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (imports.%Core.import_ref.0b9), @impl.49c [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (imports.%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.711: <witness> = impl_witness %Destroy.impl_witness_table, @impl.49c(%ptr.235) [concrete]
// CHECK:STDOUT:   %Op.type.ba2: type = fn_type @Op.2, @impl.49c(%ptr.235) [concrete]
// CHECK:STDOUT:   %Op.649: %Op.type.ba2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.2bf: %Destroy.type = facet_value %ptr.235, (%Destroy.impl_witness.711) [concrete]
// CHECK:STDOUT:   %.a79: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.2bf [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness %ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.d04: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.1da: <bound method> = bound_method %int_4.0c1, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.14a: <witness> = impl_witness %Destroy.impl_witness_table, @impl.49c(%array_type) [concrete]
// CHECK:STDOUT:   %Op.type.1e4: type = fn_type @Op.2, @impl.49c(%array_type) [concrete]
// CHECK:STDOUT:   %Op.f0b: %Op.type.1e4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.75f: %Destroy.type = facet_value %array_type, (%Destroy.impl_witness.14a) [concrete]
// CHECK:STDOUT:   %.9c9: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.75f [concrete]
// CHECK:STDOUT:   %Op.specific_fn.c51: <specific function> = specific_function %Op.f0b, @Op.2(%array_type) [concrete]
// CHECK:STDOUT:   %Op.specific_fn.829: <specific function> = specific_function %Op.649, @Op.2(%ptr.235) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %array_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %array_type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:     %return.param: ref %array_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %array_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %b.patt: %array_type = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %array_type = value_param_pattern %b.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %array_type = value_param call_param0
// CHECK:STDOUT:     %.loc13: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:       %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:       %array_type: type = array_type %int_3, %i32.loc13 [concrete = constants.%array_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %array_type = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return.param: %array_type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b.param: %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pb.patt: %ptr.235 = binding_pattern pb
// CHECK:STDOUT:     %.loc19_3.1: %ptr.235 = var_pattern %pb.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pb.var: ref %ptr.235 = var pb
// CHECK:STDOUT:   %impl.elem0.loc19_3: %.a79 = impl_witness_access constants.%Destroy.impl_witness.711, element0 [concrete = constants.%Op.649]
// CHECK:STDOUT:   %bound_method.loc19_3: <bound method> = bound_method %pb.var, %impl.elem0.loc19_3
// CHECK:STDOUT:   %b.ref.loc19: %array_type = name_ref b, %b
// CHECK:STDOUT:   %int_0.loc19: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc19_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc19_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc19_21: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc19_21.1: <bound method> = bound_method %int_0.loc19, %impl.elem0.loc19_21 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc19_21: <specific function> = specific_function %impl.elem0.loc19_21, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19_21.2: <bound method> = bound_method %int_0.loc19, %specific_fn.loc19_21 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc19: init %i32 = call %bound_method.loc19_21.2(%int_0.loc19) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc19_21.1: %i32 = value_of_initializer %int.convert_checked.loc19 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc19_21.2: %i32 = converted %int_0.loc19, %.loc19_21.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc19_22.1: ref %array_type = value_as_ref %b.ref.loc19
// CHECK:STDOUT:   %.loc19_22.2: ref %i32 = array_index %.loc19_22.1, %.loc19_21.2
// CHECK:STDOUT:   %.loc19_22.3: %i32 = bind_value %.loc19_22.2
// CHECK:STDOUT:   %addr.loc19: %ptr.235 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   assign %pb.var, %addr.loc19
// CHECK:STDOUT:   %.loc19_14: type = splice_block %ptr.loc19 [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc19_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc19_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc19: type = ptr_type %i32.loc19_11 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pb: ref %ptr.235 = bind_name pb, %pb.var
// CHECK:STDOUT:   %b.ref.loc24: %array_type = name_ref b, %b
// CHECK:STDOUT:   %int_0.loc24: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc24_5: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc24_5.1: <bound method> = bound_method %int_0.loc24, %impl.elem0.loc24_5 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc24_5: <specific function> = specific_function %impl.elem0.loc24_5, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_5.2: <bound method> = bound_method %int_0.loc24, %specific_fn.loc24_5 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc24_5: init %i32 = call %bound_method.loc24_5.2(%int_0.loc24) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc24_5.1: %i32 = value_of_initializer %int.convert_checked.loc24_5 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc24_5.2: %i32 = converted %int_0.loc24, %.loc24_5.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc24_6.1: ref %array_type = value_as_ref %b.ref.loc24
// CHECK:STDOUT:   %.loc24_6.2: ref %i32 = array_index %.loc24_6.1, %.loc24_5.2
// CHECK:STDOUT:   %.loc24_6.3: %i32 = bind_value %.loc24_6.2
// CHECK:STDOUT:   %int_4.loc24: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc24_8: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc24_8.1: <bound method> = bound_method %int_4.loc24, %impl.elem0.loc24_8 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc24_8: <specific function> = specific_function %impl.elem0.loc24_8, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_8.2: <bound method> = bound_method %int_4.loc24, %specific_fn.loc24_8 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc24_8: init %i32 = call %bound_method.loc24_8.2(%int_4.loc24) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc24_8: init %i32 = converted %int_4.loc24, %int.convert_checked.loc24_8 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc24_6.3, %.loc24_8
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pf.patt: %ptr.235 = binding_pattern pf
// CHECK:STDOUT:     %.loc32_3.1: %ptr.235 = var_pattern %pf.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pf.var: ref %ptr.235 = var pf
// CHECK:STDOUT:   %impl.elem0.loc32_3: %.a79 = impl_witness_access constants.%Destroy.impl_witness.711, element0 [concrete = constants.%Op.649]
// CHECK:STDOUT:   %bound_method.loc32_3: <bound method> = bound_method %pf.var, %impl.elem0.loc32_3
// CHECK:STDOUT:   %F.ref.loc32: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc32_21.1: ref %array_type = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc32_21: %.9c9 = impl_witness_access constants.%Destroy.impl_witness.14a, element0 [concrete = constants.%Op.f0b]
// CHECK:STDOUT:   %bound_method.loc32_21: <bound method> = bound_method %.loc32_21.1, %impl.elem0.loc32_21
// CHECK:STDOUT:   %F.call.loc32: init %array_type = call %F.ref.loc32() to %.loc32_21.1
// CHECK:STDOUT:   %int_0.loc32: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc32_21.2: ref %array_type = temporary %.loc32_21.1, %F.call.loc32
// CHECK:STDOUT:   %int_32.loc32_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc32_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc32_23: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc32_23.1: <bound method> = bound_method %int_0.loc32, %impl.elem0.loc32_23 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc32_23: <specific function> = specific_function %impl.elem0.loc32_23, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc32_23.2: <bound method> = bound_method %int_0.loc32, %specific_fn.loc32_23 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc32: init %i32 = call %bound_method.loc32_23.2(%int_0.loc32) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc32_23.1: %i32 = value_of_initializer %int.convert_checked.loc32 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc32_23.2: %i32 = converted %int_0.loc32, %.loc32_23.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc32_24.1: ref %i32 = array_index %.loc32_21.2, %.loc32_23.2
// CHECK:STDOUT:   %.loc32_24.2: %i32 = bind_value %.loc32_24.1
// CHECK:STDOUT:   %addr.loc32: %ptr.235 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   assign %pf.var, %addr.loc32
// CHECK:STDOUT:   %.loc32_14: type = splice_block %ptr.loc32 [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc32_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc32_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc32: type = ptr_type %i32.loc32_11 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pf: ref %ptr.235 = bind_name pf, %pf.var
// CHECK:STDOUT:   %F.ref.loc37: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc37_5.1: ref %array_type = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc37_5: %.9c9 = impl_witness_access constants.%Destroy.impl_witness.14a, element0 [concrete = constants.%Op.f0b]
// CHECK:STDOUT:   %bound_method.loc37_5: <bound method> = bound_method %.loc37_5.1, %impl.elem0.loc37_5
// CHECK:STDOUT:   %F.call.loc37: init %array_type = call %F.ref.loc37() to %.loc37_5.1
// CHECK:STDOUT:   %int_0.loc37: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc37_5.2: ref %array_type = temporary %.loc37_5.1, %F.call.loc37
// CHECK:STDOUT:   %int_32.loc37: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc37: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc37_7: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc37_7.1: <bound method> = bound_method %int_0.loc37, %impl.elem0.loc37_7 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc37_7: <specific function> = specific_function %impl.elem0.loc37_7, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_7.2: <bound method> = bound_method %int_0.loc37, %specific_fn.loc37_7 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc37_7: init %i32 = call %bound_method.loc37_7.2(%int_0.loc37) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc37_7.1: %i32 = value_of_initializer %int.convert_checked.loc37_7 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc37_7.2: %i32 = converted %int_0.loc37, %.loc37_7.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc37_8.1: ref %i32 = array_index %.loc37_5.2, %.loc37_7.2
// CHECK:STDOUT:   %.loc37_8.2: %i32 = bind_value %.loc37_8.1
// CHECK:STDOUT:   %int_4.loc37: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc37_10: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc37_10.1: <bound method> = bound_method %int_4.loc37, %impl.elem0.loc37_10 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc37_10: <specific function> = specific_function %impl.elem0.loc37_10, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_10.2: <bound method> = bound_method %int_4.loc37, %specific_fn.loc37_10 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc37_10: init %i32 = call %bound_method.loc37_10.2(%int_4.loc37) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc37_10: init %i32 = converted %int_4.loc37, %int.convert_checked.loc37_10 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc37_8.2, %.loc37_10
// CHECK:STDOUT:   %specific_fn.loc37_5: <specific function> = specific_function %impl.elem0.loc37_5, @Op.2(constants.%array_type) [concrete = constants.%Op.specific_fn.c51]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %.loc37_5.1, %specific_fn.loc37_5
// CHECK:STDOUT:   %.loc37_5.3: %array_type = bind_value %.loc37_5.1
// CHECK:STDOUT:   %no_op.1: init %empty_tuple.type = call %bound_method.1(%.loc37_5.3)
// CHECK:STDOUT:   %specific_fn.loc32_21: <specific function> = specific_function %impl.elem0.loc32_21, @Op.2(constants.%array_type) [concrete = constants.%Op.specific_fn.c51]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %.loc32_21.1, %specific_fn.loc32_21
// CHECK:STDOUT:   %.loc32_21.3: %array_type = bind_value %.loc32_21.1
// CHECK:STDOUT:   %no_op.2: init %empty_tuple.type = call %bound_method.2(%.loc32_21.3)
// CHECK:STDOUT:   %specific_fn.loc32_3: <specific function> = specific_function %impl.elem0.loc32_3, @Op.2(constants.%ptr.235) [concrete = constants.%Op.specific_fn.829]
// CHECK:STDOUT:   %bound_method.3: <bound method> = bound_method %pf.var, %specific_fn.loc32_3
// CHECK:STDOUT:   %.loc32_3.2: %ptr.235 = bind_value %pf.var
// CHECK:STDOUT:   %no_op.3: init %empty_tuple.type = call %bound_method.3(%.loc32_3.2)
// CHECK:STDOUT:   %specific_fn.loc19_3: <specific function> = specific_function %impl.elem0.loc19_3, @Op.2(constants.%ptr.235) [concrete = constants.%Op.specific_fn.829]
// CHECK:STDOUT:   %bound_method.4: <bound method> = bound_method %pb.var, %specific_fn.loc19_3
// CHECK:STDOUT:   %.loc19_3.2: %ptr.235 = bind_value %pb.var
// CHECK:STDOUT:   %no_op.4: init %empty_tuple.type = call %bound_method.4(%.loc19_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
