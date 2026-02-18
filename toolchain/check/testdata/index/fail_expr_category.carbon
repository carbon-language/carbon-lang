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
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3, %i32 [concrete]
// CHECK:STDOUT:   %.241: Core.Form = init_form %array_type, call_param0 [concrete]
// CHECK:STDOUT:   %pattern_type.5d8: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %ptr.235 [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.d2e: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.843: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.c3c: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.011: %ptr.as.Copy.impl.Op.type.c3c = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.235, (%Copy.impl_witness.843) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e01: type = fn_type @Copy.WithSelf.Op, @Copy(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a62: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e01, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.011, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.6d7: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.1: type = fn_type @DestroyOp.loc41 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.1: %DestroyOp.type.3e79c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.2: type = fn_type @DestroyOp.loc36 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.2: %DestroyOp.type.3e79c2.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %return.patt: %pattern_type.5d8 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.5d8 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:     %.loc15: Core.Form = init_form %array_type, call_param0 [concrete = constants.%.241]
// CHECK:STDOUT:     %return.param: ref %array_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %array_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %b.patt: %pattern_type.5d8 = value_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.5d8 = value_param_pattern %b.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %array_type = value_param call_param0
// CHECK:STDOUT:     %.loc17: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:       %int_32.loc17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:       %array_type: type = array_type %int_3, %i32.loc17 [concrete = constants.%array_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %array_type = value_binding b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %array_type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b.param: %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pb.patt: %pattern_type.fe8 = ref_binding_pattern pb [concrete]
// CHECK:STDOUT:     %pb.var_patt: %pattern_type.fe8 = var_pattern %pb.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pb.var: ref %ptr.235 = var %pb.var_patt
// CHECK:STDOUT:   %b.ref.loc23: %array_type = name_ref b, %b
// CHECK:STDOUT:   %int_0.loc23: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc23_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc23_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc23_21: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc23_21.1: <bound method> = bound_method %int_0.loc23, %impl.elem0.loc23_21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:   %specific_fn.loc23_21: <specific function> = specific_function %impl.elem0.loc23_21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_21.2: <bound method> = bound_method %int_0.loc23, %specific_fn.loc23_21 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23: init %i32 = call %bound_method.loc23_21.2(%int_0.loc23) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc23_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc23_21.2: %i32 = converted %int_0.loc23, %.loc23_21.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc23_22.1: ref %array_type = value_as_ref %b.ref.loc23
// CHECK:STDOUT:   %.loc23_22.2: ref %i32 = array_index %.loc23_22.1, %.loc23_21.2
// CHECK:STDOUT:   %.loc23_22.3: %i32 = acquire_value %.loc23_22.2
// CHECK:STDOUT:   %addr.loc23: %ptr.235 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %impl.elem0.loc23_18: %.a62 = impl_witness_access constants.%Copy.impl_witness.843, element0 [concrete = constants.%ptr.as.Copy.impl.Op.011]
// CHECK:STDOUT:   %bound_method.loc23_18.1: <bound method> = bound_method %addr.loc23, %impl.elem0.loc23_18 [concrete = <error>]
// CHECK:STDOUT:   %specific_fn.loc23_18: <specific function> = specific_function %impl.elem0.loc23_18, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_18.2: <bound method> = bound_method %addr.loc23, %specific_fn.loc23_18 [concrete = <error>]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc23: init %ptr.235 = call %bound_method.loc23_18.2(%addr.loc23) [concrete = <error>]
// CHECK:STDOUT:   assign %pb.var, %ptr.as.Copy.impl.Op.call.loc23
// CHECK:STDOUT:   %.loc23_14: type = splice_block %ptr.loc23 [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc23_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc23_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc23: type = ptr_type %i32.loc23_11 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pb: ref %ptr.235 = ref_binding pb, %pb.var
// CHECK:STDOUT:   %b.ref.loc28: %array_type = name_ref b, %b
// CHECK:STDOUT:   %int_0.loc28: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc28: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc28: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc28_5: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc28_5.1: <bound method> = bound_method %int_0.loc28, %impl.elem0.loc28_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:   %specific_fn.loc28_5: <specific function> = specific_function %impl.elem0.loc28_5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc28_5.2: <bound method> = bound_method %int_0.loc28, %specific_fn.loc28_5 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28_5: init %i32 = call %bound_method.loc28_5.2(%int_0.loc28) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc28_5.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28_5 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc28_5.2: %i32 = converted %int_0.loc28, %.loc28_5.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc28_6.1: ref %array_type = value_as_ref %b.ref.loc28
// CHECK:STDOUT:   %.loc28_6.2: ref %i32 = array_index %.loc28_6.1, %.loc28_5.2
// CHECK:STDOUT:   %.loc28_6.3: %i32 = acquire_value %.loc28_6.2
// CHECK:STDOUT:   %int_4.loc28: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc28_8: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc28_8.1: <bound method> = bound_method %int_4.loc28, %impl.elem0.loc28_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c]
// CHECK:STDOUT:   %specific_fn.loc28_8: <specific function> = specific_function %impl.elem0.loc28_8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc28_8.2: <bound method> = bound_method %int_4.loc28, %specific_fn.loc28_8 [concrete = constants.%bound_method.6d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28_8: init %i32 = call %bound_method.loc28_8.2(%int_4.loc28) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc28_8: init %i32 = converted %int_4.loc28, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28_8 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc28_6.3, %.loc28_8
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pf.patt: %pattern_type.fe8 = ref_binding_pattern pf [concrete]
// CHECK:STDOUT:     %pf.var_patt: %pattern_type.fe8 = var_pattern %pf.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pf.var: ref %ptr.235 = var %pf.var_patt
// CHECK:STDOUT:   %F.ref.loc36: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc36_21.1: ref %array_type = temporary_storage
// CHECK:STDOUT:   %F.call.loc36: init %array_type to %.loc36_21.1 = call %F.ref.loc36()
// CHECK:STDOUT:   %int_0.loc36: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc36_21.2: ref %array_type = temporary %.loc36_21.1, %F.call.loc36
// CHECK:STDOUT:   %int_32.loc36_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc36_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc36_23: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc36_23.1: <bound method> = bound_method %int_0.loc36, %impl.elem0.loc36_23 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:   %specific_fn.loc36_23: <specific function> = specific_function %impl.elem0.loc36_23, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc36_23.2: <bound method> = bound_method %int_0.loc36, %specific_fn.loc36_23 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc36: init %i32 = call %bound_method.loc36_23.2(%int_0.loc36) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc36_23.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc36 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc36_23.2: %i32 = converted %int_0.loc36, %.loc36_23.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc36_24.1: ref %i32 = array_index %.loc36_21.2, %.loc36_23.2
// CHECK:STDOUT:   %.loc36_24.2: %i32 = acquire_value %.loc36_24.1
// CHECK:STDOUT:   %addr.loc36: %ptr.235 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %impl.elem0.loc36_18: %.a62 = impl_witness_access constants.%Copy.impl_witness.843, element0 [concrete = constants.%ptr.as.Copy.impl.Op.011]
// CHECK:STDOUT:   %bound_method.loc36_18.1: <bound method> = bound_method %addr.loc36, %impl.elem0.loc36_18 [concrete = <error>]
// CHECK:STDOUT:   %specific_fn.loc36_18: <specific function> = specific_function %impl.elem0.loc36_18, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc36_18.2: <bound method> = bound_method %addr.loc36, %specific_fn.loc36_18 [concrete = <error>]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc36: init %ptr.235 = call %bound_method.loc36_18.2(%addr.loc36) [concrete = <error>]
// CHECK:STDOUT:   assign %pf.var, %ptr.as.Copy.impl.Op.call.loc36
// CHECK:STDOUT:   %.loc36_14: type = splice_block %ptr.loc36 [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc36_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc36_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc36: type = ptr_type %i32.loc36_11 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pf: ref %ptr.235 = ref_binding pf, %pf.var
// CHECK:STDOUT:   %F.ref.loc41: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc41_5.1: ref %array_type = temporary_storage
// CHECK:STDOUT:   %F.call.loc41: init %array_type to %.loc41_5.1 = call %F.ref.loc41()
// CHECK:STDOUT:   %int_0.loc41: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc41_5.2: ref %array_type = temporary %.loc41_5.1, %F.call.loc41
// CHECK:STDOUT:   %int_32.loc41: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc41: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc41_7: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc41_7.1: <bound method> = bound_method %int_0.loc41, %impl.elem0.loc41_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:   %specific_fn.loc41_7: <specific function> = specific_function %impl.elem0.loc41_7, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc41_7.2: <bound method> = bound_method %int_0.loc41, %specific_fn.loc41_7 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc41_7: init %i32 = call %bound_method.loc41_7.2(%int_0.loc41) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc41_7.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc41_7 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc41_7.2: %i32 = converted %int_0.loc41, %.loc41_7.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc41_8.1: ref %i32 = array_index %.loc41_5.2, %.loc41_7.2
// CHECK:STDOUT:   %.loc41_8.2: %i32 = acquire_value %.loc41_8.1
// CHECK:STDOUT:   %int_4.loc41: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc41_10: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc41_10.1: <bound method> = bound_method %int_4.loc41, %impl.elem0.loc41_10 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c]
// CHECK:STDOUT:   %specific_fn.loc41_10: <specific function> = specific_function %impl.elem0.loc41_10, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc41_10.2: <bound method> = bound_method %int_4.loc41, %specific_fn.loc41_10 [concrete = constants.%bound_method.6d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc41_10: init %i32 = call %bound_method.loc41_10.2(%int_4.loc41) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc41_10: init %i32 = converted %int_4.loc41, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc41_10 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc41_8.2, %.loc41_10
// CHECK:STDOUT:   %DestroyOp.bound.loc41: <bound method> = bound_method %.loc41_5.2, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc41: init %empty_tuple.type = call %DestroyOp.bound.loc41(%.loc41_5.2)
// CHECK:STDOUT:   %DestroyOp.bound.loc36_21: <bound method> = bound_method %.loc36_21.2, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc36_21: init %empty_tuple.type = call %DestroyOp.bound.loc36_21(%.loc36_21.2)
// CHECK:STDOUT:   %DestroyOp.bound.loc36_3: <bound method> = bound_method %pf.var, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc36_3: init %empty_tuple.type = call %DestroyOp.bound.loc36_3(%pf.var)
// CHECK:STDOUT:   %DestroyOp.bound.loc23: <bound method> = bound_method %pb.var, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc23: init %empty_tuple.type = call %DestroyOp.bound.loc23(%pb.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc41(%self.param: ref %array_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc36(%self.param: ref %ptr.235) = "no_op";
// CHECK:STDOUT:
