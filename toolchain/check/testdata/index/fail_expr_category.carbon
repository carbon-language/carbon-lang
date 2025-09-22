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
// CHECK:STDOUT:   %pattern_type.5d8: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.f01: type = ptr_type %array_type [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %ptr.235 [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.204: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9e9, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.204) [concrete]
// CHECK:STDOUT:   %.1df: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.931: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.c25: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.31f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.8a8: %ptr.as.Copy.impl.Op.type.31f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a93: <witness> = impl_witness imports.%Copy.impl_witness_table.53c, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.b57: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.9fb: %ptr.as.Copy.impl.Op.type.b57 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.7d5: %Copy.type = facet_value %ptr.235, (%Copy.impl_witness.a93) [concrete]
// CHECK:STDOUT:   %.9e2: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.7d5 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.9fb, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5bb: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %bound_method.9cd: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %array_type, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.ffb: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.e7e: %AggregateT.as_type.as.Destroy.impl.Op.type.ffb = struct_value () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.e7e, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.type.142: type = fn_type @ptr.as.Destroy.impl.Op, @ptr.as.Destroy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.c04: %ptr.as.Destroy.impl.Op.type.142 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5d5: type = ptr_type %ptr.235 [concrete]
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Destroy.impl.Op.c04, @ptr.as.Destroy.impl.Op(%i32) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.ee7: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340) = import_ref Core//prelude/parts/int, loc23_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9e9 = impl_witness_table (%Core.import_ref.ee7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.0e4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.31f) = import_ref Core//prelude/parts/copy, loc36_31, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.8a8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.53c = impl_witness_table (%Core.import_ref.0e4), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %return.param: ref %array_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %array_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %b.patt: %pattern_type.5d8 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.5d8 = value_param_pattern %b.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %array_type = value_param call_param0
// CHECK:STDOUT:     %.loc17: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:       %int_32.loc17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:       %array_type: type = array_type %int_3, %i32.loc17 [concrete = constants.%array_type]
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
// CHECK:STDOUT:     %pb.patt: %pattern_type.fe8 = binding_pattern pb [concrete]
// CHECK:STDOUT:     %pb.var_patt: %pattern_type.fe8 = var_pattern %pb.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pb.var: ref %ptr.235 = var %pb.var_patt
// CHECK:STDOUT:   %b.ref.loc23: %array_type = name_ref b, %b
// CHECK:STDOUT:   %int_0.loc23: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc23_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc23_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc23_21: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc23_21.1: <bound method> = bound_method %int_0.loc23, %impl.elem0.loc23_21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.931]
// CHECK:STDOUT:   %specific_fn.loc23_21: <specific function> = specific_function %impl.elem0.loc23_21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_21.2: <bound method> = bound_method %int_0.loc23, %specific_fn.loc23_21 [concrete = constants.%bound_method.c25]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23: init %i32 = call %bound_method.loc23_21.2(%int_0.loc23) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc23_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc23_21.2: %i32 = converted %int_0.loc23, %.loc23_21.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc23_22.1: ref %array_type = value_as_ref %b.ref.loc23
// CHECK:STDOUT:   %.loc23_22.2: ref %i32 = array_index %.loc23_22.1, %.loc23_21.2
// CHECK:STDOUT:   %.loc23_22.3: %i32 = bind_value %.loc23_22.2
// CHECK:STDOUT:   %addr.loc23_18: %ptr.235 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %impl.elem0.loc23_18: %.9e2 = impl_witness_access constants.%Copy.impl_witness.a93, element0 [concrete = constants.%ptr.as.Copy.impl.Op.9fb]
// CHECK:STDOUT:   %bound_method.loc23_18.1: <bound method> = bound_method %addr.loc23_18, %impl.elem0.loc23_18 [concrete = <error>]
// CHECK:STDOUT:   %specific_fn.loc23_18: <specific function> = specific_function %impl.elem0.loc23_18, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_18.2: <bound method> = bound_method %addr.loc23_18, %specific_fn.loc23_18 [concrete = <error>]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc23: init %ptr.235 = call %bound_method.loc23_18.2(%addr.loc23_18) [concrete = <error>]
// CHECK:STDOUT:   assign %pb.var, %ptr.as.Copy.impl.Op.call.loc23
// CHECK:STDOUT:   %.loc23_14: type = splice_block %ptr.loc23 [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc23_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc23_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc23: type = ptr_type %i32.loc23_11 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pb: ref %ptr.235 = bind_name pb, %pb.var
// CHECK:STDOUT:   %b.ref.loc28: %array_type = name_ref b, %b
// CHECK:STDOUT:   %int_0.loc28: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc28: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc28: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc28_5: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc28_5.1: <bound method> = bound_method %int_0.loc28, %impl.elem0.loc28_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.931]
// CHECK:STDOUT:   %specific_fn.loc28_5: <specific function> = specific_function %impl.elem0.loc28_5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc28_5.2: <bound method> = bound_method %int_0.loc28, %specific_fn.loc28_5 [concrete = constants.%bound_method.c25]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28_5: init %i32 = call %bound_method.loc28_5.2(%int_0.loc28) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc28_5.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28_5 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc28_5.2: %i32 = converted %int_0.loc28, %.loc28_5.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc28_6.1: ref %array_type = value_as_ref %b.ref.loc28
// CHECK:STDOUT:   %.loc28_6.2: ref %i32 = array_index %.loc28_6.1, %.loc28_5.2
// CHECK:STDOUT:   %.loc28_6.3: %i32 = bind_value %.loc28_6.2
// CHECK:STDOUT:   %int_4.loc28: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc28_8: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc28_8.1: <bound method> = bound_method %int_4.loc28, %impl.elem0.loc28_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5bb]
// CHECK:STDOUT:   %specific_fn.loc28_8: <specific function> = specific_function %impl.elem0.loc28_8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc28_8.2: <bound method> = bound_method %int_4.loc28, %specific_fn.loc28_8 [concrete = constants.%bound_method.9cd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28_8: init %i32 = call %bound_method.loc28_8.2(%int_4.loc28) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc28_8: init %i32 = converted %int_4.loc28, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28_8 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc28_6.3, %.loc28_8
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pf.patt: %pattern_type.fe8 = binding_pattern pf [concrete]
// CHECK:STDOUT:     %pf.var_patt: %pattern_type.fe8 = var_pattern %pf.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pf.var: ref %ptr.235 = var %pf.var_patt
// CHECK:STDOUT:   %F.ref.loc36: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc36_21.1: ref %array_type = temporary_storage
// CHECK:STDOUT:   %F.call.loc36: init %array_type = call %F.ref.loc36() to %.loc36_21.1
// CHECK:STDOUT:   %int_0.loc36: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc36_21.2: ref %array_type = temporary %.loc36_21.1, %F.call.loc36
// CHECK:STDOUT:   %int_32.loc36_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc36_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc36_23: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc36_23.1: <bound method> = bound_method %int_0.loc36, %impl.elem0.loc36_23 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.931]
// CHECK:STDOUT:   %specific_fn.loc36_23: <specific function> = specific_function %impl.elem0.loc36_23, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc36_23.2: <bound method> = bound_method %int_0.loc36, %specific_fn.loc36_23 [concrete = constants.%bound_method.c25]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc36: init %i32 = call %bound_method.loc36_23.2(%int_0.loc36) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc36_23.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc36 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc36_23.2: %i32 = converted %int_0.loc36, %.loc36_23.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc36_24.1: ref %i32 = array_index %.loc36_21.2, %.loc36_23.2
// CHECK:STDOUT:   %.loc36_24.2: %i32 = bind_value %.loc36_24.1
// CHECK:STDOUT:   %addr.loc36_18: %ptr.235 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %impl.elem0.loc36_18: %.9e2 = impl_witness_access constants.%Copy.impl_witness.a93, element0 [concrete = constants.%ptr.as.Copy.impl.Op.9fb]
// CHECK:STDOUT:   %bound_method.loc36_18.1: <bound method> = bound_method %addr.loc36_18, %impl.elem0.loc36_18 [concrete = <error>]
// CHECK:STDOUT:   %specific_fn.loc36_18: <specific function> = specific_function %impl.elem0.loc36_18, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc36_18.2: <bound method> = bound_method %addr.loc36_18, %specific_fn.loc36_18 [concrete = <error>]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc36: init %ptr.235 = call %bound_method.loc36_18.2(%addr.loc36_18) [concrete = <error>]
// CHECK:STDOUT:   assign %pf.var, %ptr.as.Copy.impl.Op.call.loc36
// CHECK:STDOUT:   %.loc36_14: type = splice_block %ptr.loc36 [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc36_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc36_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc36: type = ptr_type %i32.loc36_11 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pf: ref %ptr.235 = bind_name pf, %pf.var
// CHECK:STDOUT:   %F.ref.loc41: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc41_5.1: ref %array_type = temporary_storage
// CHECK:STDOUT:   %F.call.loc41: init %array_type = call %F.ref.loc41() to %.loc41_5.1
// CHECK:STDOUT:   %int_0.loc41: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc41_5.2: ref %array_type = temporary %.loc41_5.1, %F.call.loc41
// CHECK:STDOUT:   %int_32.loc41: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc41: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc41_7: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc41_7.1: <bound method> = bound_method %int_0.loc41, %impl.elem0.loc41_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.931]
// CHECK:STDOUT:   %specific_fn.loc41_7: <specific function> = specific_function %impl.elem0.loc41_7, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc41_7.2: <bound method> = bound_method %int_0.loc41, %specific_fn.loc41_7 [concrete = constants.%bound_method.c25]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc41_7: init %i32 = call %bound_method.loc41_7.2(%int_0.loc41) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc41_7.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc41_7 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc41_7.2: %i32 = converted %int_0.loc41, %.loc41_7.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc41_8.1: ref %i32 = array_index %.loc41_5.2, %.loc41_7.2
// CHECK:STDOUT:   %.loc41_8.2: %i32 = bind_value %.loc41_8.1
// CHECK:STDOUT:   %int_4.loc41: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc41_10: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc41_10.1: <bound method> = bound_method %int_4.loc41, %impl.elem0.loc41_10 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5bb]
// CHECK:STDOUT:   %specific_fn.loc41_10: <specific function> = specific_function %impl.elem0.loc41_10, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc41_10.2: <bound method> = bound_method %int_4.loc41, %specific_fn.loc41_10 [concrete = constants.%bound_method.9cd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc41_10: init %i32 = call %bound_method.loc41_10.2(%int_4.loc41) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc41_10: init %i32 = converted %int_4.loc41, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc41_10 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc41_8.2, %.loc41_10
// CHECK:STDOUT:   %facet_value.loc41: %type_where = facet_value constants.%array_type, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc41_5.3: %type_where = converted constants.%array_type, %facet_value.loc41 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc41: <bound method> = bound_method %.loc41_5.2, constants.%AggregateT.as_type.as.Destroy.impl.Op.e7e
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.e7e, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc41_5: <bound method> = bound_method %.loc41_5.2, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc41: %ptr.f01 = addr_of %.loc41_5.2
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc41: init %empty_tuple.type = call %bound_method.loc41_5(%addr.loc41)
// CHECK:STDOUT:   %facet_value.loc36: %type_where = facet_value constants.%array_type, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc36_21.3: %type_where = converted constants.%array_type, %facet_value.loc36 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc36: <bound method> = bound_method %.loc36_21.2, constants.%AggregateT.as_type.as.Destroy.impl.Op.e7e
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.e7e, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc36_21: <bound method> = bound_method %.loc36_21.2, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc36_21: %ptr.f01 = addr_of %.loc36_21.2
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc36: init %empty_tuple.type = call %bound_method.loc36_21(%addr.loc36_21)
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.bound.loc36: <bound method> = bound_method %pf.var, constants.%ptr.as.Destroy.impl.Op.c04
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%ptr.as.Destroy.impl.Op.c04, @ptr.as.Destroy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc36_3: <bound method> = bound_method %pf.var, %ptr.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc36_3: %ptr.5d5 = addr_of %pf.var
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.call.loc36: init %empty_tuple.type = call %bound_method.loc36_3(%addr.loc36_3)
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.bound.loc23: <bound method> = bound_method %pb.var, constants.%ptr.as.Destroy.impl.Op.c04
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%ptr.as.Destroy.impl.Op.c04, @ptr.as.Destroy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_3: <bound method> = bound_method %pb.var, %ptr.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc23_3: %ptr.5d5 = addr_of %pb.var
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.call.loc23: init %empty_tuple.type = call %bound_method.loc23_3(%addr.loc23_3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
