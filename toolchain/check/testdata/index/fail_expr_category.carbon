// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3, %i32 [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.3, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.3, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.921: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.921 [concrete]
// CHECK:STDOUT:   %Convert.bound.d04: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.3(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.1da: <bound method> = bound_method %int_4.0c1, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
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
// CHECK:STDOUT:     %.loc16: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:       %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:       %array_type: type = array_type %int_3, %i32.loc16 [concrete = constants.%array_type]
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
// CHECK:STDOUT:   %b.ref.loc22: %array_type = name_ref b, %b
// CHECK:STDOUT:   %int_0.loc22: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc22_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc22_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc22: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc22_21.1: <bound method> = bound_method %int_0.loc22, %impl.elem0.loc22 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem0.loc22, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_21.2: <bound method> = bound_method %int_0.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc22: init %i32 = call %bound_method.loc22_21.2(%int_0.loc22) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc22_21.1: %i32 = value_of_initializer %int.convert_checked.loc22 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc22_21.2: %i32 = converted %int_0.loc22, %.loc22_21.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc22_22.1: ref %array_type = value_as_ref %b.ref.loc22
// CHECK:STDOUT:   %.loc22_22.2: ref %i32 = array_index %.loc22_22.1, %.loc22_21.2
// CHECK:STDOUT:   %.loc22_22.3: %i32 = bind_value %.loc22_22.2
// CHECK:STDOUT:   %addr.loc22: %ptr.235 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   assign %pb.var, %addr.loc22
// CHECK:STDOUT:   %.loc22_14: type = splice_block %ptr.loc22 [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc22_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc22: type = ptr_type %i32.loc22_11 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pb: ref %ptr.235 = bind_name pb, %pb.var
// CHECK:STDOUT:   %b.ref.loc27: %array_type = name_ref b, %b
// CHECK:STDOUT:   %int_0.loc27: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc27_5: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc27_5.1: <bound method> = bound_method %int_0.loc27, %impl.elem0.loc27_5 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc27_5: <specific function> = specific_function %impl.elem0.loc27_5, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_5.2: <bound method> = bound_method %int_0.loc27, %specific_fn.loc27_5 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc27_5: init %i32 = call %bound_method.loc27_5.2(%int_0.loc27) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc27_5.1: %i32 = value_of_initializer %int.convert_checked.loc27_5 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc27_5.2: %i32 = converted %int_0.loc27, %.loc27_5.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc27_6.1: ref %array_type = value_as_ref %b.ref.loc27
// CHECK:STDOUT:   %.loc27_6.2: ref %i32 = array_index %.loc27_6.1, %.loc27_5.2
// CHECK:STDOUT:   %.loc27_6.3: %i32 = bind_value %.loc27_6.2
// CHECK:STDOUT:   %int_4.loc27: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc27_8: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc27_8.1: <bound method> = bound_method %int_4.loc27, %impl.elem0.loc27_8 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc27_8: <specific function> = specific_function %impl.elem0.loc27_8, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_8.2: <bound method> = bound_method %int_4.loc27, %specific_fn.loc27_8 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc27_8: init %i32 = call %bound_method.loc27_8.2(%int_4.loc27) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc27_8: init %i32 = converted %int_4.loc27, %int.convert_checked.loc27_8 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc27_6.3, %.loc27_8
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pf.patt: %pattern_type.fe8 = binding_pattern pf [concrete]
// CHECK:STDOUT:     %pf.var_patt: %pattern_type.fe8 = var_pattern %pf.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pf.var: ref %ptr.235 = var %pf.var_patt
// CHECK:STDOUT:   %F.ref.loc35: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc35_21.1: ref %array_type = temporary_storage
// CHECK:STDOUT:   %F.call.loc35: init %array_type = call %F.ref.loc35() to %.loc35_21.1
// CHECK:STDOUT:   %int_0.loc35: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc35_21.2: ref %array_type = temporary %.loc35_21.1, %F.call.loc35
// CHECK:STDOUT:   %int_32.loc35_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc35_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc35: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc35_23.1: <bound method> = bound_method %int_0.loc35, %impl.elem0.loc35 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc35: <specific function> = specific_function %impl.elem0.loc35, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc35_23.2: <bound method> = bound_method %int_0.loc35, %specific_fn.loc35 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc35: init %i32 = call %bound_method.loc35_23.2(%int_0.loc35) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc35_23.1: %i32 = value_of_initializer %int.convert_checked.loc35 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc35_23.2: %i32 = converted %int_0.loc35, %.loc35_23.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc35_24.1: ref %i32 = array_index %.loc35_21.2, %.loc35_23.2
// CHECK:STDOUT:   %.loc35_24.2: %i32 = bind_value %.loc35_24.1
// CHECK:STDOUT:   %addr.loc35: %ptr.235 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   assign %pf.var, %addr.loc35
// CHECK:STDOUT:   %.loc35_14: type = splice_block %ptr.loc35 [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc35_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc35_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc35: type = ptr_type %i32.loc35_11 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pf: ref %ptr.235 = bind_name pf, %pf.var
// CHECK:STDOUT:   %F.ref.loc40: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc40_5.1: ref %array_type = temporary_storage
// CHECK:STDOUT:   %F.call.loc40: init %array_type = call %F.ref.loc40() to %.loc40_5.1
// CHECK:STDOUT:   %int_0.loc40: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc40_5.2: ref %array_type = temporary %.loc40_5.1, %F.call.loc40
// CHECK:STDOUT:   %int_32.loc40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc40: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc40_7: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc40_7.1: <bound method> = bound_method %int_0.loc40, %impl.elem0.loc40_7 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc40_7: <specific function> = specific_function %impl.elem0.loc40_7, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc40_7.2: <bound method> = bound_method %int_0.loc40, %specific_fn.loc40_7 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc40_7: init %i32 = call %bound_method.loc40_7.2(%int_0.loc40) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc40_7.1: %i32 = value_of_initializer %int.convert_checked.loc40_7 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc40_7.2: %i32 = converted %int_0.loc40, %.loc40_7.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc40_8.1: ref %i32 = array_index %.loc40_5.2, %.loc40_7.2
// CHECK:STDOUT:   %.loc40_8.2: %i32 = bind_value %.loc40_8.1
// CHECK:STDOUT:   %int_4.loc40: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc40_10: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc40_10.1: <bound method> = bound_method %int_4.loc40, %impl.elem0.loc40_10 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc40_10: <specific function> = specific_function %impl.elem0.loc40_10, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc40_10.2: <bound method> = bound_method %int_4.loc40, %specific_fn.loc40_10 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc40_10: init %i32 = call %bound_method.loc40_10.2(%int_4.loc40) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc40_10: init %i32 = converted %int_4.loc40, %int.convert_checked.loc40_10 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc40_8.2, %.loc40_10
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
