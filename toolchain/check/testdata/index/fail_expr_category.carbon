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
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.1(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.1(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.d04: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.d62: <specific function> = specific_function %Convert.bound.d04, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.450: <specific function> = specific_function %Convert.bound.ac3, @Convert.2(%int_32) [concrete]
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
// CHECK:STDOUT:     %return.param_patt: %array_type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:     %return.param: ref %array_type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %array_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %b.patt: %array_type = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %array_type = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %array_type = value_param runtime_param0
// CHECK:STDOUT:     %.loc13: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:       %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:       %array_type: type = array_type %int_3, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %array_type = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %array_type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b.param_patt: %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pb.patt: %ptr.235 = binding_pattern pb
// CHECK:STDOUT:     %.loc19_3: %ptr.235 = var_pattern %pb.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pb.var: ref %ptr.235 = var pb
// CHECK:STDOUT:   %b.ref.loc19: %array_type = name_ref b, %b
// CHECK:STDOUT:   %int_0.loc19: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc19_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc19_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc19: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc19: <bound method> = bound_method %int_0.loc19, %impl.elem0.loc19 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %bound_method.loc19, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.d62]
// CHECK:STDOUT:   %int.convert_checked.loc19: init %i32 = call %specific_fn.loc19(%int_0.loc19) [concrete = constants.%int_0.6a9]
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
// CHECK:STDOUT:     %ptr.loc19: type = ptr_type %i32 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pb: ref %ptr.235 = bind_name pb, %pb.var
// CHECK:STDOUT:   %b.ref.loc24: %array_type = name_ref b, %b
// CHECK:STDOUT:   %int_0.loc24: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc24_5: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc24_5: <bound method> = bound_method %int_0.loc24, %impl.elem0.loc24_5 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc24_5: <specific function> = specific_function %bound_method.loc24_5, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.d62]
// CHECK:STDOUT:   %int.convert_checked.loc24_5: init %i32 = call %specific_fn.loc24_5(%int_0.loc24) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc24_5.1: %i32 = value_of_initializer %int.convert_checked.loc24_5 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc24_5.2: %i32 = converted %int_0.loc24, %.loc24_5.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc24_6.1: ref %array_type = value_as_ref %b.ref.loc24
// CHECK:STDOUT:   %.loc24_6.2: ref %i32 = array_index %.loc24_6.1, %.loc24_5.2
// CHECK:STDOUT:   %.loc24_6.3: %i32 = bind_value %.loc24_6.2
// CHECK:STDOUT:   %int_4.loc24: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc24_8: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc24_8: <bound method> = bound_method %int_4.loc24, %impl.elem0.loc24_8 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc24_8: <specific function> = specific_function %bound_method.loc24_8, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.450]
// CHECK:STDOUT:   %int.convert_checked.loc24_8: init %i32 = call %specific_fn.loc24_8(%int_4.loc24) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc24_8: init %i32 = converted %int_4.loc24, %int.convert_checked.loc24_8 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc24_6.3, %.loc24_8
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pf.patt: %ptr.235 = binding_pattern pf
// CHECK:STDOUT:     %.loc32_3: %ptr.235 = var_pattern %pf.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pf.var: ref %ptr.235 = var pf
// CHECK:STDOUT:   %F.ref.loc32: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc32_21.1: ref %array_type = temporary_storage
// CHECK:STDOUT:   %F.call.loc32: init %array_type = call %F.ref.loc32() to %.loc32_21.1
// CHECK:STDOUT:   %int_0.loc32: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc32_21.2: ref %array_type = temporary %.loc32_21.1, %F.call.loc32
// CHECK:STDOUT:   %int_32.loc32_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc32_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc32: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc32: <bound method> = bound_method %int_0.loc32, %impl.elem0.loc32 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc32: <specific function> = specific_function %bound_method.loc32, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.d62]
// CHECK:STDOUT:   %int.convert_checked.loc32: init %i32 = call %specific_fn.loc32(%int_0.loc32) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc32_23.1: %i32 = value_of_initializer %int.convert_checked.loc32 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc32_23.2: %i32 = converted %int_0.loc32, %.loc32_23.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc32_24.1: ref %i32 = array_index %.loc32_21.2, %.loc32_23.2
// CHECK:STDOUT:   %.loc32_24.2: %i32 = bind_value %.loc32_24.1
// CHECK:STDOUT:   %addr.loc32: %ptr.235 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   assign %pf.var, %addr.loc32
// CHECK:STDOUT:   %.loc32_14: type = splice_block %ptr.loc32 [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc32_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc32_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc32: type = ptr_type %i32 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pf: ref %ptr.235 = bind_name pf, %pf.var
// CHECK:STDOUT:   %F.ref.loc37: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc37_5.1: ref %array_type = temporary_storage
// CHECK:STDOUT:   %F.call.loc37: init %array_type = call %F.ref.loc37() to %.loc37_5.1
// CHECK:STDOUT:   %int_0.loc37: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc37_5.2: ref %array_type = temporary %.loc37_5.1, %F.call.loc37
// CHECK:STDOUT:   %int_32.loc37: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc37: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc37_7: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc37_7: <bound method> = bound_method %int_0.loc37, %impl.elem0.loc37_7 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc37_7: <specific function> = specific_function %bound_method.loc37_7, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.d62]
// CHECK:STDOUT:   %int.convert_checked.loc37_7: init %i32 = call %specific_fn.loc37_7(%int_0.loc37) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc37_7.1: %i32 = value_of_initializer %int.convert_checked.loc37_7 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc37_7.2: %i32 = converted %int_0.loc37, %.loc37_7.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc37_8.1: ref %i32 = array_index %.loc37_5.2, %.loc37_7.2
// CHECK:STDOUT:   %.loc37_8.2: %i32 = bind_value %.loc37_8.1
// CHECK:STDOUT:   %int_4.loc37: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc37_10: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc37_10: <bound method> = bound_method %int_4.loc37, %impl.elem0.loc37_10 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc37_10: <specific function> = specific_function %bound_method.loc37_10, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.450]
// CHECK:STDOUT:   %int.convert_checked.loc37_10: init %i32 = call %specific_fn.loc37_10(%int_4.loc37) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc37_10: init %i32 = converted %int_4.loc37, %int.convert_checked.loc37_10 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc37_8.2, %.loc37_10
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
