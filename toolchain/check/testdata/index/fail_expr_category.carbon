// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/fail_expr_category.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/fail_expr_category.carbon

fn F() -> [i32; 3];

fn G(b: [i32; 3]) {
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
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+3]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   F()[0] = 4;
  // CHECK:STDERR:   ^~~~~~
  F()[0] = 4;
}

// CHECK:STDOUT: --- fail_expr_category.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_3, %i32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %int_4.1: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_4.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_4.2: %i32 = int_value 4 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %array_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %array_type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:     %.loc11_12.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_12.2: type = converted %int.make_type_signed, %.loc11_12.1 [template = constants.%i32]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %i32 [template = constants.%array_type]
// CHECK:STDOUT:     %return.param: ref %array_type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %array_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %b.patt: %array_type = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %array_type = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc13: init type = call constants.%Int(%int_32.loc13) [template = constants.%i32]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:     %.loc13_10.1: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_10.2: type = converted %int.make_type_signed.loc13, %.loc13_10.1 [template = constants.%i32]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %i32 [template = constants.%array_type]
// CHECK:STDOUT:     %b.param: %array_type = value_param runtime_param0
// CHECK:STDOUT:     %b: %array_type = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %array_type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b.param_patt: %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32.loc19_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc19_11: init type = call constants.%Int(%int_32.loc19_11) [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_14.1: type = value_of_initializer %int.make_type_signed.loc19_11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_14.2: type = converted %int.make_type_signed.loc19_11, %.loc19_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %ptr.loc19: type = ptr_type %i32 [template = constants.%ptr.2]
// CHECK:STDOUT:   %pb.var: ref %ptr.2 = var pb
// CHECK:STDOUT:   %pb: ref %ptr.2 = bind_name pb, %pb.var
// CHECK:STDOUT:   %b.ref.loc19: %array_type = name_ref b, %b
// CHECK:STDOUT:   %int_0.loc19: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int_32.loc19_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc19_22: init type = call constants.%Int(%int_32.loc19_22) [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_22.1: type = value_of_initializer %int.make_type_signed.loc19_22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_22.2: type = converted %int.make_type_signed.loc19_22, %.loc19_22.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc19: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc19: <bound method> = bound_method %int_0.loc19, %impl.elem0.loc19 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc19: <specific function> = specific_function %Convert.bound.loc19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc19: init %i32 = call %Convert.specific_fn.loc19(%int_0.loc19) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc19_21.1: %i32 = value_of_initializer %int.convert_checked.loc19 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc19_21.2: %i32 = converted %int_0.loc19, %.loc19_21.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc19_22.3: ref %array_type = value_as_ref %b.ref.loc19
// CHECK:STDOUT:   %.loc19_22.4: ref %i32 = array_index %.loc19_22.3, %.loc19_21.2
// CHECK:STDOUT:   %.loc19_22.5: %i32 = bind_value %.loc19_22.4
// CHECK:STDOUT:   %addr.loc19: %ptr.2 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   assign %pb.var, %addr.loc19
// CHECK:STDOUT:   %b.ref.loc24: %array_type = name_ref b, %b
// CHECK:STDOUT:   %int_0.loc24: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int_32.loc24: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc24: init type = call constants.%Int(%int_32.loc24) [template = constants.%i32]
// CHECK:STDOUT:   %.loc24_6.1: type = value_of_initializer %int.make_type_signed.loc24 [template = constants.%i32]
// CHECK:STDOUT:   %.loc24_6.2: type = converted %int.make_type_signed.loc24, %.loc24_6.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc24_5: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc24_5: <bound method> = bound_method %int_0.loc24, %impl.elem0.loc24_5 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc24_5: <specific function> = specific_function %Convert.bound.loc24_5, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc24_5: init %i32 = call %Convert.specific_fn.loc24_5(%int_0.loc24) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc24_5.1: %i32 = value_of_initializer %int.convert_checked.loc24_5 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc24_5.2: %i32 = converted %int_0.loc24, %.loc24_5.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc24_6.3: ref %array_type = value_as_ref %b.ref.loc24
// CHECK:STDOUT:   %.loc24_6.4: ref %i32 = array_index %.loc24_6.3, %.loc24_5.2
// CHECK:STDOUT:   %.loc24_6.5: %i32 = bind_value %.loc24_6.4
// CHECK:STDOUT:   %int_4.loc24: Core.IntLiteral = int_value 4 [template = constants.%int_4.1]
// CHECK:STDOUT:   %impl.elem0.loc24_8: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc24_8: <bound method> = bound_method %int_4.loc24, %impl.elem0.loc24_8 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc24_8: <specific function> = specific_function %Convert.bound.loc24_8, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc24_8: init %i32 = call %Convert.specific_fn.loc24_8(%int_4.loc24) [template = constants.%int_4.2]
// CHECK:STDOUT:   %.loc24_8: init %i32 = converted %int_4.loc24, %int.convert_checked.loc24_8 [template = constants.%int_4.2]
// CHECK:STDOUT:   assign %.loc24_6.5, %.loc24_8
// CHECK:STDOUT:   %int_32.loc32_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc32_11: init type = call constants.%Int(%int_32.loc32_11) [template = constants.%i32]
// CHECK:STDOUT:   %.loc32_14.1: type = value_of_initializer %int.make_type_signed.loc32_11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc32_14.2: type = converted %int.make_type_signed.loc32_11, %.loc32_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %ptr.loc32: type = ptr_type %i32 [template = constants.%ptr.2]
// CHECK:STDOUT:   %pf.var: ref %ptr.2 = var pf
// CHECK:STDOUT:   %pf: ref %ptr.2 = bind_name pf, %pf.var
// CHECK:STDOUT:   %F.ref.loc32: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc32_21.1: ref %array_type = temporary_storage
// CHECK:STDOUT:   %F.call.loc32: init %array_type = call %F.ref.loc32() to %.loc32_21.1
// CHECK:STDOUT:   %int_0.loc32: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc32_21.2: ref %array_type = temporary %.loc32_21.1, %F.call.loc32
// CHECK:STDOUT:   %int_32.loc32_24: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc32_24: init type = call constants.%Int(%int_32.loc32_24) [template = constants.%i32]
// CHECK:STDOUT:   %.loc32_24.1: type = value_of_initializer %int.make_type_signed.loc32_24 [template = constants.%i32]
// CHECK:STDOUT:   %.loc32_24.2: type = converted %int.make_type_signed.loc32_24, %.loc32_24.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc32: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc32: <bound method> = bound_method %int_0.loc32, %impl.elem0.loc32 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc32: <specific function> = specific_function %Convert.bound.loc32, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc32: init %i32 = call %Convert.specific_fn.loc32(%int_0.loc32) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc32_23.1: %i32 = value_of_initializer %int.convert_checked.loc32 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc32_23.2: %i32 = converted %int_0.loc32, %.loc32_23.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc32_24.3: ref %i32 = array_index %.loc32_21.2, %.loc32_23.2
// CHECK:STDOUT:   %.loc32_24.4: %i32 = bind_value %.loc32_24.3
// CHECK:STDOUT:   %addr.loc32: %ptr.2 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   assign %pf.var, %addr.loc32
// CHECK:STDOUT:   %F.ref.loc36: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc36_5.1: ref %array_type = temporary_storage
// CHECK:STDOUT:   %F.call.loc36: init %array_type = call %F.ref.loc36() to %.loc36_5.1
// CHECK:STDOUT:   %int_0.loc36: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc36_5.2: ref %array_type = temporary %.loc36_5.1, %F.call.loc36
// CHECK:STDOUT:   %int_32.loc36: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc36: init type = call constants.%Int(%int_32.loc36) [template = constants.%i32]
// CHECK:STDOUT:   %.loc36_8.1: type = value_of_initializer %int.make_type_signed.loc36 [template = constants.%i32]
// CHECK:STDOUT:   %.loc36_8.2: type = converted %int.make_type_signed.loc36, %.loc36_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc36_7: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc36_7: <bound method> = bound_method %int_0.loc36, %impl.elem0.loc36_7 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc36_7: <specific function> = specific_function %Convert.bound.loc36_7, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc36_7: init %i32 = call %Convert.specific_fn.loc36_7(%int_0.loc36) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc36_7.1: %i32 = value_of_initializer %int.convert_checked.loc36_7 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc36_7.2: %i32 = converted %int_0.loc36, %.loc36_7.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc36_8.3: ref %i32 = array_index %.loc36_5.2, %.loc36_7.2
// CHECK:STDOUT:   %.loc36_8.4: %i32 = bind_value %.loc36_8.3
// CHECK:STDOUT:   %int_4.loc36: Core.IntLiteral = int_value 4 [template = constants.%int_4.1]
// CHECK:STDOUT:   %impl.elem0.loc36_10: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc36_10: <bound method> = bound_method %int_4.loc36, %impl.elem0.loc36_10 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc36_10: <specific function> = specific_function %Convert.bound.loc36_10, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc36_10: init %i32 = call %Convert.specific_fn.loc36_10(%int_4.loc36) [template = constants.%int_4.2]
// CHECK:STDOUT:   %.loc36_10: init %i32 = converted %int_4.loc36, %int.convert_checked.loc36_10 [template = constants.%int_4.2]
// CHECK:STDOUT:   assign %.loc36_8.4, %.loc36_10
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
