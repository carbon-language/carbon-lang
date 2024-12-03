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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %int.3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.1: type = array_type %int.3, %i32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %int.0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.19: <bound method> = bound_method %int.0.1, %Convert.14 [template]
// CHECK:STDOUT:   %.20: <specific function> = specific_function %.19, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %int.4.1: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.21: <bound method> = bound_method %int.4.1, %Convert.14 [template]
// CHECK:STDOUT:   %.22: <specific function> = specific_function %.21, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.4.2: %i32 = int_value 4 [template]
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
// CHECK:STDOUT:     %return.patt: %.1 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.1 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.32: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int.32) [template = constants.%i32]
// CHECK:STDOUT:     %int.3: Core.IntLiteral = int_value 3 [template = constants.%int.3]
// CHECK:STDOUT:     %.loc11_12.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_12.2: type = converted %int.make_type_signed, %.loc11_12.1 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_18: type = array_type %int.3, %i32 [template = constants.%.1]
// CHECK:STDOUT:     %return.param: ref %.1 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.1 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %b.patt: %.1 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %.1 = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc13: init type = call constants.%Int(%int.32.loc13) [template = constants.%i32]
// CHECK:STDOUT:     %int.3: Core.IntLiteral = int_value 3 [template = constants.%int.3]
// CHECK:STDOUT:     %.loc13_10.1: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_10.2: type = converted %int.make_type_signed.loc13, %.loc13_10.1 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_16: type = array_type %int.3, %i32 [template = constants.%.1]
// CHECK:STDOUT:     %b.param: %.1 = value_param runtime_param0
// CHECK:STDOUT:     %b: %.1 = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %.1;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b.param_patt: %.1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.32.loc19: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc19: init type = call constants.%Int(%int.32.loc19) [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_14.1: type = value_of_initializer %int.make_type_signed.loc19 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_14.2: type = converted %int.make_type_signed.loc19, %.loc19_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_14.3: type = ptr_type %i32 [template = constants.%.3]
// CHECK:STDOUT:   %pb.var: ref %.3 = var pb
// CHECK:STDOUT:   %pb: ref %.3 = bind_name pb, %pb.var
// CHECK:STDOUT:   %b.ref.loc19: %.1 = name_ref b, %b
// CHECK:STDOUT:   %int.0.loc19: Core.IntLiteral = int_value 0 [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc19_21.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc19_21.2: <bound method> = bound_method %int.0.loc19, %.loc19_21.1 [template = constants.%.19]
// CHECK:STDOUT:   %.loc19_21.3: <specific function> = specific_function %.loc19_21.2, @Convert.2(constants.%int.32) [template = constants.%.20]
// CHECK:STDOUT:   %int.convert_checked.loc19: init %i32 = call %.loc19_21.3(%int.0.loc19) [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc19_21.4: %i32 = value_of_initializer %int.convert_checked.loc19 [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc19_21.5: %i32 = converted %int.0.loc19, %.loc19_21.4 [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc19_22.1: ref %.1 = value_as_ref %b.ref.loc19
// CHECK:STDOUT:   %.loc19_22.2: ref %i32 = array_index %.loc19_22.1, %.loc19_21.5
// CHECK:STDOUT:   %.loc19_22.3: %i32 = bind_value %.loc19_22.2
// CHECK:STDOUT:   %.loc19_18: %.3 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   assign %pb.var, %.loc19_18
// CHECK:STDOUT:   %b.ref.loc24: %.1 = name_ref b, %b
// CHECK:STDOUT:   %int.0.loc24: Core.IntLiteral = int_value 0 [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc24_5.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc24_5.2: <bound method> = bound_method %int.0.loc24, %.loc24_5.1 [template = constants.%.19]
// CHECK:STDOUT:   %.loc24_5.3: <specific function> = specific_function %.loc24_5.2, @Convert.2(constants.%int.32) [template = constants.%.20]
// CHECK:STDOUT:   %int.convert_checked.loc24_5: init %i32 = call %.loc24_5.3(%int.0.loc24) [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc24_5.4: %i32 = value_of_initializer %int.convert_checked.loc24_5 [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc24_5.5: %i32 = converted %int.0.loc24, %.loc24_5.4 [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc24_6.1: ref %.1 = value_as_ref %b.ref.loc24
// CHECK:STDOUT:   %.loc24_6.2: ref %i32 = array_index %.loc24_6.1, %.loc24_5.5
// CHECK:STDOUT:   %.loc24_6.3: %i32 = bind_value %.loc24_6.2
// CHECK:STDOUT:   %int.4.loc24: Core.IntLiteral = int_value 4 [template = constants.%int.4.1]
// CHECK:STDOUT:   %.loc24_8.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc24_8.2: <bound method> = bound_method %int.4.loc24, %.loc24_8.1 [template = constants.%.21]
// CHECK:STDOUT:   %.loc24_8.3: <specific function> = specific_function %.loc24_8.2, @Convert.2(constants.%int.32) [template = constants.%.22]
// CHECK:STDOUT:   %int.convert_checked.loc24_8: init %i32 = call %.loc24_8.3(%int.4.loc24) [template = constants.%int.4.2]
// CHECK:STDOUT:   %.loc24_8.4: init %i32 = converted %int.4.loc24, %int.convert_checked.loc24_8 [template = constants.%int.4.2]
// CHECK:STDOUT:   assign %.loc24_6.3, %.loc24_8.4
// CHECK:STDOUT:   %int.32.loc32: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc32: init type = call constants.%Int(%int.32.loc32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc32_14.1: type = value_of_initializer %int.make_type_signed.loc32 [template = constants.%i32]
// CHECK:STDOUT:   %.loc32_14.2: type = converted %int.make_type_signed.loc32, %.loc32_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc32_14.3: type = ptr_type %i32 [template = constants.%.3]
// CHECK:STDOUT:   %pf.var: ref %.3 = var pf
// CHECK:STDOUT:   %pf: ref %.3 = bind_name pf, %pf.var
// CHECK:STDOUT:   %F.ref.loc32: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc32_21.1: ref %.1 = temporary_storage
// CHECK:STDOUT:   %F.call.loc32: init %.1 = call %F.ref.loc32() to %.loc32_21.1
// CHECK:STDOUT:   %int.0.loc32: Core.IntLiteral = int_value 0 [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc32_21.2: ref %.1 = temporary %.loc32_21.1, %F.call.loc32
// CHECK:STDOUT:   %.loc32_23.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc32_23.2: <bound method> = bound_method %int.0.loc32, %.loc32_23.1 [template = constants.%.19]
// CHECK:STDOUT:   %.loc32_23.3: <specific function> = specific_function %.loc32_23.2, @Convert.2(constants.%int.32) [template = constants.%.20]
// CHECK:STDOUT:   %int.convert_checked.loc32: init %i32 = call %.loc32_23.3(%int.0.loc32) [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc32_23.4: %i32 = value_of_initializer %int.convert_checked.loc32 [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc32_23.5: %i32 = converted %int.0.loc32, %.loc32_23.4 [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc32_24.1: ref %i32 = array_index %.loc32_21.2, %.loc32_23.5
// CHECK:STDOUT:   %.loc32_24.2: %i32 = bind_value %.loc32_24.1
// CHECK:STDOUT:   %.loc32_18: %.3 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   assign %pf.var, %.loc32_18
// CHECK:STDOUT:   %F.ref.loc36: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc36_5.1: ref %.1 = temporary_storage
// CHECK:STDOUT:   %F.call.loc36: init %.1 = call %F.ref.loc36() to %.loc36_5.1
// CHECK:STDOUT:   %int.0.loc36: Core.IntLiteral = int_value 0 [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc36_5.2: ref %.1 = temporary %.loc36_5.1, %F.call.loc36
// CHECK:STDOUT:   %.loc36_7.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc36_7.2: <bound method> = bound_method %int.0.loc36, %.loc36_7.1 [template = constants.%.19]
// CHECK:STDOUT:   %.loc36_7.3: <specific function> = specific_function %.loc36_7.2, @Convert.2(constants.%int.32) [template = constants.%.20]
// CHECK:STDOUT:   %int.convert_checked.loc36_7: init %i32 = call %.loc36_7.3(%int.0.loc36) [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc36_7.4: %i32 = value_of_initializer %int.convert_checked.loc36_7 [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc36_7.5: %i32 = converted %int.0.loc36, %.loc36_7.4 [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc36_8.1: ref %i32 = array_index %.loc36_5.2, %.loc36_7.5
// CHECK:STDOUT:   %.loc36_8.2: %i32 = bind_value %.loc36_8.1
// CHECK:STDOUT:   %int.4.loc36: Core.IntLiteral = int_value 4 [template = constants.%int.4.1]
// CHECK:STDOUT:   %.loc36_10.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc36_10.2: <bound method> = bound_method %int.4.loc36, %.loc36_10.1 [template = constants.%.21]
// CHECK:STDOUT:   %.loc36_10.3: <specific function> = specific_function %.loc36_10.2, @Convert.2(constants.%int.32) [template = constants.%.22]
// CHECK:STDOUT:   %int.convert_checked.loc36_10: init %i32 = call %.loc36_10.3(%int.4.loc36) [template = constants.%int.4.2]
// CHECK:STDOUT:   %.loc36_10.4: init %i32 = converted %int.4.loc36, %int.convert_checked.loc36_10 [template = constants.%int.4.2]
// CHECK:STDOUT:   assign %.loc36_8.2, %.loc36_10.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
