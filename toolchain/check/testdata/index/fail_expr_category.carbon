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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 3 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.5: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.6: i32 = int_value 4 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %return.patt: %.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_17: i32 = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_12.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc11_12.2: type = converted %int.make_type_32, %.loc11_12.1 [template = i32]
// CHECK:STDOUT:     %.loc11_18: type = array_type %.loc11_17, i32 [template = constants.%.2]
// CHECK:STDOUT:     %return.param: ref %.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %b.patt: %.2 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %.2 = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc13_15: i32 = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:     %.loc13_10.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:     %.loc13_10.2: type = converted %int.make_type_32.loc13, %.loc13_10.1 [template = i32]
// CHECK:STDOUT:     %.loc13_16: type = array_type %.loc13_15, i32 [template = constants.%.2]
// CHECK:STDOUT:     %b.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %b: %.2 = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %.2;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b.param_patt: %.2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc19_14.1: type = value_of_initializer %int.make_type_32.loc19 [template = i32]
// CHECK:STDOUT:   %.loc19_14.2: type = converted %int.make_type_32.loc19, %.loc19_14.1 [template = i32]
// CHECK:STDOUT:   %.loc19_14.3: type = ptr_type i32 [template = constants.%.4]
// CHECK:STDOUT:   %pb.var: ref %.4 = var pb
// CHECK:STDOUT:   %pb: ref %.4 = bind_name pb, %pb.var
// CHECK:STDOUT:   %b.ref.loc19: %.2 = name_ref b, %b
// CHECK:STDOUT:   %.loc19_21: i32 = int_value 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc19_22.1: ref %.2 = value_as_ref %b.ref.loc19
// CHECK:STDOUT:   %.loc19_22.2: ref i32 = array_index %.loc19_22.1, %.loc19_21
// CHECK:STDOUT:   %.loc19_22.3: i32 = bind_value %.loc19_22.2
// CHECK:STDOUT:   %.loc19_18: %.4 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   assign %pb.var, %.loc19_18
// CHECK:STDOUT:   %b.ref.loc24: %.2 = name_ref b, %b
// CHECK:STDOUT:   %.loc24_5: i32 = int_value 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc24_6.1: ref %.2 = value_as_ref %b.ref.loc24
// CHECK:STDOUT:   %.loc24_6.2: ref i32 = array_index %.loc24_6.1, %.loc24_5
// CHECK:STDOUT:   %.loc24_6.3: i32 = bind_value %.loc24_6.2
// CHECK:STDOUT:   %.loc24_10: i32 = int_value 4 [template = constants.%.6]
// CHECK:STDOUT:   assign %.loc24_6.3, %.loc24_10
// CHECK:STDOUT:   %int.make_type_32.loc32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc32_14.1: type = value_of_initializer %int.make_type_32.loc32 [template = i32]
// CHECK:STDOUT:   %.loc32_14.2: type = converted %int.make_type_32.loc32, %.loc32_14.1 [template = i32]
// CHECK:STDOUT:   %.loc32_14.3: type = ptr_type i32 [template = constants.%.4]
// CHECK:STDOUT:   %pf.var: ref %.4 = var pf
// CHECK:STDOUT:   %pf: ref %.4 = bind_name pf, %pf.var
// CHECK:STDOUT:   %F.ref.loc32: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc32_20.1: ref %.2 = temporary_storage
// CHECK:STDOUT:   %F.call.loc32: init %.2 = call %F.ref.loc32() to %.loc32_20.1
// CHECK:STDOUT:   %.loc32_23: i32 = int_value 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc32_20.2: ref %.2 = temporary %.loc32_20.1, %F.call.loc32
// CHECK:STDOUT:   %.loc32_24.1: ref i32 = array_index %.loc32_20.2, %.loc32_23
// CHECK:STDOUT:   %.loc32_24.2: i32 = bind_value %.loc32_24.1
// CHECK:STDOUT:   %.loc32_18: %.4 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   assign %pf.var, %.loc32_18
// CHECK:STDOUT:   %F.ref.loc36: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc36_4.1: ref %.2 = temporary_storage
// CHECK:STDOUT:   %F.call.loc36: init %.2 = call %F.ref.loc36() to %.loc36_4.1
// CHECK:STDOUT:   %.loc36_7: i32 = int_value 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc36_4.2: ref %.2 = temporary %.loc36_4.1, %F.call.loc36
// CHECK:STDOUT:   %.loc36_8.1: ref i32 = array_index %.loc36_4.2, %.loc36_7
// CHECK:STDOUT:   %.loc36_8.2: i32 = bind_value %.loc36_8.1
// CHECK:STDOUT:   %.loc36_12: i32 = int_value 4 [template = constants.%.6]
// CHECK:STDOUT:   assign %.loc36_8.2, %.loc36_12
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
