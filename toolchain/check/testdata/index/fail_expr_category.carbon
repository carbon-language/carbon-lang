// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> [i32; 3];

fn G(b: [i32; 3]) {
  // Indexing an array value gives a value.
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+4]]:18: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   var pb: i32* = &b[0];
  // CHECK:STDERR:                  ^
  // CHECK:STDERR:
  var pb: i32* = &b[0];
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+4]]:3: ERROR: Expression is not assignable.
  // CHECK:STDERR:   b[0] = 4;
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  b[0] = 4;

  // Indexing an ephemeral reference (materialized from an initializing
  // expression) gives a value.
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+4]]:18: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   var pf: i32* = &F()[0];
  // CHECK:STDERR:                  ^
  // CHECK:STDERR:
  var pf: i32* = &F()[0];
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+3]]:3: ERROR: Expression is not assignable.
  // CHECK:STDERR:   F()[0] = 4;
  // CHECK:STDERR:   ^~~~~~
  F()[0] = 4;
}

// CHECK:STDOUT: --- fail_expr_category.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.2: G = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type [i32; 3] [template]
// CHECK:STDOUT:   %.5: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 4 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {
// CHECK:STDOUT:     %.loc7_17: i32 = int_literal 3 [template = constants.%.1]
// CHECK:STDOUT:     %.loc7_18: type = array_type %.loc7_17, i32 [template = constants.%.2]
// CHECK:STDOUT:     @F.%return: ref [i32; 3] = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.2] {
// CHECK:STDOUT:     %.loc9_15: i32 = int_literal 3 [template = constants.%.1]
// CHECK:STDOUT:     %.loc9_16: type = array_type %.loc9_15, i32 [template = constants.%.2]
// CHECK:STDOUT:     %b.loc9_6.1: [i32; 3] = param b
// CHECK:STDOUT:     @G.%b: [i32; 3] = bind_name b, %b.loc9_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> [i32; 3];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b: [i32; 3]) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_14: type = ptr_type i32 [template = constants.%.5]
// CHECK:STDOUT:   %pb.var: ref i32* = var pb
// CHECK:STDOUT:   %pb: ref i32* = bind_name pb, %pb.var
// CHECK:STDOUT:   %b.ref.loc15: [i32; 3] = name_ref b, %b
// CHECK:STDOUT:   %.loc15_21: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc15_22.1: ref [i32; 3] = value_as_ref %b.ref.loc15
// CHECK:STDOUT:   %.loc15_22.2: ref i32 = array_index %.loc15_22.1, %.loc15_21
// CHECK:STDOUT:   %.loc15_22.3: i32 = bind_value %.loc15_22.2
// CHECK:STDOUT:   %.loc15_18: i32* = addr_of <error> [template = <error>]
// CHECK:STDOUT:   assign %pb.var, %.loc15_18
// CHECK:STDOUT:   %b.ref.loc20: [i32; 3] = name_ref b, %b
// CHECK:STDOUT:   %.loc20_5: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc20_6.1: ref [i32; 3] = value_as_ref %b.ref.loc20
// CHECK:STDOUT:   %.loc20_6.2: ref i32 = array_index %.loc20_6.1, %.loc20_5
// CHECK:STDOUT:   %.loc20_6.3: i32 = bind_value %.loc20_6.2
// CHECK:STDOUT:   %.loc20_10: i32 = int_literal 4 [template = constants.%.7]
// CHECK:STDOUT:   assign %.loc20_6.3, %.loc20_10
// CHECK:STDOUT:   %.loc28_14: type = ptr_type i32 [template = constants.%.5]
// CHECK:STDOUT:   %pf.var: ref i32* = var pf
// CHECK:STDOUT:   %pf: ref i32* = bind_name pf, %pf.var
// CHECK:STDOUT:   %F.ref.loc28: F = name_ref F, file.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc28_20.1: ref [i32; 3] = temporary_storage
// CHECK:STDOUT:   %F.call.loc28: init [i32; 3] = call %F.ref.loc28() to %.loc28_20.1
// CHECK:STDOUT:   %.loc28_23: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc28_20.2: ref [i32; 3] = temporary %.loc28_20.1, %F.call.loc28
// CHECK:STDOUT:   %.loc28_24.1: ref i32 = array_index %.loc28_20.2, %.loc28_23
// CHECK:STDOUT:   %.loc28_24.2: i32 = bind_value %.loc28_24.1
// CHECK:STDOUT:   %.loc28_18: i32* = addr_of <error> [template = <error>]
// CHECK:STDOUT:   assign %pf.var, %.loc28_18
// CHECK:STDOUT:   %F.ref.loc32: F = name_ref F, file.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc32_4.1: ref [i32; 3] = temporary_storage
// CHECK:STDOUT:   %F.call.loc32: init [i32; 3] = call %F.ref.loc32() to %.loc32_4.1
// CHECK:STDOUT:   %.loc32_7: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc32_4.2: ref [i32; 3] = temporary %.loc32_4.1, %F.call.loc32
// CHECK:STDOUT:   %.loc32_8.1: ref i32 = array_index %.loc32_4.2, %.loc32_7
// CHECK:STDOUT:   %.loc32_8.2: i32 = bind_value %.loc32_8.1
// CHECK:STDOUT:   %.loc32_12: i32 = int_literal 4 [template = constants.%.7]
// CHECK:STDOUT:   assign %.loc32_8.2, %.loc32_12
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
