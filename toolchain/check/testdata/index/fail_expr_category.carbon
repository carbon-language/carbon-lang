// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> [i32; 3];

fn G(b: [i32; 3]) {
  // Indexing an array value gives a value.
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+3]]:18: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   var pb: i32* = &b[0];
  // CHECK:STDERR:                  ^
  var pb: i32* = &b[0];
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+3]]:3: ERROR: Expression is not assignable.
  // CHECK:STDERR:   b[0] = 4;
  // CHECK:STDERR:   ^~~~
  b[0] = 4;

  // Indexing an ephemeral reference (materialized from an initializing
  // expression) gives a value.
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+3]]:18: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   var pf: i32* = &F()[0];
  // CHECK:STDERR:                  ^
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
// CHECK:STDOUT:   %.3: type = ptr_type [i32; 3] [template]
// CHECK:STDOUT:   %.4: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 4 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F, .G = %G} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: [i32; 3];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b: [i32; 3]) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_14: type = ptr_type i32 [template = constants.%.4]
// CHECK:STDOUT:   %pb.var: ref i32* = var pb
// CHECK:STDOUT:   %pb: ref i32* = bind_name pb, %pb.var
// CHECK:STDOUT:   %b.ref.loc14: [i32; 3] = name_ref b, %b
// CHECK:STDOUT:   %.loc14_21: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_22.1: ref [i32; 3] = value_as_ref %b.ref.loc14
// CHECK:STDOUT:   %.loc14_22.2: ref i32 = array_index %.loc14_22.1, %.loc14_21
// CHECK:STDOUT:   %.loc14_22.3: i32 = bind_value %.loc14_22.2
// CHECK:STDOUT:   %.loc14_18: i32* = addr_of <error> [template = <error>]
// CHECK:STDOUT:   assign %pb.var, %.loc14_18
// CHECK:STDOUT:   %b.ref.loc18: [i32; 3] = name_ref b, %b
// CHECK:STDOUT:   %.loc18_5: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc18_6.1: ref [i32; 3] = value_as_ref %b.ref.loc18
// CHECK:STDOUT:   %.loc18_6.2: ref i32 = array_index %.loc18_6.1, %.loc18_5
// CHECK:STDOUT:   %.loc18_6.3: i32 = bind_value %.loc18_6.2
// CHECK:STDOUT:   %.loc18_10: i32 = int_literal 4 [template = constants.%.6]
// CHECK:STDOUT:   assign %.loc18_6.3, %.loc18_10
// CHECK:STDOUT:   %.loc25_14: type = ptr_type i32 [template = constants.%.4]
// CHECK:STDOUT:   %pf.var: ref i32* = var pf
// CHECK:STDOUT:   %pf: ref i32* = bind_name pf, %pf.var
// CHECK:STDOUT:   %F.ref.loc25: <function> = name_ref F, file.%F [template = file.%F]
// CHECK:STDOUT:   %.loc25_20.1: ref [i32; 3] = temporary_storage
// CHECK:STDOUT:   %.loc25_20.2: init [i32; 3] = call %F.ref.loc25() to %.loc25_20.1
// CHECK:STDOUT:   %.loc25_23: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc25_20.3: ref [i32; 3] = temporary %.loc25_20.1, %.loc25_20.2
// CHECK:STDOUT:   %.loc25_24.1: ref i32 = array_index %.loc25_20.3, %.loc25_23
// CHECK:STDOUT:   %.loc25_24.2: i32 = bind_value %.loc25_24.1
// CHECK:STDOUT:   %.loc25_18: i32* = addr_of <error> [template = <error>]
// CHECK:STDOUT:   assign %pf.var, %.loc25_18
// CHECK:STDOUT:   %F.ref.loc29: <function> = name_ref F, file.%F [template = file.%F]
// CHECK:STDOUT:   %.loc29_4.1: ref [i32; 3] = temporary_storage
// CHECK:STDOUT:   %.loc29_4.2: init [i32; 3] = call %F.ref.loc29() to %.loc29_4.1
// CHECK:STDOUT:   %.loc29_7: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc29_4.3: ref [i32; 3] = temporary %.loc29_4.1, %.loc29_4.2
// CHECK:STDOUT:   %.loc29_8.1: ref i32 = array_index %.loc29_4.3, %.loc29_7
// CHECK:STDOUT:   %.loc29_8.2: i32 = bind_value %.loc29_8.1
// CHECK:STDOUT:   %.loc29_12: i32 = int_literal 4 [template = constants.%.6]
// CHECK:STDOUT:   assign %.loc29_8.2, %.loc29_12
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
