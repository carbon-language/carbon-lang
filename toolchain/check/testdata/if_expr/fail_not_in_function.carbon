// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/if_expr/fail_not_in_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/if_expr/fail_not_in_function.carbon

// CHECK:STDERR: fail_not_in_function.carbon:[[@LINE+12]]:14: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: let x: i32 = if true then 1 else 0;
// CHECK:STDERR:              ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_not_in_function.carbon:[[@LINE+8]]:14: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: let x: i32 = if true then 1 else 0;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_not_in_function.carbon:[[@LINE+4]]:22: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: let x: i32 = if true then 1 else 0;
// CHECK:STDERR:                      ^~~~~~
// CHECK:STDERR:
let x: i32 = if true then 1 else 0;

class C {
  // CHECK:STDERR: fail_not_in_function.carbon:[[@LINE+11]]:10: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
  // CHECK:STDERR:   var n: if true then i32 else f64;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_not_in_function.carbon:[[@LINE+7]]:10: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
  // CHECK:STDERR:   var n: if true then i32 else f64;
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_not_in_function.carbon:[[@LINE+3]]:18: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
  // CHECK:STDERR:   var n: if true then i32 else f64;
  // CHECK:STDERR:                  ^~~~~~~~
  var n: if true then i32 else f64;
}

// CHECK:STDOUT: --- fail_not_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %Float: type = fn_type @Float [template]
// CHECK:STDOUT:   %struct.2: Float = struct_value () [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type C, i32 [template]
// CHECK:STDOUT:   %.7: type = struct_type {.n: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc23: i32 = block_arg <unexpected instblockref block8> [template = constants.%.3]
// CHECK:STDOUT:   %x: i32 = bind_name x, %.loc23
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Float = import_ref ir3, inst+31, loaded [template = constants.%struct.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc37: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc37 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .n = <unexpected instref inst+48>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:
