// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_todo_assoc_const_default.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/interface/fail_todo_assoc_const_default.carbon

interface I {
  // CHECK:STDERR: fail_todo_assoc_const_default.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `interface modifier`.
  // CHECK:STDERR:   default let T:! type = (i32, i32);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default let T:! type = (i32, i32);
  // CHECK:STDERR: fail_todo_assoc_const_default.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `interface modifier`.
  // CHECK:STDERR:   default let N:! i32 = 42;
  // CHECK:STDERR:   ^~~~~~~
  default let N:! i32 = 42;
}

// CHECK:STDOUT: --- fail_todo_assoc_const_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %Self: I = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type @I, type [template]
// CHECK:STDOUT:   %.6: <associated type in I> = assoc_entity element0, @I.%T [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 42 [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type @I, i32 [template]
// CHECK:STDOUT:   %.9: <associated i32 in I> = assoc_entity element1, @I.%N [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.1] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: I = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %int.make_type_32.loc16_27: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc16_32: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc16_35: (type, type) = tuple_literal (%int.make_type_32.loc16_27, %int.make_type_32.loc16_32)
// CHECK:STDOUT:   %.loc16_36.1: type = value_of_initializer %int.make_type_32.loc16_27 [template = i32]
// CHECK:STDOUT:   %.loc16_36.2: type = converted %int.make_type_32.loc16_27, %.loc16_36.1 [template = i32]
// CHECK:STDOUT:   %.loc16_36.3: type = value_of_initializer %int.make_type_32.loc16_32 [template = i32]
// CHECK:STDOUT:   %.loc16_36.4: type = converted %int.make_type_32.loc16_32, %.loc16_36.3 [template = i32]
// CHECK:STDOUT:   %.loc16_36.5: type = converted %.loc16_35, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc16_36.6: <associated type in I> = assoc_entity element0, %T [template = constants.%.6]
// CHECK:STDOUT:   %int.make_type_32.loc20: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc20_19.1: type = value_of_initializer %int.make_type_32.loc20 [template = i32]
// CHECK:STDOUT:   %.loc20_19.2: type = converted %int.make_type_32.loc20, %.loc20_19.1 [template = i32]
// CHECK:STDOUT:   %.loc20_25: i32 = int_literal 42 [template = constants.%.7]
// CHECK:STDOUT:   %N: i32 = assoc_const_decl N [template]
// CHECK:STDOUT:   %.loc20_27: <associated i32 in I> = assoc_entity element1, %N [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = %.loc16_36.6
// CHECK:STDOUT:   .N = %.loc20_27
// CHECK:STDOUT:   witness = (%T, %N)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
