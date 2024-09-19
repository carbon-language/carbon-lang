// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/fail_return_with_returned_var.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/fail_return_with_returned_var.carbon

fn F() -> i32 {
  returned var v: i32 = 0;
  // CHECK:STDERR: fail_return_with_returned_var.carbon:[[@LINE+7]]:3: error: can only `return var;` in the scope of a `returned var`
  // CHECK:STDERR:   return 1;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_return_with_returned_var.carbon:[[@LINE-4]]:16: `returned var` was declared here
  // CHECK:STDERR:   returned var v: i32 = 0;
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  return 1;
}

class C { var a: i32; var b: i32; }
fn G() -> C {
  returned var c: C = {.a = 1, .b = 2};
  // CHECK:STDERR: fail_return_with_returned_var.carbon:[[@LINE+6]]:3: error: can only `return var;` in the scope of a `returned var`
  // CHECK:STDERR:   return c;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_return_with_returned_var.carbon:[[@LINE-4]]:16: `returned var` was declared here
  // CHECK:STDERR:   returned var c: C = {.a = 1, .b = 2};
  // CHECK:STDERR:                ^
  return c;
}

// CHECK:STDOUT: --- fail_return_with_returned_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type %C, i32 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %.5 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %struct: %C = struct_value (%.3, %.8) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %int.make_type_32, %.loc11_11.1 [template = i32]
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @G.%return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %int.make_type_32.loc23_18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc23_18.1: type = value_of_initializer %int.make_type_32.loc23_18 [template = i32]
// CHECK:STDOUT:   %.loc23_18.2: type = converted %int.make_type_32.loc23_18, %.loc23_18.1 [template = i32]
// CHECK:STDOUT:   %.loc23_16: %.4 = field_decl a, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc23_30: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc23_30.1: type = value_of_initializer %int.make_type_32.loc23_30 [template = i32]
// CHECK:STDOUT:   %.loc23_30.2: type = converted %int.make_type_32.loc23_30, %.loc23_30.1 [template = i32]
// CHECK:STDOUT:   %.loc23_28: %.4 = field_decl b, element1 [template]
// CHECK:STDOUT:   %.loc23_35: <witness> = complete_type_witness %.5 [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .a = %.loc23_16
// CHECK:STDOUT:   .b = %.loc23_28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_19.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc12_19.2: type = converted %int.make_type_32, %.loc12_19.1 [template = i32]
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc12_25: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   assign %v.var, %.loc12_25
// CHECK:STDOUT:   %.loc20: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %return
// CHECK:STDOUT:   %.loc25_29: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc25_37: i32 = int_literal 2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc25_38.1: %.5 = struct_literal (%.loc25_29, %.loc25_37)
// CHECK:STDOUT:   %.loc25_38.2: ref i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc25_38.3: init i32 = initialize_from %.loc25_29 to %.loc25_38.2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc25_38.4: ref i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc25_38.5: init i32 = initialize_from %.loc25_37 to %.loc25_38.4 [template = constants.%.8]
// CHECK:STDOUT:   %.loc25_38.6: init %C = class_init (%.loc25_38.3, %.loc25_38.5), %return [template = constants.%struct]
// CHECK:STDOUT:   %.loc25_39: init %C = converted %.loc25_38.1, %.loc25_38.6 [template = constants.%struct]
// CHECK:STDOUT:   assign %return, %.loc25_39
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
