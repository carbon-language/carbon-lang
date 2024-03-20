// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C {
  // CHECK:STDERR: fail_convert_to_invalid.carbon:[[@LINE+3]]:10: ERROR: Name `NoSuchType` not found.
  // CHECK:STDERR:   var a: NoSuchType;
  // CHECK:STDERR:          ^~~~~~~~~~
  var a: NoSuchType;
}

fn Make() -> C {
  return {.a = 123};
}

// CHECK:STDOUT: --- fail_convert_to_invalid.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: i32 = int_literal 123 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Make = %Make
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %Make: <function> = fn_decl @Make [template] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @Make.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %NoSuchType.ref: <error> = name_ref NoSuchType, <error> [template = <error>]
// CHECK:STDOUT:   %.loc11: <error> = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .a = %.loc11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_16: i32 = int_literal 123 [template = constants.%.1]
// CHECK:STDOUT:   %.loc15_19: {.a: i32} = struct_literal (%.loc15_16)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
