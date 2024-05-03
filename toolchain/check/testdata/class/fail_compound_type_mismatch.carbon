// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class A {
  var a: i32;
}

class B {
  var b: i32;
}

fn AccessBInA(a: A) -> i32 {
  // CHECK:STDERR: fail_compound_type_mismatch.carbon:[[@LINE+3]]:10: ERROR: Cannot implicitly convert from `A` to `B`.
  // CHECK:STDERR:   return a.(B.b);
  // CHECK:STDERR:          ^~~~~~~
  return a.(B.b);
}

// CHECK:STDOUT: --- fail_compound_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type A, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type B, i32 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.b: i32} [template]
// CHECK:STDOUT:   %AccessBInA: type = fn_type @AccessBInA [template]
// CHECK:STDOUT:   %.5: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: AccessBInA = struct_value () [template]
// CHECK:STDOUT:   %.6: type = ptr_type {.a: i32} [template]
// CHECK:STDOUT:   %.7: type = ptr_type {.b: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .AccessBInA = %AccessBInA.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %AccessBInA.decl: AccessBInA = fn_decl @AccessBInA [template = constants.%struct] {
// CHECK:STDOUT:     %A.ref: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %a.loc15_15.1: A = param a
// CHECK:STDOUT:     @AccessBInA.%a: A = bind_name a, %a.loc15_15.1
// CHECK:STDOUT:     @AccessBInA.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc8: <unbound element of class A> = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .a = %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %.loc12: <unbound element of class B> = field_decl b, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .b = %.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessBInA(%a: A) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: A = name_ref a, %a
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %b.ref: <unbound element of class B> = name_ref b, @B.%.loc12 [template = @B.%.loc12]
// CHECK:STDOUT:   %.loc19: i32 = class_element_access <error>, element0 [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
