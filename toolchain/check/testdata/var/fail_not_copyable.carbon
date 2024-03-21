// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class X {
}

fn F(x: X) {
  // TODO: Strings should eventually be copyable, once we decide how to
  // represent them.
  // CHECK:STDERR: fail_not_copyable.carbon:[[@LINE+4]]:19: ERROR: Cannot copy value of type `String`.
  // CHECK:STDERR:   var s: String = "hello";
  // CHECK:STDERR:                   ^~~~~~~
  // CHECK:STDERR:
  var s: String = "hello";

  // TODO: Decide on rules for when classes are copyable.
  // CHECK:STDERR: fail_not_copyable.carbon:[[@LINE+3]]:14: ERROR: Cannot copy value of type `X`.
  // CHECK:STDERR:   var y: X = x;
  // CHECK:STDERR:              ^
  var y: X = x;
}

// CHECK:STDOUT: --- fail_not_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = ptr_type String [template]
// CHECK:STDOUT:   %.5: String = string_literal "hello" [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %X.ref: type = name_ref X, %X.decl [template = constants.%X]
// CHECK:STDOUT:     %x.loc10_6.1: X = param x
// CHECK:STDOUT:     @F.%x: X = bind_name x, %x.loc10_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x: X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref String = var s
// CHECK:STDOUT:   %s: ref String = bind_name s, %s.var
// CHECK:STDOUT:   %.loc17: String = string_literal "hello" [template = constants.%.5]
// CHECK:STDOUT:   assign %s.var, <error>
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %y.var: ref X = var y
// CHECK:STDOUT:   %y: ref X = bind_name y, %y.var
// CHECK:STDOUT:   %x.ref: X = name_ref x, %x
// CHECK:STDOUT:   assign %y.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
