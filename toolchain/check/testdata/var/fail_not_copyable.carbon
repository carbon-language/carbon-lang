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
  // CHECK:STDERR: fail_not_copyable.carbon:[[@LINE+3]]:19: ERROR: Cannot copy value of type `String`.
  // CHECK:STDERR:   var s: String = "hello";
  // CHECK:STDERR:                   ^~~~~~~
  var s: String = "hello";

  // TODO: Decide on rules for when classes are copyable.
  // CHECK:STDERR: fail_not_copyable.carbon:[[@LINE+3]]:14: ERROR: Cannot copy value of type `X`.
  // CHECK:STDERR:   var y: X = x;
  // CHECK:STDERR:              ^
  var y: X = x;
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc8_1.1: type = struct_type {}
// CHECK:STDOUT:   %.loc8_1.2: type = tuple_type ()
// CHECK:STDOUT:   %.loc7: type = ptr_type {}
// CHECK:STDOUT:   %.1: type = ptr_type String
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_not_copyable.carbon" {
// CHECK:STDOUT:   class_decl @X, ()
// CHECK:STDOUT:   %X: type = class_type @X
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x: X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref String = var s
// CHECK:STDOUT:   %s: ref String = bind_name s, %s.var
// CHECK:STDOUT:   %.loc16: String = string_literal "hello"
// CHECK:STDOUT:   assign %s.var, <error>
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X
// CHECK:STDOUT:   %y.var: ref X = var y
// CHECK:STDOUT:   %y: ref X = bind_name y, %y.var
// CHECK:STDOUT:   %x.ref: X = name_ref x, %x
// CHECK:STDOUT:   assign %y.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
