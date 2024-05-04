// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class(T:! type) {
  fn F(n: T) -> T {
    return n;
  }
}

// CHECK:STDOUT: --- member_inline.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.1: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Class = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: Class = class_decl @Class [template = constants.%struct.1] {
// CHECK:STDOUT:     %T.loc7_13.1: type = param T
// CHECK:STDOUT:     %T.loc7_13.2: type = bind_symbolic_name T 0, %T.loc7_13.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %T.ref.loc8_11: type = name_ref T, file.%T.loc7_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %n.loc8_8.1: T = param n
// CHECK:STDOUT:     %n.loc8_8.2: T = bind_name n, %n.loc8_8.1
// CHECK:STDOUT:     %T.ref.loc8_17: type = name_ref T, file.%T.loc7_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %return.var: ref T = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class.2
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(@Class.%n.loc8_8.2: T) -> T {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: T = name_ref n, @Class.%n.loc8_8.2
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
