// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- basic.carbon

library "basic";

class Class(T:! type) {
  fn F(n: T) -> T;
}

fn Class(T:! type).F(n: T) -> T {
  return n;
}

// --- nested.carbon

library "nested";

class A(T:! type) {
  class B(N:! T) {
    fn F[self: Self](a: T);
  }
}

fn A(T:! type).B(N:! T).F[self: Self](a: T) {}

// --- fail_mismatched.carbon

library "fail_mismatched";

class NotGeneric {
  fn F();
}

// CHECK:STDERR: fail_mismatched.carbon:[[@LINE+7]]:4: ERROR: Redeclaration differs because of parameter list.
// CHECK:STDERR: fn NotGeneric(T:! type).F() {}
// CHECK:STDERR:    ^~~~~~~~~~
// CHECK:STDERR: fail_mismatched.carbon:[[@LINE-7]]:1: Previously declared without parameter list.
// CHECK:STDERR: class NotGeneric {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NotGeneric(T:! type).F() {}

class Generic(T:! type) {
  fn TooFew();
  fn TooMany();
  fn WrongType();
}

// CHECK:STDERR: fail_mismatched.carbon:[[@LINE+7]]:4: ERROR: Redeclaration differs because of parameter count of 0.
// CHECK:STDERR: fn Generic().TooFew() {}
// CHECK:STDERR:    ^~~~~~~
// CHECK:STDERR: fail_mismatched.carbon:[[@LINE-9]]:1: Previously declared with parameter count of 1.
// CHECK:STDERR: class Generic(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Generic().TooFew() {}

// CHECK:STDERR: fail_mismatched.carbon:[[@LINE+7]]:4: ERROR: Redeclaration differs because of parameter count of 2.
// CHECK:STDERR: fn Generic(T:! type, U:! type).TooMany() {}
// CHECK:STDERR:    ^~~~~~~
// CHECK:STDERR: fail_mismatched.carbon:[[@LINE-18]]:1: Previously declared with parameter count of 1.
// CHECK:STDERR: class Generic(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Generic(T:! type, U:! type).TooMany() {}

// CHECK:STDERR: fail_mismatched.carbon:[[@LINE+6]]:12: ERROR: Redeclaration differs at parameter 1.
// CHECK:STDERR: fn Generic(T:! ()).WrongType() {}
// CHECK:STDERR:            ^
// CHECK:STDERR: fail_mismatched.carbon:[[@LINE-27]]:15: Previous declaration's corresponding parameter here.
// CHECK:STDERR: class Generic(T:! type) {
// CHECK:STDERR:               ^
fn Generic(T:! ()).WrongType() {}

// CHECK:STDOUT: --- basic.carbon
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
// CHECK:STDOUT:     %T.loc4_13.1: type = param T
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T 0, %T.loc4_13.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %T.loc8_10.1: type = param T
// CHECK:STDOUT:     %T.loc8_10.2: type = bind_symbolic_name T 0, %T.loc8_10.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %T.ref.loc8_25: type = name_ref T, %T.loc8_10.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %n.loc8_22.1: T = param n
// CHECK:STDOUT:     @F.%n: T = bind_name n, %n.loc8_22.1
// CHECK:STDOUT:     %T.ref.loc8_31: type = name_ref T, %T.loc8_10.2 [symbolic = constants.%T]
// CHECK:STDOUT:     @F.%return: ref T = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %T.ref.loc5_11: type = name_ref T, file.%T.loc4_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %n.loc5_8.1: T = param n
// CHECK:STDOUT:     %n.loc5_8.2: T = bind_name n, %n.loc5_8.1
// CHECK:STDOUT:     %T.ref.loc5_17: type = name_ref T, file.%T.loc4_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %return.var: ref T = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class.2
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n: T) -> T {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: T = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %A.1: type = generic_class_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %A.2: type = class_type @A [template]
// CHECK:STDOUT:   %N: T = bind_symbolic_name N 1 [symbolic]
// CHECK:STDOUT:   %B.1: type = generic_class_type @B [template]
// CHECK:STDOUT:   %struct.2: B = struct_value () [template]
// CHECK:STDOUT:   %B.2: type = class_type @B [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.3: F = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: A = class_decl @A [template = constants.%struct.1] {
// CHECK:STDOUT:     %T.loc4_9.1: type = param T
// CHECK:STDOUT:     %T.loc4_9.2: type = bind_symbolic_name T 0, %T.loc4_9.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.3] {
// CHECK:STDOUT:     %T.loc10_6.1: type = param T
// CHECK:STDOUT:     %T.loc10_6.2: type = bind_symbolic_name T 0, %T.loc10_6.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %T.ref.loc10_22: type = name_ref T, %T.loc10_6.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %N.loc10_18.1: T = param N
// CHECK:STDOUT:     %N.loc10_18.2: T = bind_symbolic_name N 1, %N.loc10_18.1 [symbolic = constants.%N]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B.2 [template = constants.%B.2]
// CHECK:STDOUT:     %self.loc10_27.1: B = param self
// CHECK:STDOUT:     @F.%self: B = bind_name self, %self.loc10_27.1
// CHECK:STDOUT:     %T.ref.loc10_42: type = name_ref T, %T.loc10_6.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %a.loc10_39.1: T = param a
// CHECK:STDOUT:     @F.%a: T = bind_name a, %a.loc10_39.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %B.decl: B = class_decl @B [template = constants.%struct.2] {
// CHECK:STDOUT:     %T.ref: type = name_ref T, file.%T.loc4_9.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %N.loc5_11.1: T = param N
// CHECK:STDOUT:     %N.loc5_11.2: T = bind_symbolic_name N 1, %N.loc5_11.1 [symbolic = constants.%N]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A.2
// CHECK:STDOUT:   .B = %B.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.3] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B.2 [template = constants.%B.2]
// CHECK:STDOUT:     %self.loc6_10.1: B = param self
// CHECK:STDOUT:     %self.loc6_10.2: B = bind_name self, %self.loc6_10.1
// CHECK:STDOUT:     %T.ref: type = name_ref T, file.%T.loc4_9.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %a.loc6_22.1: T = param a
// CHECK:STDOUT:     %a.loc6_22.2: T = bind_name a, %a.loc6_22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B.2
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: B](%a: T) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NotGeneric: type = class_type @NotGeneric [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.3: type = fn_type @.1 [template]
// CHECK:STDOUT:   %struct.2: <invalid> = struct_value () [template]
// CHECK:STDOUT:   %Generic.1: type = generic_class_type @Generic [template]
// CHECK:STDOUT:   %struct.3: Generic = struct_value () [template]
// CHECK:STDOUT:   %Generic.2: type = class_type @Generic [template]
// CHECK:STDOUT:   %TooFew: type = fn_type @TooFew [template]
// CHECK:STDOUT:   %struct.4: TooFew = struct_value () [template]
// CHECK:STDOUT:   %TooMany: type = fn_type @TooMany [template]
// CHECK:STDOUT:   %struct.5: TooMany = struct_value () [template]
// CHECK:STDOUT:   %WrongType: type = fn_type @WrongType [template]
// CHECK:STDOUT:   %struct.6: WrongType = struct_value () [template]
// CHECK:STDOUT:   %.4: type = fn_type @.2 [template]
// CHECK:STDOUT:   %struct.7: <invalid> = struct_value () [template]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 1 [symbolic]
// CHECK:STDOUT:   %.5: type = fn_type @.3 [template]
// CHECK:STDOUT:   %struct.8: <invalid> = struct_value () [template]
// CHECK:STDOUT:   %T.2: () = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.6: type = fn_type @.4 [template]
// CHECK:STDOUT:   %struct.9: <invalid> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .NotGeneric = %NotGeneric.decl
// CHECK:STDOUT:     .Generic = %Generic.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %NotGeneric.decl: type = class_decl @NotGeneric [template = constants.%NotGeneric] {}
// CHECK:STDOUT:   %.decl.loc15: <invalid> = fn_decl @.1 [template = constants.%struct.2] {
// CHECK:STDOUT:     %T.loc15_15.1: type = param T
// CHECK:STDOUT:     %T.loc15_15.2: type = bind_symbolic_name T 0, %T.loc15_15.1 [symbolic = constants.%T.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.decl: Generic = class_decl @Generic [template = constants.%struct.3] {
// CHECK:STDOUT:     %T.loc17_15.1: type = param T
// CHECK:STDOUT:     %T.loc17_15.2: type = bind_symbolic_name T 0, %T.loc17_15.1 [symbolic = constants.%T.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc30: <invalid> = fn_decl @.2 [template = constants.%struct.7] {}
// CHECK:STDOUT:   %.decl.loc39: <invalid> = fn_decl @.3 [template = constants.%struct.8] {
// CHECK:STDOUT:     %T.loc39_12.1: type = param T
// CHECK:STDOUT:     %T.loc39_12.2: type = bind_symbolic_name T 0, %T.loc39_12.1 [symbolic = constants.%T.1]
// CHECK:STDOUT:     %U.loc39_22.1: type = param U
// CHECK:STDOUT:     %U.loc39_22.2: type = bind_symbolic_name U 1, %U.loc39_22.1 [symbolic = constants.%U]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc47: <invalid> = fn_decl @.4 [template = constants.%struct.9] {
// CHECK:STDOUT:     %.loc47_17.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc47_17.2: type = converted %.loc47_17.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %T.loc47_12.1: () = param T
// CHECK:STDOUT:     %T.loc47_12.2: () = bind_symbolic_name T 0, %T.loc47_12.1 [symbolic = constants.%T.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NotGeneric {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NotGeneric
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Generic {
// CHECK:STDOUT:   %TooFew.decl: TooFew = fn_decl @TooFew [template = constants.%struct.4] {}
// CHECK:STDOUT:   %TooMany.decl: TooMany = fn_decl @TooMany [template = constants.%struct.5] {}
// CHECK:STDOUT:   %WrongType.decl: WrongType = fn_decl @WrongType [template = constants.%struct.6] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Generic.2
// CHECK:STDOUT:   .TooFew = %TooFew.decl
// CHECK:STDOUT:   .TooMany = %TooMany.decl
// CHECK:STDOUT:   .WrongType = %WrongType.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooFew();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooMany();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WrongType();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.3() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.4() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
