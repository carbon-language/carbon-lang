// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/member_out_of_line.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/member_out_of_line.carbon

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

// --- fail_mismatched_not_generic_vs_generic.carbon

library "fail_mismatched_not_generic_vs_generic";

class NotGeneric {
  fn F();
}

// CHECK:STDERR: fail_mismatched_not_generic_vs_generic.carbon:[[@LINE+7]]:4: ERROR: Redeclaration differs because of parameter list.
// CHECK:STDERR: fn NotGeneric(T:! type).F() {}
// CHECK:STDERR:    ^~~~~~~~~~
// CHECK:STDERR: fail_mismatched_not_generic_vs_generic.carbon:[[@LINE-7]]:1: Previously declared without parameter list.
// CHECK:STDERR: class NotGeneric {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NotGeneric(T:! type).F() {}

// --- fail_mismatched_too_few_args.carbon

library "fail_mismatched_too_few_args";

class Generic(T:! type) {
  fn TooFew();
}

// CHECK:STDERR: fail_mismatched_too_few_args.carbon:[[@LINE+7]]:4: ERROR: Redeclaration differs because of parameter count of 0.
// CHECK:STDERR: fn Generic().TooFew() {}
// CHECK:STDERR:    ^~~~~~~
// CHECK:STDERR: fail_mismatched_too_few_args.carbon:[[@LINE-7]]:1: Previously declared with parameter count of 1.
// CHECK:STDERR: class Generic(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Generic().TooFew() {}

// --- fail_mismatched_too_many_args.carbon

library "fail_mismatched_too_many_args";

class Generic(T:! type) {
  fn TooMany();
}

// CHECK:STDERR: fail_mismatched_too_many_args.carbon:[[@LINE+7]]:4: ERROR: Redeclaration differs because of parameter count of 2.
// CHECK:STDERR: fn Generic(T:! type, U:! type).TooMany() {}
// CHECK:STDERR:    ^~~~~~~
// CHECK:STDERR: fail_mismatched_too_many_args.carbon:[[@LINE-7]]:1: Previously declared with parameter count of 1.
// CHECK:STDERR: class Generic(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Generic(T:! type, U:! type).TooMany() {}

// --- fail_mismatched_wrong_arg_type.carbon

library "fail_mismatched_wrong_arg_type";

class Generic(T:! type) {
  fn WrongType();
}

// CHECK:STDERR: fail_mismatched_wrong_arg_type.carbon:[[@LINE+6]]:12: ERROR: Redeclaration differs at parameter 1.
// CHECK:STDERR: fn Generic(T:! ()).WrongType() {}
// CHECK:STDERR:            ^
// CHECK:STDERR: fail_mismatched_wrong_arg_type.carbon:[[@LINE-7]]:15: Previous declaration's corresponding parameter here.
// CHECK:STDERR: class Generic(T:! type) {
// CHECK:STDERR:               ^
fn Generic(T:! ()).WrongType() {}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, (%T) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc4_13.1: type = param T
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T 0, %T.loc4_13.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.loc8_10.1: type = param T
// CHECK:STDOUT:     %T.loc8_10.2: type = bind_symbolic_name T 0, %T.loc8_10.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %T.ref.loc8_25: type = name_ref T, %T.loc8_10.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %n.loc8_22.1: %T = param n
// CHECK:STDOUT:     @F.%n: %T = bind_runtime_name n, %n.loc8_22.1
// CHECK:STDOUT:     %T.ref.loc8_31: type = name_ref T, %T.loc8_10.2 [symbolic = constants.%T]
// CHECK:STDOUT:     @F.%return: ref %T = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class
// CHECK:STDOUT:     generic [file.%T.loc4_13.2: type] {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.ref.loc5_11: type = name_ref T, file.%T.loc4_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %n.loc5_8.1: %T = param n
// CHECK:STDOUT:     %n.loc5_8.2: %T = bind_runtime_name n, %n.loc5_8.1
// CHECK:STDOUT:     %T.ref.loc5_17: type = name_ref T, file.%T.loc4_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %return.var: ref %T = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class.2
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n: %T) -> %T
// CHECK:STDOUT:     generic [file.%T.loc4_13.2: type] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %T = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A.1: %A.type = struct_value () [template]
// CHECK:STDOUT:   %A.2: type = class_type @A, (%T) [symbolic]
// CHECK:STDOUT:   %N: %T = bind_symbolic_name N 1 [symbolic]
// CHECK:STDOUT:   %B.type: type = generic_class_type @B [template]
// CHECK:STDOUT:   %B.1: %B.type = struct_value () [template]
// CHECK:STDOUT:   %B.2: type = class_type @B, (%T, %N) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [template = constants.%A.1] {
// CHECK:STDOUT:     %T.loc4_9.1: type = param T
// CHECK:STDOUT:     %T.loc4_9.2: type = bind_symbolic_name T 0, %T.loc4_9.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.loc10_6.1: type = param T
// CHECK:STDOUT:     %T.loc10_6.2: type = bind_symbolic_name T 0, %T.loc10_6.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %T.ref.loc10_22: type = name_ref T, %T.loc10_6.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %N.loc10_18.1: %T = param N
// CHECK:STDOUT:     %N.loc10_18.2: %T = bind_symbolic_name N 1, %N.loc10_18.1 [symbolic = constants.%N]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B.2 [symbolic = constants.%B.2]
// CHECK:STDOUT:     %self.loc10_27.1: %B.2 = param self
// CHECK:STDOUT:     @F.%self: %B.2 = bind_runtime_name self, %self.loc10_27.1
// CHECK:STDOUT:     %T.ref.loc10_42: type = name_ref T, %T.loc10_6.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %a.loc10_39.1: %T = param a
// CHECK:STDOUT:     @F.%a: %T = bind_runtime_name a, %a.loc10_39.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A
// CHECK:STDOUT:     generic [file.%T.loc4_9.2: type] {
// CHECK:STDOUT:   %B.decl: %B.type = class_decl @B [template = constants.%B.1] {
// CHECK:STDOUT:     %T.ref: type = name_ref T, file.%T.loc4_9.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %N.loc5_11.1: %T = param N
// CHECK:STDOUT:     %N.loc5_11.2: %T = bind_symbolic_name N 1, %N.loc5_11.1 [symbolic = constants.%N]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A.2
// CHECK:STDOUT:   .B = %B.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B
// CHECK:STDOUT:     generic [file.%T.loc4_9.2: type, @A.%N.loc5_11.2: %T] {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B.2 [symbolic = constants.%B.2]
// CHECK:STDOUT:     %self.loc6_10.1: %B.2 = param self
// CHECK:STDOUT:     %self.loc6_10.2: %B.2 = bind_runtime_name self, %self.loc6_10.1
// CHECK:STDOUT:     %T.ref: type = name_ref T, file.%T.loc4_9.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %a.loc6_22.1: %T = param a
// CHECK:STDOUT:     %a.loc6_22.2: %T = bind_runtime_name a, %a.loc6_22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B.2
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: %B.2](%a: %T)
// CHECK:STDOUT:     generic [file.%T.loc4_9.2: type, @A.%N.loc5_11.2: %T] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_not_generic_vs_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NotGeneric: type = class_type @NotGeneric [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.3: %.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .NotGeneric = %NotGeneric.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %NotGeneric.decl: type = class_decl @NotGeneric [template = constants.%NotGeneric] {}
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.3] {
// CHECK:STDOUT:     %T.loc15_15.1: type = param T
// CHECK:STDOUT:     %T.loc15_15.2: type = bind_symbolic_name T 0, %T.loc15_15.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NotGeneric {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NotGeneric
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1()
// CHECK:STDOUT:     generic [file.%T.loc15_15.2: type] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_too_few_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Generic.type: type = generic_class_type @Generic [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Generic.1: %Generic.type = struct_value () [template]
// CHECK:STDOUT:   %Generic.2: type = class_type @Generic, (%T) [symbolic]
// CHECK:STDOUT:   %TooFew.type: type = fn_type @TooFew [template]
// CHECK:STDOUT:   %TooFew: %TooFew.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.3: %.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Generic = %Generic.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Generic.decl: %Generic.type = class_decl @Generic [template = constants.%Generic.1] {
// CHECK:STDOUT:     %T.loc4_15.1: type = param T
// CHECK:STDOUT:     %T.loc4_15.2: type = bind_symbolic_name T 0, %T.loc4_15.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.3] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Generic
// CHECK:STDOUT:     generic [file.%T.loc4_15.2: type] {
// CHECK:STDOUT:   %TooFew.decl: %TooFew.type = fn_decl @TooFew [template = constants.%TooFew] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Generic.2
// CHECK:STDOUT:   .TooFew = %TooFew.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooFew()
// CHECK:STDOUT:     generic [file.%T.loc4_15.2: type];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_too_many_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Generic.type: type = generic_class_type @Generic [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Generic.1: %Generic.type = struct_value () [template]
// CHECK:STDOUT:   %Generic.2: type = class_type @Generic, (%T) [symbolic]
// CHECK:STDOUT:   %TooMany.type: type = fn_type @TooMany [template]
// CHECK:STDOUT:   %TooMany: %TooMany.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 1 [symbolic]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.3: %.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Generic = %Generic.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Generic.decl: %Generic.type = class_decl @Generic [template = constants.%Generic.1] {
// CHECK:STDOUT:     %T.loc4_15.1: type = param T
// CHECK:STDOUT:     %T.loc4_15.2: type = bind_symbolic_name T 0, %T.loc4_15.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.3] {
// CHECK:STDOUT:     %T.loc15_12.1: type = param T
// CHECK:STDOUT:     %T.loc15_12.2: type = bind_symbolic_name T 0, %T.loc15_12.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %U.loc15_22.1: type = param U
// CHECK:STDOUT:     %U.loc15_22.2: type = bind_symbolic_name U 1, %U.loc15_22.1 [symbolic = constants.%U]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Generic
// CHECK:STDOUT:     generic [file.%T.loc4_15.2: type] {
// CHECK:STDOUT:   %TooMany.decl: %TooMany.type = fn_decl @TooMany [template = constants.%TooMany] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Generic.2
// CHECK:STDOUT:   .TooMany = %TooMany.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooMany()
// CHECK:STDOUT:     generic [file.%T.loc4_15.2: type];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1()
// CHECK:STDOUT:     generic [file.%T.loc15_12.2: type, file.%U.loc15_22.2: type] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_wrong_arg_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Generic.type: type = generic_class_type @Generic [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Generic.1: %Generic.type = struct_value () [template]
// CHECK:STDOUT:   %Generic.2: type = class_type @Generic, (%T.1) [symbolic]
// CHECK:STDOUT:   %WrongType.type: type = fn_type @WrongType [template]
// CHECK:STDOUT:   %WrongType: %WrongType.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %T.2: %.1 = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.3: %.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Generic = %Generic.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Generic.decl: %Generic.type = class_decl @Generic [template = constants.%Generic.1] {
// CHECK:STDOUT:     %T.loc4_15.1: type = param T
// CHECK:STDOUT:     %T.loc4_15.2: type = bind_symbolic_name T 0, %T.loc4_15.1 [symbolic = constants.%T.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.3] {
// CHECK:STDOUT:     %.loc14_17.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc14_17.2: type = converted %.loc14_17.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %T.loc14_12.1: %.1 = param T
// CHECK:STDOUT:     %T.loc14_12.2: %.1 = bind_symbolic_name T 0, %T.loc14_12.1 [symbolic = constants.%T.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Generic
// CHECK:STDOUT:     generic [file.%T.loc4_15.2: type] {
// CHECK:STDOUT:   %WrongType.decl: %WrongType.type = fn_decl @WrongType [template = constants.%WrongType] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Generic.2
// CHECK:STDOUT:   .WrongType = %WrongType.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WrongType()
// CHECK:STDOUT:     generic [file.%T.loc4_15.2: type];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1()
// CHECK:STDOUT:     generic [file.%T.loc14_12.2: %.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
