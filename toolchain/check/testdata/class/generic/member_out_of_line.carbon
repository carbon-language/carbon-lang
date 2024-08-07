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
  fn G[self: Self]() -> T;
  var n: T;
}

fn Class(T:! type).F(n: T) -> T {
  return n;
}

fn Class(T:! type).G[self: Self]() -> T {
  return self.n;
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
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @Class(%T) [symbolic]
// CHECK:STDOUT:   %G: %G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Class.2, %T [symbolic]
// CHECK:STDOUT:   %.3: type = struct_type {.n: %T} [symbolic]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc4_13.1: type = param T
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T 0, %T.loc4_13.1 [symbolic = @Class.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [symbolic = constants.%F] {
// CHECK:STDOUT:     %T.loc10_10.1: type = param T
// CHECK:STDOUT:     %T.loc10_10.2: type = bind_symbolic_name T 0, %T.loc10_10.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %T.ref.loc10_25: type = name_ref T, %T.loc10_10.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %n.loc10_22.1: %T = param n
// CHECK:STDOUT:     @F.%n: %T = bind_name n, %n.loc10_22.1
// CHECK:STDOUT:     %T.ref.loc10_31: type = name_ref T, %T.loc10_10.2 [symbolic = constants.%T]
// CHECK:STDOUT:     @F.%return: ref %T = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [symbolic = constants.%G] {
// CHECK:STDOUT:     %T.loc14_10.1: type = param T
// CHECK:STDOUT:     %T.loc14_10.2: type = bind_symbolic_name T 0, %T.loc14_10.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %.loc14: type = specific_constant constants.%Class.2, @Class(constants.%T) [symbolic = constants.%Class.2]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, %.loc14 [symbolic = constants.%Class.2]
// CHECK:STDOUT:     %self.loc14_22.1: %Class.2 = param self
// CHECK:STDOUT:     @G.%self: %Class.2 = bind_name self, %self.loc14_22.1
// CHECK:STDOUT:     %T.ref.loc14: type = name_ref T, %T.loc14_10.2 [symbolic = constants.%T]
// CHECK:STDOUT:     @G.%return: ref %T = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(file.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @Class.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @Class(%T) [symbolic = %G.type (constants.%G.type)]
// CHECK:STDOUT:   %G: @Class.%G.type (%G.type) = struct_value () [symbolic = %G (constants.%G)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.1: type = unbound_element_type @Class.%Class (%Class.2), @Class.%T (%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @Class.%F.type (%F.type) = fn_decl @F [symbolic = %F (constants.%F)] {
// CHECK:STDOUT:       %T.ref.loc5_11: type = name_ref T, file.%T.loc4_13.2 [symbolic = @F.%T (constants.%T)]
// CHECK:STDOUT:       %n.loc5_8.1: @F.%T (%T) = param n
// CHECK:STDOUT:       %n.loc5_8.2: @F.%T (%T) = bind_name n, %n.loc5_8.1
// CHECK:STDOUT:       %T.ref.loc5_17: type = name_ref T, file.%T.loc4_13.2 [symbolic = @F.%T (constants.%T)]
// CHECK:STDOUT:       %return.var.loc5: ref @F.%T (%T) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %G.decl: @Class.%G.type (%G.type) = fn_decl @G [symbolic = %G (constants.%G)] {
// CHECK:STDOUT:       %.loc6: type = specific_constant constants.%Class.2, @Class(constants.%T) [symbolic = @G.%Class (constants.%Class.2)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc6 [symbolic = @G.%Class (constants.%Class.2)]
// CHECK:STDOUT:       %self.loc6_8.1: @G.%Class (%Class.2) = param self
// CHECK:STDOUT:       %self.loc6_8.2: @G.%Class (%Class.2) = bind_name self, %self.loc6_8.1
// CHECK:STDOUT:       %T.ref.loc6: type = name_ref T, file.%T.loc4_13.2 [symbolic = @G.%T (constants.%T)]
// CHECK:STDOUT:       %return.var.loc6: ref @G.%T (%T) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.ref.loc7: type = name_ref T, file.%T.loc4_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %.loc7: @Class.%.1 (%.2) = field_decl n, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .n = %.loc7
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(file.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n: %T) -> %T {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @F.%T (%T) = name_ref n, %n
// CHECK:STDOUT:     return %n.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(file.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = unbound_element_type @G.%Class (%Class.2), @G.%T (%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: %Class.2]() -> %T {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @G.%Class (%Class.2) = name_ref self, %self
// CHECK:STDOUT:     %n.ref: @G.%.1 (%.2) = name_ref n, @Class.%.loc7 [template = @Class.%.loc7]
// CHECK:STDOUT:     %.loc15_14.1: ref @G.%T (%T) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc15_14.2: @G.%T (%T) = bind_value %.loc15_14.1
// CHECK:STDOUT:     return %.loc15_14.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %G.type => constants.%G.type
// CHECK:STDOUT:   %G => constants.%G
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@G.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Class.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A.1: %A.type = struct_value () [template]
// CHECK:STDOUT:   %A.2: type = class_type @A, @A(%T) [symbolic]
// CHECK:STDOUT:   %N: %T = bind_symbolic_name N 1 [symbolic]
// CHECK:STDOUT:   %B.type: type = generic_class_type @B [template]
// CHECK:STDOUT:   %B.1: %B.type = struct_value () [template]
// CHECK:STDOUT:   %B.2: type = class_type @B, @B(%T, %N) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @B(%T, %N) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [template = constants.%A.1] {
// CHECK:STDOUT:     %T.loc4_9.1: type = param T
// CHECK:STDOUT:     %T.loc4_9.2: type = bind_symbolic_name T 0, %T.loc4_9.1 [symbolic = @A.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [symbolic = constants.%F] {
// CHECK:STDOUT:     %T.loc10_6.1: type = param T
// CHECK:STDOUT:     %T.loc10_6.2: type = bind_symbolic_name T 0, %T.loc10_6.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %T.ref.loc10_22: type = name_ref T, %T.loc10_6.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %N.loc10_18.1: %T = param N
// CHECK:STDOUT:     %N.loc10_18.2: %T = bind_symbolic_name N 1, %N.loc10_18.1 [symbolic = constants.%N]
// CHECK:STDOUT:     %.loc10: type = specific_constant constants.%B.2, @B(constants.%T, constants.%N) [symbolic = constants.%B.2]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, %.loc10 [symbolic = constants.%B.2]
// CHECK:STDOUT:     %self.loc10_27.1: %B.2 = param self
// CHECK:STDOUT:     @F.%self: %B.2 = bind_name self, %self.loc10_27.1
// CHECK:STDOUT:     %T.ref.loc10_42: type = name_ref T, %T.loc10_6.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %a.loc10_39.1: %T = param a
// CHECK:STDOUT:     @F.%a: %T = bind_name a, %a.loc10_39.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(file.%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.decl: %B.type = class_decl @B [template = constants.%B.1] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, file.%T.loc4_9.2 [symbolic = @B.%T (constants.%T)]
// CHECK:STDOUT:       %N.loc5_11.1: @B.%T (%T) = param N
// CHECK:STDOUT:       %N.loc5_11.2: @B.%T (%T) = bind_symbolic_name N 1, %N.loc5_11.1 [symbolic = @B.%N (constants.%N)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.2
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @B(file.%T.loc4_9.2: type, @A.%N.loc5_11.2: @B.%T (%T)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N: %T = bind_symbolic_name N 1 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F, @B(%T, %N) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @B.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @B.%F.type (%F.type) = fn_decl @F [symbolic = %F (constants.%F)] {
// CHECK:STDOUT:       %.loc6: type = specific_constant constants.%B.2, @B(constants.%T, constants.%N) [symbolic = @F.%B (constants.%B.2)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc6 [symbolic = @F.%B (constants.%B.2)]
// CHECK:STDOUT:       %self.loc6_10.1: @F.%B (%B.2) = param self
// CHECK:STDOUT:       %self.loc6_10.2: @F.%B (%B.2) = bind_name self, %self.loc6_10.1
// CHECK:STDOUT:       %T.ref: type = name_ref T, file.%T.loc4_9.2 [symbolic = @F.%T (constants.%T)]
// CHECK:STDOUT:       %a.loc6_22.1: @F.%T (%T) = param a
// CHECK:STDOUT:       %a.loc6_22.2: @F.%T (%T) = bind_name a, %a.loc6_22.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%B.2
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(file.%T.loc4_9.2: type, @A.%N.loc5_11.2: @B.%T (%T)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N: %T = bind_symbolic_name N 1 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %B: type = class_type @B, @B(%T, %N) [symbolic = %B (constants.%B.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: %B.2](%a: %T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(@F.%T, @F.%N) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %B => constants.%B.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(@B.%T, @B.%N) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N => constants.%N
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
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NotGeneric = %NotGeneric.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NotGeneric.decl: type = class_decl @NotGeneric [template = constants.%NotGeneric] {}
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.3] {
// CHECK:STDOUT:     %T.loc15_15.1: type = param T
// CHECK:STDOUT:     %T.loc15_15.2: type = bind_symbolic_name T 0, %T.loc15_15.1 [symbolic = @.1.%T (constants.%T)]
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
// CHECK:STDOUT: generic fn @.1(file.%T.loc15_15.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_too_few_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Generic.type: type = generic_class_type @Generic [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Generic.1: %Generic.type = struct_value () [template]
// CHECK:STDOUT:   %Generic.2: type = class_type @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %TooFew.type: type = fn_type @TooFew, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %TooFew: %TooFew.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.3: %.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Generic = %Generic.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Generic.decl: %Generic.type = class_decl @Generic [template = constants.%Generic.1] {
// CHECK:STDOUT:     %T.loc4_15.1: type = param T
// CHECK:STDOUT:     %T.loc4_15.2: type = bind_symbolic_name T 0, %T.loc4_15.1 [symbolic = @Generic.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.3] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Generic(file.%T.loc4_15.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %TooFew.type: type = fn_type @TooFew, @Generic(%T) [symbolic = %TooFew.type (constants.%TooFew.type)]
// CHECK:STDOUT:   %TooFew: @Generic.%TooFew.type (%TooFew.type) = struct_value () [symbolic = %TooFew (constants.%TooFew)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %TooFew.decl: @Generic.%TooFew.type (%TooFew.type) = fn_decl @TooFew [symbolic = %TooFew (constants.%TooFew)] {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Generic.2
// CHECK:STDOUT:     .TooFew = %TooFew.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TooFew(file.%T.loc4_15.2: type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TooFew(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(@Generic.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_too_many_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Generic.type: type = generic_class_type @Generic [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Generic.1: %Generic.type = struct_value () [template]
// CHECK:STDOUT:   %Generic.2: type = class_type @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %TooMany.type: type = fn_type @TooMany, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %TooMany: %TooMany.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 1 [symbolic]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.3: %.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Generic = %Generic.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Generic.decl: %Generic.type = class_decl @Generic [template = constants.%Generic.1] {
// CHECK:STDOUT:     %T.loc4_15.1: type = param T
// CHECK:STDOUT:     %T.loc4_15.2: type = bind_symbolic_name T 0, %T.loc4_15.1 [symbolic = @Generic.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.3] {
// CHECK:STDOUT:     %T.loc15_12.1: type = param T
// CHECK:STDOUT:     %T.loc15_12.2: type = bind_symbolic_name T 0, %T.loc15_12.1 [symbolic = @.1.%T (constants.%T)]
// CHECK:STDOUT:     %U.loc15_22.1: type = param U
// CHECK:STDOUT:     %U.loc15_22.2: type = bind_symbolic_name U 1, %U.loc15_22.1 [symbolic = @.1.%U (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Generic(file.%T.loc4_15.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %TooMany.type: type = fn_type @TooMany, @Generic(%T) [symbolic = %TooMany.type (constants.%TooMany.type)]
// CHECK:STDOUT:   %TooMany: @Generic.%TooMany.type (%TooMany.type) = struct_value () [symbolic = %TooMany (constants.%TooMany)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %TooMany.decl: @Generic.%TooMany.type (%TooMany.type) = fn_decl @TooMany [symbolic = %TooMany (constants.%TooMany)] {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Generic.2
// CHECK:STDOUT:     .TooMany = %TooMany.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TooMany(file.%T.loc4_15.2: type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @.1(file.%T.loc15_12.2: type, file.%U.loc15_22.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TooMany(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(@Generic.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_wrong_arg_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Generic.type: type = generic_class_type @Generic [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Generic.1: %Generic.type = struct_value () [template]
// CHECK:STDOUT:   %Generic.2: type = class_type @Generic, @Generic(%T.1) [symbolic]
// CHECK:STDOUT:   %WrongType.type: type = fn_type @WrongType, @Generic(%T.1) [symbolic]
// CHECK:STDOUT:   %WrongType: %WrongType.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %T.2: %.1 = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.3: %.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Generic = %Generic.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Generic.decl: %Generic.type = class_decl @Generic [template = constants.%Generic.1] {
// CHECK:STDOUT:     %T.loc4_15.1: type = param T
// CHECK:STDOUT:     %T.loc4_15.2: type = bind_symbolic_name T 0, %T.loc4_15.1 [symbolic = @Generic.%T (constants.%T.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.3] {
// CHECK:STDOUT:     %.loc14_17.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc14_17.2: type = converted %.loc14_17.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %T.loc14_12.1: %.1 = param T
// CHECK:STDOUT:     %T.loc14_12.2: %.1 = bind_symbolic_name T 0, %T.loc14_12.1 [symbolic = @.1.%T (constants.%T.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Generic(file.%T.loc4_15.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %WrongType.type: type = fn_type @WrongType, @Generic(%T) [symbolic = %WrongType.type (constants.%WrongType.type)]
// CHECK:STDOUT:   %WrongType: @Generic.%WrongType.type (%WrongType.type) = struct_value () [symbolic = %WrongType (constants.%WrongType)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %WrongType.decl: @Generic.%WrongType.type (%WrongType.type) = fn_decl @WrongType [symbolic = %WrongType (constants.%WrongType)] {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Generic.2
// CHECK:STDOUT:     .WrongType = %WrongType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WrongType(file.%T.loc4_15.2: type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @.1(file.%T.loc14_12.2: %.1) {
// CHECK:STDOUT:   %T: %.1 = bind_symbolic_name T 0 [symbolic = %T (constants.%T.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T.1) {
// CHECK:STDOUT:   %T => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrongType(constants.%T.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(@Generic.%T) {
// CHECK:STDOUT:   %T => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%T.2) {
// CHECK:STDOUT:   %T => constants.%T.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
