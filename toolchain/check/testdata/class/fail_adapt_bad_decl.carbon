// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fail_not_type.carbon

library "not_type" api;

class Bad {
  // CHECK:STDERR: fail_not_type.carbon:[[@LINE+4]]:3: ERROR: Cannot implicitly convert from `i32` to `type`.
  // CHECK:STDERR:   adapt 100;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  adapt 100;
}

// CHECK:STDERR: fail_not_type.carbon:[[@LINE+4]]:18: ERROR: Name `F` not found.
// CHECK:STDERR: fn Use(b: Bad) { b.F(); }
// CHECK:STDERR:                  ^~~
// CHECK:STDERR:
fn Use(b: Bad) { b.F(); }

// --- fail_extend_not_type.carbon

library "extend_not_type" api;

class Bad {
  // CHECK:STDERR: fail_extend_not_type.carbon:[[@LINE+4]]:3: ERROR: Cannot implicitly convert from `i32` to `type`.
  // CHECK:STDERR:   extend adapt 100;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend adapt 100;
}

// No diagnostic here, we don't know what names Bad has.
fn Use(b: Bad) { b.F(); }

// --- fail_repeated.carbon

library "fail_repeated" api;

class MultipleAdapts {
  adapt ();
  // CHECK:STDERR: fail_repeated.carbon:[[@LINE+7]]:3: ERROR: Multiple `adapt` declarations in class.
  // CHECK:STDERR:   adapt {};
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_repeated.carbon:[[@LINE-4]]:3: Previous `adapt` declaration is here.
  // CHECK:STDERR:   adapt ();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  adapt {};
}

class MultipleAdaptsSameType {
  adapt ();
  // CHECK:STDERR: fail_repeated.carbon:[[@LINE+7]]:3: ERROR: Multiple `adapt` declarations in class.
  // CHECK:STDERR:   adapt ();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_repeated.carbon:[[@LINE-4]]:3: Previous `adapt` declaration is here.
  // CHECK:STDERR:   adapt ();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  adapt ();
}

// --- fail_bad_scope.carbon

library "fail_bad_scope" api;

// CHECK:STDERR: fail_bad_scope.carbon:[[@LINE+4]]:1: ERROR: `adapt` declaration can only be used in a class.
// CHECK:STDERR: adapt {};
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
adapt {};

interface I {
  // CHECK:STDERR: fail_bad_scope.carbon:[[@LINE+4]]:3: ERROR: `adapt` declaration can only be used in a class.
  // CHECK:STDERR:   adapt {};
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  adapt {};
}

class C {
  interface I {
    // CHECK:STDERR: fail_bad_scope.carbon:[[@LINE+3]]:5: ERROR: `adapt` declaration can only be used in a class.
    // CHECK:STDERR:     adapt {};
    // CHECK:STDERR:     ^~~~~~~~~
    adapt {};
  }
}

// CHECK:STDOUT: --- fail_not_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bad: type = class_type @Bad [template]
// CHECK:STDOUT:   %.1: i32 = int_literal 100 [template]
// CHECK:STDOUT:   %Use: type = fn_type @Use [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Use = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Bad = %Bad.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Bad.decl: type = class_decl @Bad [template = constants.%Bad] {}
// CHECK:STDOUT:   %Use.decl: Use = fn_decl @Use [template = constants.%struct] {
// CHECK:STDOUT:     %Bad.ref: type = name_ref Bad, %Bad.decl [template = constants.%Bad]
// CHECK:STDOUT:     %b.loc16_8.1: Bad = param b
// CHECK:STDOUT:     @Use.%b: Bad = bind_name b, %b.loc16_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Bad {
// CHECK:STDOUT:   %.loc9: i32 = int_literal 100 [template = constants.%.1]
// CHECK:STDOUT:   adapt_decl <error>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Bad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use(%b: Bad) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: Bad = name_ref b, %b
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extend_not_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bad: type = class_type @Bad [template]
// CHECK:STDOUT:   %.1: i32 = int_literal 100 [template]
// CHECK:STDOUT:   %Use: type = fn_type @Use [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Use = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Bad = %Bad.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Bad.decl: type = class_decl @Bad [template = constants.%Bad] {}
// CHECK:STDOUT:   %Use.decl: Use = fn_decl @Use [template = constants.%struct] {
// CHECK:STDOUT:     %Bad.ref: type = name_ref Bad, %Bad.decl [template = constants.%Bad]
// CHECK:STDOUT:     %b.loc13_8.1: Bad = param b
// CHECK:STDOUT:     @Use.%b: Bad = bind_name b, %b.loc13_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Bad {
// CHECK:STDOUT:   %.loc9: i32 = int_literal 100 [template = constants.%.1]
// CHECK:STDOUT:   adapt_decl <error>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Bad
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use(%b: Bad) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: Bad = name_ref b, %b
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_repeated.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MultipleAdapts: type = class_type @MultipleAdapts [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %MultipleAdaptsSameType: type = class_type @MultipleAdaptsSameType [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .MultipleAdapts = %MultipleAdapts.decl
// CHECK:STDOUT:     .MultipleAdaptsSameType = %MultipleAdaptsSameType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %MultipleAdapts.decl: type = class_decl @MultipleAdapts [template = constants.%MultipleAdapts] {}
// CHECK:STDOUT:   %MultipleAdaptsSameType.decl: type = class_decl @MultipleAdaptsSameType [template = constants.%MultipleAdaptsSameType] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @MultipleAdapts {
// CHECK:STDOUT:   %.loc5_10: () = tuple_literal ()
// CHECK:STDOUT:   %.loc5_11: type = converted %.loc5_10, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   adapt_decl ()
// CHECK:STDOUT:   %.loc13: {} = struct_literal ()
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%MultipleAdapts
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @MultipleAdaptsSameType {
// CHECK:STDOUT:   %.loc17_10: () = tuple_literal ()
// CHECK:STDOUT:   %.loc17_11: type = converted %.loc17_10, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   adapt_decl ()
// CHECK:STDOUT:   %.loc25: () = tuple_literal ()
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%MultipleAdaptsSameType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = interface_type @I.1 [template]
// CHECK:STDOUT:   %Self.1: I = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.3: type = interface_type @I.2 [template]
// CHECK:STDOUT:   %Self.2: I = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc8: {} = struct_literal ()
// CHECK:STDOUT:   %I.decl: type = interface_decl @I.1 [template = constants.%.2] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I.1 {
// CHECK:STDOUT:   %Self: I = bind_symbolic_name Self 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %.loc15: {} = struct_literal ()
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I.2 {
// CHECK:STDOUT:   %Self: I = bind_symbolic_name Self 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %.loc23: {} = struct_literal ()
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %I.decl: type = interface_decl @I.2 [template = constants.%.3] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I = %I.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
