// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- basic.carbon

library "basic" api;

class SomeClass {
  var a: i32;
  var b: i32;
}

class SomeClassAdapter {
  adapt SomeClass;
}

class StructAdapter {
  adapt {.a: i32, .b: i32};
}

// --- fail_not_extend.carbon

library "fail_not_extend" api;

class Adapted {
  fn F();
}

class AdaptNotExtend {
  adapt Adapted;
}

fn F(a: AdaptNotExtend) {
  // `Adapted` is not extended, so lookup for `F` finds nothing.
  // CHECK:STDERR: fail_not_extend.carbon:[[@LINE+3]]:3: ERROR: Name `F` not found.
  // CHECK:STDERR:   a.F();
  // CHECK:STDERR:   ^~~
  a.F();
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SomeClass: type = class_type @SomeClass [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type SomeClass, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %SomeClassAdapter: type = class_type @SomeClassAdapter [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %StructAdapter: type = class_type @StructAdapter [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .SomeClass = %SomeClass.decl
// CHECK:STDOUT:     .SomeClassAdapter = %SomeClassAdapter.decl
// CHECK:STDOUT:     .StructAdapter = %StructAdapter.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %SomeClass.decl: type = class_decl @SomeClass [template = constants.%SomeClass] {}
// CHECK:STDOUT:   %SomeClassAdapter.decl: type = class_decl @SomeClassAdapter [template = constants.%SomeClassAdapter] {}
// CHECK:STDOUT:   %StructAdapter.decl: type = class_decl @StructAdapter [template = constants.%StructAdapter] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClass {
// CHECK:STDOUT:   %.loc5: <unbound element of class SomeClass> = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc6: <unbound element of class SomeClass> = field_decl b, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClass
// CHECK:STDOUT:   .a = %.loc5
// CHECK:STDOUT:   .b = %.loc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClassAdapter {
// CHECK:STDOUT:   %SomeClass.ref: type = name_ref SomeClass, file.%SomeClass.decl [template = constants.%SomeClass]
// CHECK:STDOUT:   adapt_decl SomeClass
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClassAdapter
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @StructAdapter {
// CHECK:STDOUT:   %.loc14: type = struct_type {.a: i32, .b: i32} [template = constants.%.2]
// CHECK:STDOUT:   adapt_decl {.a: i32, .b: i32}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%StructAdapter
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_not_extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Adapted: type = class_type @Adapted [template]
// CHECK:STDOUT:   %F.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %AdaptNotExtend: type = class_type @AdaptNotExtend [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %F.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Adapted = %Adapted.decl
// CHECK:STDOUT:     .AdaptNotExtend = %AdaptNotExtend.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Adapted.decl: type = class_decl @Adapted [template = constants.%Adapted] {}
// CHECK:STDOUT:   %AdaptNotExtend.decl: type = class_decl @AdaptNotExtend [template = constants.%AdaptNotExtend] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.2 [template = constants.%struct.2] {
// CHECK:STDOUT:     %AdaptNotExtend.ref: type = name_ref AdaptNotExtend, %AdaptNotExtend.decl [template = constants.%AdaptNotExtend]
// CHECK:STDOUT:     %a.loc12_6.1: AdaptNotExtend = param a
// CHECK:STDOUT:     @F.2.%a: AdaptNotExtend = bind_name a, %a.loc12_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapted {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.1 [template = constants.%struct.1] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Adapted
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptNotExtend {
// CHECK:STDOUT:   %Adapted.ref: type = name_ref Adapted, file.%Adapted.decl [template = constants.%Adapted]
// CHECK:STDOUT:   adapt_decl Adapted
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptNotExtend
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%a: AdaptNotExtend) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: AdaptNotExtend = name_ref a, %a
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
