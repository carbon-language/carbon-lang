// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Inner {
  var a: i32;
  var b: i32;
}

fn MakeInner() -> Inner;

class Outer {
  var c: Inner;
  var d: Inner;
}

fn MakeOuter() -> Outer {
  return {.c = MakeInner(), .d = MakeInner()};
}

// CHECK:STDOUT: --- init_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Inner: type = class_type @Inner [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type Inner, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %MakeInner: type = fn_type @MakeInner [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: MakeInner = struct_value () [template]
// CHECK:STDOUT:   %Outer: type = class_type @Outer [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type Outer, Inner [template]
// CHECK:STDOUT:   %.6: type = struct_type {.c: Inner, .d: Inner} [template]
// CHECK:STDOUT:   %MakeOuter: type = fn_type @MakeOuter [template]
// CHECK:STDOUT:   %struct.2: MakeOuter = struct_value () [template]
// CHECK:STDOUT:   %.7: type = struct_type {.c: {.a: i32, .b: i32}*, .d: {.a: i32, .b: i32}*} [template]
// CHECK:STDOUT:   %.8: type = ptr_type {.c: {.a: i32, .b: i32}*, .d: {.a: i32, .b: i32}*} [template]
// CHECK:STDOUT:   %.9: type = ptr_type {.c: Inner, .d: Inner} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:     .MakeInner = %MakeInner.decl
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:     .MakeOuter = %MakeOuter.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Inner.decl: type = class_decl @Inner [template = constants.%Inner] {}
// CHECK:STDOUT:   %MakeInner.decl: MakeInner = fn_decl @MakeInner [template = constants.%struct.1] {
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:     @MakeInner.%return: ref Inner = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Outer.decl: type = class_decl @Outer [template = constants.%Outer] {}
// CHECK:STDOUT:   %MakeOuter.decl: MakeOuter = fn_decl @MakeOuter [template = constants.%struct.2] {
// CHECK:STDOUT:     %Outer.ref: type = name_ref Outer, %Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:     @MakeOuter.%return: ref Outer = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Inner {
// CHECK:STDOUT:   %.loc8: <unbound element of class Inner> = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc9: <unbound element of class Inner> = field_decl b, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Inner
// CHECK:STDOUT:   .a = %.loc8
// CHECK:STDOUT:   .b = %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Outer {
// CHECK:STDOUT:   %Inner.ref.loc15: type = name_ref Inner, file.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %.loc15: <unbound element of class Outer> = field_decl c, element0 [template]
// CHECK:STDOUT:   %Inner.ref.loc16: type = name_ref Inner, file.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %.loc16: <unbound element of class Outer> = field_decl d, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Outer
// CHECK:STDOUT:   .c = %.loc15
// CHECK:STDOUT:   .d = %.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeInner() -> Inner;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeOuter() -> %return: Outer {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MakeInner.ref.loc20_16: MakeInner = name_ref MakeInner, file.%MakeInner.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc20_45.1: ref Inner = class_element_access %return, element0
// CHECK:STDOUT:   %MakeInner.call.loc20_25: init Inner = call %MakeInner.ref.loc20_16() to %.loc20_45.1
// CHECK:STDOUT:   %MakeInner.ref.loc20_34: MakeInner = name_ref MakeInner, file.%MakeInner.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc20_45.2: ref Inner = class_element_access %return, element1
// CHECK:STDOUT:   %MakeInner.call.loc20_43: init Inner = call %MakeInner.ref.loc20_34() to %.loc20_45.2
// CHECK:STDOUT:   %.loc20_45.3: {.c: Inner, .d: Inner} = struct_literal (%MakeInner.call.loc20_25, %MakeInner.call.loc20_43)
// CHECK:STDOUT:   %.loc20_45.4: init Outer = class_init (%MakeInner.call.loc20_25, %MakeInner.call.loc20_43), %return
// CHECK:STDOUT:   %.loc20_46: init Outer = converted %.loc20_45.3, %.loc20_45.4
// CHECK:STDOUT:   return %.loc20_46 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
