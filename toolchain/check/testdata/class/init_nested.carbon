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
// CHECK:STDOUT:   %.3: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %Outer: type = class_type @Outer [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type Outer, Inner [template]
// CHECK:STDOUT:   %.5: type = struct_type {.c: Inner, .d: Inner} [template]
// CHECK:STDOUT:   %.6: type = struct_type {.c: {.a: i32, .b: i32}*, .d: {.a: i32, .b: i32}*} [template]
// CHECK:STDOUT:   %.7: type = ptr_type {.c: {.a: i32, .b: i32}*, .d: {.a: i32, .b: i32}*} [template]
// CHECK:STDOUT:   %.8: type = ptr_type {.c: Inner, .d: Inner} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Inner = %Inner.decl, .MakeInner = %MakeInner, .Outer = %Outer.decl, .MakeOuter = %MakeOuter} [template]
// CHECK:STDOUT:   %Inner.decl = class_decl @Inner, ()
// CHECK:STDOUT:   %MakeInner: <function> = fn_decl @MakeInner [template]
// CHECK:STDOUT:   %Outer.decl = class_decl @Outer, ()
// CHECK:STDOUT:   %MakeOuter: <function> = fn_decl @MakeOuter [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Inner {
// CHECK:STDOUT:   %.loc8: <unbound element of class Inner> = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc9: <unbound element of class Inner> = field_decl b, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %.loc8
// CHECK:STDOUT:   .b = %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Outer {
// CHECK:STDOUT:   %Inner.ref.loc15: type = name_ref Inner, constants.%Inner [template = constants.%Inner]
// CHECK:STDOUT:   %.loc15: <unbound element of class Outer> = field_decl c, element0 [template]
// CHECK:STDOUT:   %Inner.ref.loc16: type = name_ref Inner, constants.%Inner [template = constants.%Inner]
// CHECK:STDOUT:   %.loc16: <unbound element of class Outer> = field_decl d, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .c = %.loc15
// CHECK:STDOUT:   .d = %.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeInner() -> %return: Inner;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeOuter() -> %return: Outer {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MakeInner.ref.loc20_16: <function> = name_ref MakeInner, file.%MakeInner [template = file.%MakeInner]
// CHECK:STDOUT:   %.loc20_45.1: ref Inner = class_element_access %return, element0
// CHECK:STDOUT:   %.loc20_25: init Inner = call %MakeInner.ref.loc20_16() to %.loc20_45.1
// CHECK:STDOUT:   %MakeInner.ref.loc20_34: <function> = name_ref MakeInner, file.%MakeInner [template = file.%MakeInner]
// CHECK:STDOUT:   %.loc20_45.2: ref Inner = class_element_access %return, element1
// CHECK:STDOUT:   %.loc20_43: init Inner = call %MakeInner.ref.loc20_34() to %.loc20_45.2
// CHECK:STDOUT:   %.loc20_45.3: {.c: Inner, .d: Inner} = struct_literal (%.loc20_25, %.loc20_43)
// CHECK:STDOUT:   %.loc20_45.4: init Outer = class_init (%.loc20_25, %.loc20_43), %return
// CHECK:STDOUT:   %.loc20_45.5: init Outer = converted %.loc20_45.3, %.loc20_45.4
// CHECK:STDOUT:   return %.loc20_45.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
