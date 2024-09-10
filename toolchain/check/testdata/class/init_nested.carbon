// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/init_nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/init_nested.carbon

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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Inner, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %MakeInner.type: type = fn_type @MakeInner [template]
// CHECK:STDOUT:   %MakeInner: %MakeInner.type = struct_value () [template]
// CHECK:STDOUT:   %Outer: type = class_type @Outer [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %Outer, %Inner [template]
// CHECK:STDOUT:   %.6: type = struct_type {.c: %Inner, .d: %Inner} [template]
// CHECK:STDOUT:   %MakeOuter.type: type = fn_type @MakeOuter [template]
// CHECK:STDOUT:   %MakeOuter: %MakeOuter.type = struct_value () [template]
// CHECK:STDOUT:   %.7: type = struct_type {.c: %.4, .d: %.4} [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.7 [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.6 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:     .MakeInner = %MakeInner.decl
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:     .MakeOuter = %MakeOuter.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Inner.decl: type = class_decl @Inner [template = constants.%Inner] {}
// CHECK:STDOUT:   %MakeInner.decl: %MakeInner.type = fn_decl @MakeInner [template = constants.%MakeInner] {
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:     @MakeInner.%return: ref %Inner = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Outer.decl: type = class_decl @Outer [template = constants.%Outer] {}
// CHECK:STDOUT:   %MakeOuter.decl: %MakeOuter.type = fn_decl @MakeOuter [template = constants.%MakeOuter] {
// CHECK:STDOUT:     %Outer.ref: type = name_ref Outer, %Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:     @MakeOuter.%return: ref %Outer = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Inner {
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_32.loc12, %.loc12_10.1 [template = i32]
// CHECK:STDOUT:   %.loc12_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_10.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_10.2: type = converted %int.make_type_32.loc13, %.loc13_10.1 [template = i32]
// CHECK:STDOUT:   %.loc13_8: %.2 = field_decl b, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Inner
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT:   .b = %.loc13_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Outer {
// CHECK:STDOUT:   %Inner.ref.loc19: type = name_ref Inner, file.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %.loc19: %.5 = field_decl c, element0 [template]
// CHECK:STDOUT:   %Inner.ref.loc20: type = name_ref Inner, file.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %.loc20: %.5 = field_decl d, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Outer
// CHECK:STDOUT:   .c = %.loc19
// CHECK:STDOUT:   .d = %.loc20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeInner() -> %Inner;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeOuter() -> %return: %Outer {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MakeInner.ref.loc24_16: %MakeInner.type = name_ref MakeInner, file.%MakeInner.decl [template = constants.%MakeInner]
// CHECK:STDOUT:   %.loc24_45.1: ref %Inner = class_element_access %return, element0
// CHECK:STDOUT:   %MakeInner.call.loc24_25: init %Inner = call %MakeInner.ref.loc24_16() to %.loc24_45.1
// CHECK:STDOUT:   %MakeInner.ref.loc24_34: %MakeInner.type = name_ref MakeInner, file.%MakeInner.decl [template = constants.%MakeInner]
// CHECK:STDOUT:   %.loc24_45.2: ref %Inner = class_element_access %return, element1
// CHECK:STDOUT:   %MakeInner.call.loc24_43: init %Inner = call %MakeInner.ref.loc24_34() to %.loc24_45.2
// CHECK:STDOUT:   %.loc24_45.3: %.6 = struct_literal (%MakeInner.call.loc24_25, %MakeInner.call.loc24_43)
// CHECK:STDOUT:   %.loc24_45.4: init %Outer = class_init (%MakeInner.call.loc24_25, %MakeInner.call.loc24_43), %return
// CHECK:STDOUT:   %.loc24_46: init %Outer = converted %.loc24_45.3, %.loc24_45.4
// CHECK:STDOUT:   return %.loc24_46 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
