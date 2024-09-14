// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/basic.carbon

class Class {
  fn F(n: i32) -> i32 {
    return n;
  }

  fn G(n: i32) -> i32;

  var k: i32;
}

fn Class.G(n: i32) -> i32 {
  return n;
}

fn Run() -> i32 {
  return Class.F(4);
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Class, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.k: i32} [template]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 4 [template]
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
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %int.make_type_32.loc21_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc21_15.1: type = value_of_initializer %int.make_type_32.loc21_15 [template = i32]
// CHECK:STDOUT:     %.loc21_15.2: type = converted %int.make_type_32.loc21_15, %.loc21_15.1 [template = i32]
// CHECK:STDOUT:     %n.loc21_12.1: i32 = param n, runtime_param0
// CHECK:STDOUT:     @G.%n: i32 = bind_name n, %n.loc21_12.1
// CHECK:STDOUT:     %int.make_type_32.loc21_23: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc21_23.1: type = value_of_initializer %int.make_type_32.loc21_23 [template = i32]
// CHECK:STDOUT:     %.loc21_23.2: type = converted %int.make_type_32.loc21_23, %.loc21_23.1 [template = i32]
// CHECK:STDOUT:     @G.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [template = constants.%Run] {
// CHECK:STDOUT:     %int.make_type_32.loc25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc25_13.1: type = value_of_initializer %int.make_type_32.loc25 [template = i32]
// CHECK:STDOUT:     %.loc25_13.2: type = converted %int.make_type_32.loc25, %.loc25_13.1 [template = i32]
// CHECK:STDOUT:     @Run.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %int.make_type_32.loc12_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc12_11.1: type = value_of_initializer %int.make_type_32.loc12_11 [template = i32]
// CHECK:STDOUT:     %.loc12_11.2: type = converted %int.make_type_32.loc12_11, %.loc12_11.1 [template = i32]
// CHECK:STDOUT:     %n.loc12_8.1: i32 = param n, runtime_param0
// CHECK:STDOUT:     %n.loc12_8.2: i32 = bind_name n, %n.loc12_8.1
// CHECK:STDOUT:     %int.make_type_32.loc12_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc12_19.1: type = value_of_initializer %int.make_type_32.loc12_19 [template = i32]
// CHECK:STDOUT:     %.loc12_19.2: type = converted %int.make_type_32.loc12_19, %.loc12_19.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc12: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %int.make_type_32.loc16_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_11.1: type = value_of_initializer %int.make_type_32.loc16_11 [template = i32]
// CHECK:STDOUT:     %.loc16_11.2: type = converted %int.make_type_32.loc16_11, %.loc16_11.1 [template = i32]
// CHECK:STDOUT:     %n.loc16_8.1: i32 = param n, runtime_param0
// CHECK:STDOUT:     %n.loc16_8.2: i32 = bind_name n, %n.loc16_8.1
// CHECK:STDOUT:     %int.make_type_32.loc16_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_19.1: type = value_of_initializer %int.make_type_32.loc16_19 [template = i32]
// CHECK:STDOUT:     %.loc16_19.2: type = converted %int.make_type_32.loc16_19, %.loc16_19.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc16: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc18_10.1: type = value_of_initializer %int.make_type_32.loc18 [template = i32]
// CHECK:STDOUT:   %.loc18_10.2: type = converted %int.make_type_32.loc18, %.loc18_10.1 [template = i32]
// CHECK:STDOUT:   %.loc18_8: %.2 = field_decl k, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .k = %.loc18_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(@Class.%n.loc12_8.2: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, @Class.%n.loc12_8.2
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc26_18: i32 = int_literal 4 [template = constants.%.5]
// CHECK:STDOUT:   %F.call: init i32 = call %F.ref(%.loc26_18)
// CHECK:STDOUT:   %.loc26_20.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc26_20.2: i32 = converted %F.call, %.loc26_20.1
// CHECK:STDOUT:   return %.loc26_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
