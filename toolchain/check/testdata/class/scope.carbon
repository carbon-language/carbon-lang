// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/scope.carbon

class Class {
  fn F() -> i32 {
    return 1;
  }

  fn G() -> i32 {
    return F();
  }
}

fn F() -> i32 {
  return 2;
}

fn Run() {
  var a: i32 = F();
  var b: i32 = Class.F();
}

// CHECK:STDOUT: --- scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.2 [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc21_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc21_11.2: type = converted %int.make_type_32, %.loc21_11.1 [template = i32]
// CHECK:STDOUT:     @F.2.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [template = constants.%Run] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc12_13.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:     %.loc12_13.2: type = converted %int.make_type_32.loc12, %.loc12_13.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc12: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %int.make_type_32.loc16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_13.1: type = value_of_initializer %int.make_type_32.loc16 [template = i32]
// CHECK:STDOUT:     %.loc16_13.2: type = converted %int.make_type_32.loc16, %.loc16_13.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc16: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   return %.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type.1 = name_ref F, @Class.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %F.call: init i32 = call %F.ref()
// CHECK:STDOUT:   %.loc17_15.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc17_15.2: i32 = converted %F.call, %.loc17_15.1
// CHECK:STDOUT:   return %.loc17_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   return %.loc22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc26: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc26_10.1: type = value_of_initializer %int.make_type_32.loc26 [template = i32]
// CHECK:STDOUT:   %.loc26_10.2: type = converted %int.make_type_32.loc26, %.loc26_10.1 [template = i32]
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref.loc26: %F.type.2 = name_ref F, file.%F.decl [template = constants.%F.2]
// CHECK:STDOUT:   %F.call.loc26: init i32 = call %F.ref.loc26()
// CHECK:STDOUT:   assign %a.var, %F.call.loc26
// CHECK:STDOUT:   %int.make_type_32.loc27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc27_10.1: type = value_of_initializer %int.make_type_32.loc27 [template = i32]
// CHECK:STDOUT:   %.loc27_10.2: type = converted %int.make_type_32.loc27, %.loc27_10.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %F.ref.loc27: %F.type.1 = name_ref F, @Class.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %F.call.loc27: init i32 = call %F.ref.loc27()
// CHECK:STDOUT:   assign %b.var, %F.call.loc27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
