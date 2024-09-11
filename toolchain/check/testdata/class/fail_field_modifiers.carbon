// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_field_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_field_modifiers.carbon

class Class {

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+4]]:3: ERROR: `default` not allowed on `var` declaration.
  // CHECK:STDERR:   default var j: i32;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default var j: i32;

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+4]]:3: ERROR: `final` not allowed on `var` declaration.
  // CHECK:STDERR:   final var k: i32;
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  final var k: i32;

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+4]]:3: ERROR: `default` not allowed on `let` declaration outside of an interface.
  // CHECK:STDERR:   default let l: i32 = 0;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default let l: i32 = 0;

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+3]]:3: ERROR: `final` not allowed on `let` declaration outside of an interface.
  // CHECK:STDERR:   final let m: i32 = 1;
  // CHECK:STDERR:   ^~~~~
  final let m: i32 = 1;
}

// CHECK:STDOUT: --- fail_field_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Class, i32 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.j: i32, .k: i32} [template]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %int.make_type_32.loc17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc17_18.1: type = value_of_initializer %int.make_type_32.loc17 [template = i32]
// CHECK:STDOUT:   %.loc17_18.2: type = converted %int.make_type_32.loc17, %.loc17_18.1 [template = i32]
// CHECK:STDOUT:   %.loc17_16: %.2 = field_decl j, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc23: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc23_16.1: type = value_of_initializer %int.make_type_32.loc23 [template = i32]
// CHECK:STDOUT:   %.loc23_16.2: type = converted %int.make_type_32.loc23, %.loc23_16.1 [template = i32]
// CHECK:STDOUT:   %.loc23_14: %.2 = field_decl k, element1 [template]
// CHECK:STDOUT:   %int.make_type_32.loc29: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc29_18.1: type = value_of_initializer %int.make_type_32.loc29 [template = i32]
// CHECK:STDOUT:   %.loc29_18.2: type = converted %int.make_type_32.loc29, %.loc29_18.1 [template = i32]
// CHECK:STDOUT:   %.loc29_24: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %l: i32 = bind_name l, %.loc29_24
// CHECK:STDOUT:   %int.make_type_32.loc34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc34_16.1: type = value_of_initializer %int.make_type_32.loc34 [template = i32]
// CHECK:STDOUT:   %.loc34_16.2: type = converted %int.make_type_32.loc34, %.loc34_16.1 [template = i32]
// CHECK:STDOUT:   %.loc34_22: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %m: i32 = bind_name m, %.loc34_22
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .j = %.loc17_16
// CHECK:STDOUT:   .k = %.loc23_14
// CHECK:STDOUT:   .l = %l
// CHECK:STDOUT:   .m = %m
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
