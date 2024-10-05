// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_todo_define_default_fn_inline.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_todo_define_default_fn_inline.carbon

interface Interface {
  // CHECK:STDERR: fail_todo_define_default_fn_inline.carbon:[[@LINE+4]]:3: error: semantics TODO: `interface modifier`
  // CHECK:STDERR:   default fn F() {}
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn F() {}

  // CHECK:STDERR: fail_todo_define_default_fn_inline.carbon:[[@LINE+3]]:3: error: semantics TODO: `interface modifier`
  // CHECK:STDERR:   default fn G(a: i32, b: i32) -> i32 = "int.sadd";
  // CHECK:STDERR:   ^~~~~~~
  default fn G(a: i32, b: i32) -> i32 = "int.sadd";
}

// CHECK:STDOUT: --- fail_todo_define_default_fn_inline.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = interface_type @Interface [template]
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %Interface.type, %F.type [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @Interface.%F.decl [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %Interface.type, %G.type [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element1, @Interface.%G.decl [template]
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
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%Interface.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %.loc16: %.2 = assoc_entity element0, %F.decl [template = constants.%.3]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %b.patt: i32 = binding_pattern b
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc21_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc21_19.1: type = value_of_initializer %int.make_type_32.loc21_19 [template = i32]
// CHECK:STDOUT:     %.loc21_19.2: type = converted %int.make_type_32.loc21_19, %.loc21_19.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = param a, runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %int.make_type_32.loc21_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc21_27.1: type = value_of_initializer %int.make_type_32.loc21_27 [template = i32]
// CHECK:STDOUT:     %.loc21_27.2: type = converted %int.make_type_32.loc21_27, %.loc21_27.1 [template = i32]
// CHECK:STDOUT:     %b.param: i32 = param b, runtime_param1
// CHECK:STDOUT:     %b: i32 = bind_name b, %b.param
// CHECK:STDOUT:     %int.make_type_32.loc21_35: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc21_35.1: type = value_of_initializer %int.make_type_32.loc21_35 [template = i32]
// CHECK:STDOUT:     %.loc21_35.2: type = converted %int.make_type_32.loc21_35, %.loc21_35.1 [template = i32]
// CHECK:STDOUT:     %return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21: %.4 = assoc_entity element1, %G.decl [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc16
// CHECK:STDOUT:   .G = %.loc21
// CHECK:STDOUT:   witness = (%F.decl, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a: i32, %b: i32) -> i32 = "int.sadd";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self) {}
// CHECK:STDOUT:
