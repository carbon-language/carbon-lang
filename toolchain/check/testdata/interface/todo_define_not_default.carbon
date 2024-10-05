// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/todo_define_not_default.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/todo_define_not_default.carbon

interface I {
  // TODO: A definition without `default` in an interface should be rejected.
  fn F() {}
  fn G(a: i32, b: i32) -> i32 = "int.sadd";

  // TODO: An associated constant with an initializer without `default` in an
  // interface should be rejected.
  let T:! type = (i32, i32);
  let N:! i32 = 42;
}

// CHECK:STDOUT: --- todo_define_not_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = interface_type @I [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %I.type, %F.type [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %I.type, %G.type [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element1, @I.%G.decl [template]
// CHECK:STDOUT:   %.6: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.7: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element2, @I.%T [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 42 [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %I.type, i32 [template]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element3, @I.%N [template]
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
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %.loc13: %.2 = assoc_entity element0, %F.decl [template = constants.%.3]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %b.patt: i32 = binding_pattern b
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc14_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc14_11.1: type = value_of_initializer %int.make_type_32.loc14_11 [template = i32]
// CHECK:STDOUT:     %.loc14_11.2: type = converted %int.make_type_32.loc14_11, %.loc14_11.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = param a, runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %int.make_type_32.loc14_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc14_19.1: type = value_of_initializer %int.make_type_32.loc14_19 [template = i32]
// CHECK:STDOUT:     %.loc14_19.2: type = converted %int.make_type_32.loc14_19, %.loc14_19.1 [template = i32]
// CHECK:STDOUT:     %b.param: i32 = param b, runtime_param1
// CHECK:STDOUT:     %b: i32 = bind_name b, %b.param
// CHECK:STDOUT:     %int.make_type_32.loc14_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc14_27.1: type = value_of_initializer %int.make_type_32.loc14_27 [template = i32]
// CHECK:STDOUT:     %.loc14_27.2: type = converted %int.make_type_32.loc14_27, %.loc14_27.1 [template = i32]
// CHECK:STDOUT:     %return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: %.4 = assoc_entity element1, %G.decl [template = constants.%.5]
// CHECK:STDOUT:   %int.make_type_32.loc18_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc18_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc18_27: %.6 = tuple_literal (%int.make_type_32.loc18_19, %int.make_type_32.loc18_24)
// CHECK:STDOUT:   %.loc18_28.1: type = value_of_initializer %int.make_type_32.loc18_19 [template = i32]
// CHECK:STDOUT:   %.loc18_28.2: type = converted %int.make_type_32.loc18_19, %.loc18_28.1 [template = i32]
// CHECK:STDOUT:   %.loc18_28.3: type = value_of_initializer %int.make_type_32.loc18_24 [template = i32]
// CHECK:STDOUT:   %.loc18_28.4: type = converted %int.make_type_32.loc18_24, %.loc18_28.3 [template = i32]
// CHECK:STDOUT:   %.loc18_28.5: type = converted %.loc18_27, constants.%.7 [template = constants.%.7]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc18_28.6: %.8 = assoc_entity element2, %T [template = constants.%.9]
// CHECK:STDOUT:   %int.make_type_32.loc19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc19_11.1: type = value_of_initializer %int.make_type_32.loc19 [template = i32]
// CHECK:STDOUT:   %.loc19_11.2: type = converted %int.make_type_32.loc19, %.loc19_11.1 [template = i32]
// CHECK:STDOUT:   %.loc19_17: i32 = int_literal 42 [template = constants.%.10]
// CHECK:STDOUT:   %N: i32 = assoc_const_decl N [template]
// CHECK:STDOUT:   %.loc19_19: %.11 = assoc_entity element3, %N [template = constants.%.12]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc13
// CHECK:STDOUT:   .G = %.loc14
// CHECK:STDOUT:   .T = %.loc18_28.6
// CHECK:STDOUT:   .N = %.loc19_19
// CHECK:STDOUT:   witness = (%F.decl, %G.decl, %T, %N)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%Self: %I.type) {
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
// CHECK:STDOUT: generic fn @G(@I.%Self: %I.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a: i32, %b: i32) -> i32 = "int.sadd";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self) {}
// CHECK:STDOUT:
