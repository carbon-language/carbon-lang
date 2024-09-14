// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_todo_define_default_fn_out_of_line.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_todo_define_default_fn_out_of_line.carbon

// --- fail_todo_define_default_fn_out_of_line.carbon

interface Interface {
  // CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `interface modifier`.
  // CHECK:STDERR:   default fn F();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn F();

  // CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `interface modifier`.
  // CHECK:STDERR:   default fn G(a: i32, b: i32) -> i32;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn G(a: i32, b: i32) -> i32;
}

// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE+7]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn Interface.F() {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE-12]]:3: Name is previously declared here.
// CHECK:STDERR:   default fn F();
// CHECK:STDERR:   ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Interface.F() {}

// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE+6]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn Interface.G(a: i32, b: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE-15]]:3: Name is previously declared here.
// CHECK:STDERR:   default fn G(a: i32, b: i32) -> i32;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn Interface.G(a: i32, b: i32) -> i32 = "int.sadd";

// --- dependent_return_type.carbon

library "dependent_return_type.carbon";

// Ensure that a dependent return type matches between an interface definition
// and an out-of-line member. This requires `Self` to be properly reintroduced
// into the list of generic parameters.

interface Interface {
  // TODO: This should be
  //   default fn F[self: Self](U:! type, u: U) -> U;
  // rather than a class member, but we don't currently accept that due to the
  // TODOs above.
  class C {
    fn F[self: Self](U:! type, u: U) -> U;
  }
}

// TODO: This should be
//   fn Interface.F[self: Self](U:! type, u: U) -> U { return u; }
fn Interface.C.F[self: Self](U:! type, u: U) -> U { return u; }

// CHECK:STDOUT: --- fail_todo_define_default_fn_out_of_line.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Interface [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.1, %F.type [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @Interface.%F.decl [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.1, %G.type [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element1, @Interface.%G.decl [template]
// CHECK:STDOUT:   %.type.1: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.7: %.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.type.2: type = fn_type @.2 [template]
// CHECK:STDOUT:   %.8: %.type.2 = struct_value () [template]
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
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%.1] {}
// CHECK:STDOUT:   %.decl.loc23: %.type.1 = fn_decl @.1 [template = constants.%.7] {}
// CHECK:STDOUT:   %.decl.loc31: %.type.2 = fn_decl @.2 [template = constants.%.8] {
// CHECK:STDOUT:     %int.make_type_32.loc31_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc31_19.1: type = value_of_initializer %int.make_type_32.loc31_19 [template = i32]
// CHECK:STDOUT:     %.loc31_19.2: type = converted %int.make_type_32.loc31_19, %.loc31_19.1 [template = i32]
// CHECK:STDOUT:     %a.loc31_16.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     @.2.%a: i32 = bind_name a, %a.loc31_16.1
// CHECK:STDOUT:     %int.make_type_32.loc31_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc31_27.1: type = value_of_initializer %int.make_type_32.loc31_27 [template = i32]
// CHECK:STDOUT:     %.loc31_27.2: type = converted %int.make_type_32.loc31_27, %.loc31_27.1 [template = i32]
// CHECK:STDOUT:     %b.loc31_24.1: i32 = param b, runtime_param1
// CHECK:STDOUT:     @.2.%b: i32 = bind_name b, %b.loc31_24.1
// CHECK:STDOUT:     %int.make_type_32.loc31_35: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc31_35.1: type = value_of_initializer %int.make_type_32.loc31_35 [template = i32]
// CHECK:STDOUT:     %.loc31_35.2: type = converted %int.make_type_32.loc31_35, %.loc31_35.1 [template = i32]
// CHECK:STDOUT:     @.2.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT:   %.loc7: %.3 = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %int.make_type_32.loc13_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc13_19.1: type = value_of_initializer %int.make_type_32.loc13_19 [template = i32]
// CHECK:STDOUT:     %.loc13_19.2: type = converted %int.make_type_32.loc13_19, %.loc13_19.1 [template = i32]
// CHECK:STDOUT:     %a.loc13_16.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     %a.loc13_16.2: i32 = bind_name a, %a.loc13_16.1
// CHECK:STDOUT:     %int.make_type_32.loc13_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc13_27.1: type = value_of_initializer %int.make_type_32.loc13_27 [template = i32]
// CHECK:STDOUT:     %.loc13_27.2: type = converted %int.make_type_32.loc13_27, %.loc13_27.1 [template = i32]
// CHECK:STDOUT:     %b.loc13_24.1: i32 = param b, runtime_param1
// CHECK:STDOUT:     %b.loc13_24.2: i32 = bind_name b, %b.loc13_24.1
// CHECK:STDOUT:     %int.make_type_32.loc13_35: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc13_35.1: type = value_of_initializer %int.make_type_32.loc13_35 [template = i32]
// CHECK:STDOUT:     %.loc13_35.2: type = converted %int.make_type_32.loc13_35, %.loc13_35.1 [template = i32]
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_38: %.5 = assoc_entity element1, %G.decl [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc7
// CHECK:STDOUT:   .G = %.loc13_38
// CHECK:STDOUT:   witness = (%F.decl, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Interface.%Self: %.1) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@Interface.%Self: %.1) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(@Interface.%a.loc13_16.2: i32, @Interface.%b.loc13_24.2: i32) -> i32;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @.1(@Interface.%Self: %.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @.2(@Interface.%Self: %.1) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a: i32, %b: i32) -> i32 = "int.sadd";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.2(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- dependent_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Interface [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %C.1: type = class_type @C [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%Self) [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @C(%Self) [symbolic]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%.1] {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [symbolic = constants.%F] {
// CHECK:STDOUT:     %.loc20: type = specific_constant constants.%C.2, @C(constants.%Self) [symbolic = constants.%C.2]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, %.loc20 [symbolic = constants.%C.2]
// CHECK:STDOUT:     %self.loc20_18.1: %C.2 = param self, runtime_param0
// CHECK:STDOUT:     @F.%self: %C.2 = bind_name self, %self.loc20_18.1
// CHECK:STDOUT:     %U.loc20_30.1: type = param U, runtime_param<invalid>
// CHECK:STDOUT:     @F.%U.loc20: type = bind_symbolic_name U 1, %U.loc20_30.1 [symbolic = constants.%U]
// CHECK:STDOUT:     %U.ref.loc20_43: type = name_ref U, @F.%U.loc20 [symbolic = constants.%U]
// CHECK:STDOUT:     %u.loc20_40.1: %U = param u, runtime_param1
// CHECK:STDOUT:     @F.%u: %U = bind_name u, %u.loc20_40.1
// CHECK:STDOUT:     %U.ref.loc20_49: type = name_ref U, @F.%U.loc20 [symbolic = constants.%U]
// CHECK:STDOUT:     @F.%return: ref %U = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C.1] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .C = %C.decl
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(@Interface.%Self: %.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @C(%Self) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @C.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @C.%F.type (%F.type) = fn_decl @F [symbolic = %F (constants.%F)] {
// CHECK:STDOUT:       %.loc14: type = specific_constant constants.%C.2, @C(constants.%Self) [symbolic = @F.%C (constants.%C.2)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc14 [symbolic = @F.%C (constants.%C.2)]
// CHECK:STDOUT:       %self.loc14_10.1: @F.%C (%C.2) = param self, runtime_param0
// CHECK:STDOUT:       %self.loc14_10.2: @F.%C (%C.2) = bind_name self, %self.loc14_10.1
// CHECK:STDOUT:       %U.loc14_22.1: type = param U, runtime_param<invalid>
// CHECK:STDOUT:       %U.loc14_22.2: type = bind_symbolic_name U 1, %U.loc14_22.1 [symbolic = @F.%U.1 (constants.%U)]
// CHECK:STDOUT:       %U.ref.loc14_35: type = name_ref U, %U.loc14_22.2 [symbolic = @F.%U.1 (constants.%U)]
// CHECK:STDOUT:       %u.loc14_32.1: @F.%U.1 (%U) = param u, runtime_param1
// CHECK:STDOUT:       %u.loc14_32.2: @F.%U.1 (%U) = bind_name u, %u.loc14_32.1
// CHECK:STDOUT:       %U.ref.loc14_41: type = name_ref U, %U.loc14_22.2 [symbolic = @F.%U.1 (constants.%U)]
// CHECK:STDOUT:       %return.var: ref @F.%U.1 (%U) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Interface.%Self: %.1, @C.%U.loc14_22.2: type) {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Self) [symbolic = %C (constants.%C.2)]
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U 1 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: %C.2](%U.loc20: type, %u: %U) -> %U {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %u.ref: @F.%U.1 (%U) = name_ref u, %u
// CHECK:STDOUT:     return %u.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@F.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self, constants.%U) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %C => constants.%C.2
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@C.%Self) {}
// CHECK:STDOUT:
