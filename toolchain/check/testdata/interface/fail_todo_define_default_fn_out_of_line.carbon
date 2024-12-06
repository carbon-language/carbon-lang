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
  // CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE+4]]:3: error: semantics TODO: `interface modifier` [SemanticsTodo]
  // CHECK:STDERR:   default fn F();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn F();

  // CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE+4]]:3: error: semantics TODO: `interface modifier` [SemanticsTodo]
  // CHECK:STDERR:   default fn G(a: i32, b: i32) -> i32;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn G(a: i32, b: i32) -> i32;
}

// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE+7]]:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn Interface.F() {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE-12]]:3: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:   default fn F();
// CHECK:STDERR:   ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Interface.F() {}

// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE+6]]:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn Interface.G(a: i32, b: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE-15]]:3: note: name is previously declared here [NameDeclPrevious]
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
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [template]
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %Interface.type, %F.type [template]
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, @Interface.%F.decl [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %G.assoc_type: type = assoc_entity_type %Interface.type, %G.type [template]
// CHECK:STDOUT:   %assoc1: %G.assoc_type = assoc_entity element1, @Interface.%G.decl [template]
// CHECK:STDOUT:   %.type.1: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.1: %.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.type.2: type = fn_type @.2 [template]
// CHECK:STDOUT:   %.2: %.type.2 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %.decl.loc23: %.type.1 = fn_decl @.1 [template = constants.%.1] {} {}
// CHECK:STDOUT:   %.decl.loc31: %.type.2 = fn_decl @.2 [template = constants.%.2] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc31_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc31_19: init type = call constants.%Int(%int_32.loc31_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc31_19.1: type = value_of_initializer %int.make_type_signed.loc31_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc31_19.2: type = converted %int.make_type_signed.loc31_19, %.loc31_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc31_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc31_27: init type = call constants.%Int(%int_32.loc31_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc31_27.1: type = value_of_initializer %int.make_type_signed.loc31_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc31_27.2: type = converted %int.make_type_signed.loc31_27, %.loc31_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc31_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc31_35: init type = call constants.%Int(%int_32.loc31_35) [template = constants.%i32]
// CHECK:STDOUT:     %.loc31_35.1: type = value_of_initializer %int.make_type_signed.loc31_35 [template = constants.%i32]
// CHECK:STDOUT:     %.loc31_35.2: type = converted %int.make_type_signed.loc31_35, %.loc31_35.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, %F.decl [template = constants.%assoc0]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc13_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc13_19: init type = call constants.%Int(%int_32.loc13_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_19.1: type = value_of_initializer %int.make_type_signed.loc13_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_19.2: type = converted %int.make_type_signed.loc13_19, %.loc13_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc13_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc13_27: init type = call constants.%Int(%int_32.loc13_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_27.1: type = value_of_initializer %int.make_type_signed.loc13_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_27.2: type = converted %int.make_type_signed.loc13_27, %.loc13_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc13_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc13_35: init type = call constants.%Int(%int_32.loc13_35) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_35.1: type = value_of_initializer %int.make_type_signed.loc13_35 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_35.2: type = converted %int.make_type_signed.loc13_35, %.loc13_35.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %G.assoc_type = assoc_entity element1, %G.decl [template = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%F.decl, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @.1(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @.2(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.sadd";
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
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [template]
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %C.1: type = class_type @C [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%Self) [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @C(%Self) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [symbolic = constants.%F] {
// CHECK:STDOUT:     %self.patt: %C.2 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.2 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %U.patt.loc20: type = symbolic_binding_pattern U, 1 [symbolic = constants.%U.patt]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc20, runtime_param<invalid> [symbolic = constants.%U.patt]
// CHECK:STDOUT:     %u.patt: %U = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: %U = value_param_pattern %u.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %U = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %U = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc20: type = specific_constant constants.%C.2, @C(constants.%Self) [symbolic = constants.%C.2]
// CHECK:STDOUT:     %Self.ref.loc20: type = name_ref Self, %.loc20 [symbolic = constants.%C.2]
// CHECK:STDOUT:     %U.ref.loc20_43: type = name_ref U, %U.loc20 [symbolic = constants.%U]
// CHECK:STDOUT:     %U.ref.loc20_49: type = name_ref U, %U.loc20 [symbolic = constants.%U]
// CHECK:STDOUT:     %self.param.loc20: %C.2 = value_param runtime_param0
// CHECK:STDOUT:     %self.loc20: %C.2 = bind_name self, %self.param.loc20
// CHECK:STDOUT:     %U.param.loc20: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc20: type = bind_symbolic_name U, 1, %U.param.loc20 [symbolic = constants.%U]
// CHECK:STDOUT:     %u.param.loc20: %U = value_param runtime_param1
// CHECK:STDOUT:     %u.loc20: %U = bind_name u, %u.param.loc20
// CHECK:STDOUT:     %return.param.loc20: ref %U = out_param runtime_param2
// CHECK:STDOUT:     %return.loc20: ref %U = return_slot %return.param.loc20
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C.1] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .C = %C.decl
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @C(%Self) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @C.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @C.%F.type (%F.type) = fn_decl @F [symbolic = @C.%F (constants.%F)] {
// CHECK:STDOUT:       %self.patt: %C.2 = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: %C.2 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %U.patt.loc20: type = symbolic_binding_pattern U, 1 [symbolic = constants.%U.patt]
// CHECK:STDOUT:       %U.param_patt: type = value_param_pattern %U.patt.loc20, runtime_param<invalid> [symbolic = constants.%U.patt]
// CHECK:STDOUT:       %u.patt: %U = binding_pattern u
// CHECK:STDOUT:       %u.param_patt: %U = value_param_pattern %u.patt, runtime_param1
// CHECK:STDOUT:       %return.patt: %U = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: %U = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc14: type = specific_constant constants.%C.2, @C(constants.%Self) [symbolic = %C (constants.%C.2)]
// CHECK:STDOUT:       %Self.ref.loc14: type = name_ref Self, %.loc14 [symbolic = %C (constants.%C.2)]
// CHECK:STDOUT:       %U.ref.loc14_35: type = name_ref U, %U.loc14_22.2 [symbolic = %U.loc14_22.1 (constants.%U)]
// CHECK:STDOUT:       %U.ref.loc14_41: type = name_ref U, %U.loc14_22.2 [symbolic = %U.loc14_22.1 (constants.%U)]
// CHECK:STDOUT:       %self.param.loc14: @F.%C (%C.2) = value_param runtime_param0
// CHECK:STDOUT:       %self.loc14: @F.%C (%C.2) = bind_name self, %self.param.loc14
// CHECK:STDOUT:       %U.param.loc14: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %U.loc14_22.2: type = bind_symbolic_name U, 1, %U.param.loc14 [symbolic = %U.loc14_22.1 (constants.%U)]
// CHECK:STDOUT:       %u.param.loc14: @F.%U.loc14_22.1 (%U) = value_param runtime_param1
// CHECK:STDOUT:       %u.loc14: @F.%U.loc14_22.1 (%U) = bind_name u, %u.param.loc14
// CHECK:STDOUT:       %return.param.loc14: ref @F.%U.loc14_22.1 (%U) = out_param runtime_param2
// CHECK:STDOUT:       %return.loc14: ref @F.%U.loc14_22.1 (%U) = return_slot %return.param.loc14
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Interface.%Self: %Interface.type, %U.loc14_22.2: type) {
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Self) [symbolic = %C (constants.%C.2)]
// CHECK:STDOUT:   %U.loc14_22.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc14_22.1 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc14: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc14 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: %C.2](%U.param_patt: type, %u.param_patt: %U) -> %U {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %u.ref: @F.%U.loc14_22.1 (%U) = name_ref u, %u.loc20
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
// CHECK:STDOUT:   %U.loc14_22.1 => constants.%U
// CHECK:STDOUT:   %U.patt.loc14 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(%Self) {}
// CHECK:STDOUT:
