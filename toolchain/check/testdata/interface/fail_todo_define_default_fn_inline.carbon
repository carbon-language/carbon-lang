// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_todo_define_default_fn_inline.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_todo_define_default_fn_inline.carbon

interface Interface {
  // CHECK:STDERR: fail_todo_define_default_fn_inline.carbon:[[@LINE+4]]:3: error: semantics TODO: `interface modifier` [SemanticsTodo]
  // CHECK:STDERR:   default fn F() {}
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn F() {}

  // CHECK:STDERR: fail_todo_define_default_fn_inline.carbon:[[@LINE+4]]:3: error: semantics TODO: `interface modifier` [SemanticsTodo]
  // CHECK:STDERR:   default fn G(a: i32, b: i32) -> i32 = "int.sadd";
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn G(a: i32, b: i32) -> i32 = "int.sadd";
}

// CHECK:STDOUT: --- fail_todo_define_default_fn_inline.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [concrete]
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Interface.WithSelf.F.type: type = fn_type @Interface.WithSelf.F, @Interface.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %Interface.WithSelf.F: %Interface.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface [concrete]
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, @Interface.WithSelf.%Interface.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.6b6 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %b.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.6b6 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   %Interface.WithSelf.G.type: type = fn_type @Interface.WithSelf.G, @Interface.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %Interface.WithSelf.G: %Interface.WithSelf.G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %Interface.assoc_type = assoc_entity element1, @Interface.WithSelf.%Interface.WithSelf.G.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [concrete = constants.%Interface.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Interface.WithSelf.decl = interface_with_self_decl @Interface [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Interface.WithSelf.F.decl: @Interface.WithSelf.%Interface.WithSelf.F.type (%Interface.WithSelf.F.type) = fn_decl @Interface.WithSelf.F [symbolic = @Interface.WithSelf.%Interface.WithSelf.F (constants.%Interface.WithSelf.F)] {} {}
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, %Interface.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Interface.WithSelf.G.decl: @Interface.WithSelf.%Interface.WithSelf.G.type (%Interface.WithSelf.G.type) = fn_decl @Interface.WithSelf.G [symbolic = @Interface.WithSelf.%Interface.WithSelf.G (constants.%Interface.WithSelf.G)] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.6b6 = at_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%b.param_patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.6b6 = at_binding_pattern b, %b.param_patt [concrete = constants.%b.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc26_35 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc26_35: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc26: Core.Form = init_form %i32.loc26_35 [concrete = constants.%.795]
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc26_19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %a: %i32 = value_binding a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc26_27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %b: %i32 = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %Interface.assoc_type = assoc_entity element1, %Interface.WithSelf.G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @Interface.WithSelf.%assoc0
// CHECK:STDOUT:   .G = @Interface.WithSelf.%assoc1
// CHECK:STDOUT:   witness = (@Interface.WithSelf.%Interface.WithSelf.F.decl, @Interface.WithSelf.%Interface.WithSelf.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.WithSelf.F(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.WithSelf.G(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: %i32, %b.param: %i32) -> out %return.param: %i32 = "int.sadd";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Interface.WithSelf.F.type => constants.%Interface.WithSelf.F.type
// CHECK:STDOUT:   %Interface.WithSelf.F => constants.%Interface.WithSelf.F
// CHECK:STDOUT:   %Interface.WithSelf.G.type => constants.%Interface.WithSelf.G.type
// CHECK:STDOUT:   %Interface.WithSelf.G => constants.%Interface.WithSelf.G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf.F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf.G(constants.%Self) {}
// CHECK:STDOUT:
