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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/define_default_fn_out_of_line.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/define_default_fn_out_of_line.carbon

// --- fail_todo_define_default_fn_out_of_line.carbon
library "[[@TEST_NAME]]";

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

// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE+7]]:14: error: duplicate name `F` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn Interface.F() {}
// CHECK:STDERR:              ^
// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE-12]]:3: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:   default fn F();
// CHECK:STDERR:   ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Interface.F() {}

// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE+7]]:14: error: duplicate name `G` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn Interface.G(a: i32, b: i32) -> i32 = "int.sadd";
// CHECK:STDERR:              ^
// CHECK:STDERR: fail_todo_define_default_fn_out_of_line.carbon:[[@LINE-15]]:3: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:   default fn G(a: i32, b: i32) -> i32;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Interface.G(a: i32, b: i32) -> i32 = "int.sadd";

// --- fail_todo_dependent_return_type.carbon
library "[[@TEST_NAME]]";

// Ensure that a dependent return type matches between an interface definition
// and an out-of-line member. This requires `Self` to be properly reintroduced
// into the list of generic parameters.

interface Interface {
  // CHECK:STDERR: fail_todo_dependent_return_type.carbon:[[@LINE+4]]:3: error: semantics TODO: `interface modifier` [SemanticsTodo]
  // CHECK:STDERR:   default fn F[self: Self](U:! type, u: U*) -> U*;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn F[self: Self](U:! type, u: U*) -> U*;
}

// TODO: This should be
//   fn Interface.F[self: Self](U:! type, u: U*) -> U* { return u; }
// CHECK:STDERR: fail_todo_dependent_return_type.carbon:[[@LINE+7]]:14: error: duplicate name `F` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn Interface.F[self: Self](U:! type, u: U*) -> U* { return u; }
// CHECK:STDERR:              ^
// CHECK:STDERR: fail_todo_dependent_return_type.carbon:[[@LINE-8]]:3: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:   default fn F[self: Self](U:! type, u: U*) -> U*;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Interface.F[self: Self](U:! type, u: U*) -> U* { return u; }

// CHECK:STDOUT: --- fail_todo_define_default_fn_out_of_line.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [concrete]
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Interface.F.type.3f3064.1: type = fn_type @Interface.F.loc8 [concrete]
// CHECK:STDOUT:   %Interface.F.77c12a.1: %Interface.F.type.3f3064.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface [concrete]
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, @Interface.%Interface.F.decl [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Interface.G.type.cde93b.1: type = fn_type @Interface.G.loc14 [concrete]
// CHECK:STDOUT:   %Interface.G.433e0c.1: %Interface.G.type.cde93b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %Interface.assoc_type = assoc_entity element1, @Interface.%Interface.G.decl [concrete]
// CHECK:STDOUT:   %Interface.F.type.3f3064.2: type = fn_type @Interface.F.loc24 [concrete]
// CHECK:STDOUT:   %Interface.F.77c12a.2: %Interface.F.type.3f3064.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.G.type.cde93b.2: type = fn_type @Interface.G.loc33 [concrete]
// CHECK:STDOUT:   %Interface.G.433e0c.2: %Interface.G.type.cde93b.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Interface.F.decl: %Interface.F.type.3f3064.2 = fn_decl @Interface.F.loc24 [concrete = constants.%Interface.F.77c12a.2] {} {}
// CHECK:STDOUT:   %Interface.G.decl: %Interface.G.type.cde93b.2 = fn_decl @Interface.G.loc33 [concrete = constants.%Interface.G.433e0c.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = value_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.7ce = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc33_35: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc33_35: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc33_19: type = splice_block %i32.loc33_19 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc33_19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc33_19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = value_binding a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc33_27: type = splice_block %i32.loc33_27 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc33_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc33_27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Interface.F.decl: %Interface.F.type.3f3064.1 = fn_decl @Interface.F.loc8 [concrete = constants.%Interface.F.77c12a.1] {} {}
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, %Interface.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Interface.G.decl: %Interface.G.type.cde93b.1 = fn_decl @Interface.G.loc14 [concrete = constants.%Interface.G.433e0c.1] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = value_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.7ce = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc14_35: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_35: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc14_19: type = splice_block %i32.loc14_19 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc14_19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc14_19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = value_binding a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc14_27: type = splice_block %i32.loc14_27 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc14_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc14_27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %Interface.assoc_type = assoc_entity element1, %Interface.G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%Interface.F.decl, %Interface.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.F.loc8(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.G.loc14(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT:   fn(%a.param: %i32, %b.param: %i32) -> %i32;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.F.loc24(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.G.loc33(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: %i32, %b.param: %i32) -> %i32 = "int.sadd";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.F.loc8(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.G.loc14(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.F.loc24(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.G.loc33(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_dependent_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [concrete]
// CHECK:STDOUT:   %Self.f87: %Interface.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.d18: type = symbolic_binding_type Self, 0, %Self.f87 [symbolic]
// CHECK:STDOUT:   %pattern_type.b78: type = pattern_type %Self.binding.as_type.d18 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %U.091: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ptr.18e: type = ptr_type %U.091 [symbolic]
// CHECK:STDOUT:   %pattern_type.423: type = pattern_type %ptr.18e [symbolic]
// CHECK:STDOUT:   %Interface.F.type.3f3064.1: type = fn_type @Interface.F.loc12 [concrete]
// CHECK:STDOUT:   %Interface.F.77c12a.1: %Interface.F.type.3f3064.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface [concrete]
// CHECK:STDOUT:   %assoc0.fcf: %Interface.assoc_type = assoc_entity element0, @Interface.%Interface.F.decl [concrete]
// CHECK:STDOUT:   %Interface.F.type.3f3064.2: type = fn_type @Interface.F.loc24 [concrete]
// CHECK:STDOUT:   %Interface.F.77c12a.2: %Interface.F.type.3f3064.2 = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.56d: <witness> = require_complete_type %ptr.18e [symbolic]
// CHECK:STDOUT:   %require_complete.9f2: <witness> = require_complete_type %Self.binding.as_type.d18 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %.2bd: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%U.091) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.484: <witness> = lookup_impl_witness %ptr.18e, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.18e, (%Copy.lookup_impl_witness.484) [symbolic]
// CHECK:STDOUT:   %.4ef: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.caa: %.4ef = impl_witness_access %Copy.lookup_impl_witness.484, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.8e3: <specific function> = specific_impl_function %impl.elem0.caa, @Copy.Op(%Copy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [concrete = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %Interface.F.decl: %Interface.F.type.3f3064.2 = fn_decl @Interface.F.loc24 [concrete = constants.%Interface.F.77c12a.2] {
// CHECK:STDOUT:     %self.patt: @Interface.F.loc24.%pattern_type.loc24_16 (%pattern_type.b78) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Interface.F.loc24.%pattern_type.loc24_16 (%pattern_type.b78) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %u.patt: @Interface.F.loc24.%pattern_type.loc24_38 (%pattern_type.423) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @Interface.F.loc24.%pattern_type.loc24_38 (%pattern_type.423) = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @Interface.F.loc24.%pattern_type.loc24_38 (%pattern_type.423) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Interface.F.loc24.%pattern_type.loc24_38 (%pattern_type.423) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc24_48: type = name_ref U, %U.loc24_28.2 [symbolic = %U.loc24_28.1 (constants.%U.091)]
// CHECK:STDOUT:     %ptr.loc24_49: type = ptr_type %U.ref.loc24_48 [symbolic = %ptr.loc24_42.1 (constants.%ptr.18e)]
// CHECK:STDOUT:     %self.param: @Interface.F.loc24.%Self.binding.as_type (%Self.binding.as_type.d18) = value_param call_param0
// CHECK:STDOUT:     %.loc24_22.1: type = splice_block %.loc24_22.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d18)] {
// CHECK:STDOUT:       %Self.ref: %Interface.type = name_ref Self, @Interface.%Self [symbolic = %Self (constants.%Self.f87)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d18)]
// CHECK:STDOUT:       %.loc24_22.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d18)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Interface.F.loc24.%Self.binding.as_type (%Self.binding.as_type.d18) = value_binding self, %self.param
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc24_28.2: type = symbolic_binding U, 1 [symbolic = %U.loc24_28.1 (constants.%U.091)]
// CHECK:STDOUT:     %u.param: @Interface.F.loc24.%ptr.loc24_42.1 (%ptr.18e) = value_param call_param1
// CHECK:STDOUT:     %.loc24_42: type = splice_block %ptr.loc24_42.2 [symbolic = %ptr.loc24_42.1 (constants.%ptr.18e)] {
// CHECK:STDOUT:       %U.ref.loc24_41: type = name_ref U, %U.loc24_28.2 [symbolic = %U.loc24_28.1 (constants.%U.091)]
// CHECK:STDOUT:       %ptr.loc24_42.2: type = ptr_type %U.ref.loc24_41 [symbolic = %ptr.loc24_42.1 (constants.%ptr.18e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @Interface.F.loc24.%ptr.loc24_42.1 (%ptr.18e) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @Interface.F.loc24.%ptr.loc24_42.1 (%ptr.18e) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Interface.F.loc24.%ptr.loc24_42.1 (%ptr.18e) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic = constants.%Self.f87]
// CHECK:STDOUT:   %Interface.F.decl: %Interface.F.type.3f3064.1 = fn_decl @Interface.F.loc12 [concrete = constants.%Interface.F.77c12a.1] {
// CHECK:STDOUT:     %self.patt: @Interface.F.loc12.%pattern_type.loc12_16 (%pattern_type.b78) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Interface.F.loc12.%pattern_type.loc12_16 (%pattern_type.b78) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %u.patt: @Interface.F.loc12.%pattern_type.loc12_38 (%pattern_type.423) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @Interface.F.loc12.%pattern_type.loc12_38 (%pattern_type.423) = value_param_pattern %u.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @Interface.F.loc12.%pattern_type.loc12_38 (%pattern_type.423) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Interface.F.loc12.%pattern_type.loc12_38 (%pattern_type.423) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc12_48: type = name_ref U, %U.loc12_28.2 [symbolic = %U.loc12_28.1 (constants.%U.091)]
// CHECK:STDOUT:     %ptr.loc12_49: type = ptr_type %U.ref.loc12_48 [symbolic = %ptr.loc12_42.1 (constants.%ptr.18e)]
// CHECK:STDOUT:     %self.param: @Interface.F.loc12.%Self.binding.as_type (%Self.binding.as_type.d18) = value_param call_param0
// CHECK:STDOUT:     %.loc12_22.1: type = splice_block %.loc12_22.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d18)] {
// CHECK:STDOUT:       %Self.ref: %Interface.type = name_ref Self, @Interface.%Self [symbolic = %Self (constants.%Self.f87)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d18)]
// CHECK:STDOUT:       %.loc12_22.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d18)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Interface.F.loc12.%Self.binding.as_type (%Self.binding.as_type.d18) = value_binding self, %self.param
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc12_28.2: type = symbolic_binding U, 1 [symbolic = %U.loc12_28.1 (constants.%U.091)]
// CHECK:STDOUT:     %u.param: @Interface.F.loc12.%ptr.loc12_42.1 (%ptr.18e) = value_param call_param1
// CHECK:STDOUT:     %.loc12_42: type = splice_block %ptr.loc12_42.2 [symbolic = %ptr.loc12_42.1 (constants.%ptr.18e)] {
// CHECK:STDOUT:       %U.ref.loc12_41: type = name_ref U, %U.loc12_28.2 [symbolic = %U.loc12_28.1 (constants.%U.091)]
// CHECK:STDOUT:       %ptr.loc12_42.2: type = ptr_type %U.ref.loc12_41 [symbolic = %ptr.loc12_42.1 (constants.%ptr.18e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @Interface.F.loc12.%ptr.loc12_42.1 (%ptr.18e) = value_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @Interface.F.loc12.%ptr.loc12_42.1 (%ptr.18e) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Interface.F.loc12.%ptr.loc12_42.1 (%ptr.18e) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, %Interface.F.decl [concrete = constants.%assoc0.fcf]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%Interface.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.F.loc12(@Interface.%Self: %Interface.type, %U.loc12_28.2: type) {
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.f87)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d18)]
// CHECK:STDOUT:   %pattern_type.loc12_16: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc12_16 (constants.%pattern_type.b78)]
// CHECK:STDOUT:   %U.loc12_28.1: type = symbolic_binding U, 1 [symbolic = %U.loc12_28.1 (constants.%U.091)]
// CHECK:STDOUT:   %ptr.loc12_42.1: type = ptr_type %U.loc12_28.1 [symbolic = %ptr.loc12_42.1 (constants.%ptr.18e)]
// CHECK:STDOUT:   %pattern_type.loc12_38: type = pattern_type %ptr.loc12_42.1 [symbolic = %pattern_type.loc12_38 (constants.%pattern_type.423)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Interface.F.loc12.%Self.binding.as_type (%Self.binding.as_type.d18), %u.param: @Interface.F.loc12.%ptr.loc12_42.1 (%ptr.18e)) -> @Interface.F.loc12.%ptr.loc12_42.1 (%ptr.18e);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.F.loc24(@Interface.%Self: %Interface.type, %U.loc24_28.2: type) {
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.f87)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d18)]
// CHECK:STDOUT:   %pattern_type.loc24_16: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc24_16 (constants.%pattern_type.b78)]
// CHECK:STDOUT:   %U.loc24_28.1: type = symbolic_binding U, 1 [symbolic = %U.loc24_28.1 (constants.%U.091)]
// CHECK:STDOUT:   %ptr.loc24_42.1: type = ptr_type %U.loc24_28.1 [symbolic = %ptr.loc24_42.1 (constants.%ptr.18e)]
// CHECK:STDOUT:   %pattern_type.loc24_38: type = pattern_type %ptr.loc24_42.1 [symbolic = %pattern_type.loc24_38 (constants.%pattern_type.423)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc24_49: <witness> = require_complete_type %ptr.loc24_42.1 [symbolic = %require_complete.loc24_49 (constants.%require_complete.56d)]
// CHECK:STDOUT:   %require_complete.loc24_20: <witness> = require_complete_type %Self.binding.as_type [symbolic = %require_complete.loc24_20 (constants.%require_complete.9f2)]
// CHECK:STDOUT:   %.loc24_60.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%U.loc24_28.1) [symbolic = %.loc24_60.1 (constants.%.2bd)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc24_42.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.484)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc24_42.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.loc24_60.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc24_60.2 (constants.%.4ef)]
// CHECK:STDOUT:   %impl.elem0.loc24_60.2: @Interface.F.loc24.%.loc24_60.2 (%.4ef) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc24_60.2 (constants.%impl.elem0.caa)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_60.2: <specific function> = specific_impl_function %impl.elem0.loc24_60.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc24_60.2 (constants.%specific_impl_fn.8e3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Interface.F.loc24.%Self.binding.as_type (%Self.binding.as_type.d18), %u.param: @Interface.F.loc24.%ptr.loc24_42.1 (%ptr.18e)) -> @Interface.F.loc24.%ptr.loc24_42.1 (%ptr.18e) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %u.ref: @Interface.F.loc24.%ptr.loc24_42.1 (%ptr.18e) = name_ref u, %u
// CHECK:STDOUT:     %impl.elem0.loc24_60.1: @Interface.F.loc24.%.loc24_60.2 (%.4ef) = impl_witness_access constants.%Copy.lookup_impl_witness.484, element0 [symbolic = %impl.elem0.loc24_60.2 (constants.%impl.elem0.caa)]
// CHECK:STDOUT:     %bound_method.loc24_60.1: <bound method> = bound_method %u.ref, %impl.elem0.loc24_60.1
// CHECK:STDOUT:     %specific_impl_fn.loc24_60.1: <specific function> = specific_impl_function %impl.elem0.loc24_60.1, @Copy.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc24_60.2 (constants.%specific_impl_fn.8e3)]
// CHECK:STDOUT:     %bound_method.loc24_60.2: <bound method> = bound_method %u.ref, %specific_impl_fn.loc24_60.1
// CHECK:STDOUT:     %Copy.Op.call: init @Interface.F.loc24.%ptr.loc24_42.1 (%ptr.18e) = call %bound_method.loc24_60.2(%u.ref)
// CHECK:STDOUT:     return %Copy.Op.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.F.loc12(constants.%Self.f87, constants.%U.091) {
// CHECK:STDOUT:   %Self => constants.%Self.f87
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.d18
// CHECK:STDOUT:   %pattern_type.loc12_16 => constants.%pattern_type.b78
// CHECK:STDOUT:   %U.loc12_28.1 => constants.%U.091
// CHECK:STDOUT:   %ptr.loc12_42.1 => constants.%ptr.18e
// CHECK:STDOUT:   %pattern_type.loc12_38 => constants.%pattern_type.423
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.F.loc24(constants.%Self.f87, constants.%U.091) {
// CHECK:STDOUT:   %Self => constants.%Self.f87
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.d18
// CHECK:STDOUT:   %pattern_type.loc24_16 => constants.%pattern_type.b78
// CHECK:STDOUT:   %U.loc24_28.1 => constants.%U.091
// CHECK:STDOUT:   %ptr.loc24_42.1 => constants.%ptr.18e
// CHECK:STDOUT:   %pattern_type.loc24_38 => constants.%pattern_type.423
// CHECK:STDOUT: }
// CHECK:STDOUT:
