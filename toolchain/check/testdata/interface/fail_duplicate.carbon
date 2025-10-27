// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_duplicate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_duplicate.carbon

// --- fail_redefine_without_dependents.carbon

library "[[@TEST_NAME]]";

interface Interface { }

// CHECK:STDERR: fail_redefine_without_dependents.carbon:[[@LINE+7]]:1: error: redefinition of `interface Interface` [RedeclRedef]
// CHECK:STDERR: interface Interface {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefine_without_dependents.carbon:[[@LINE-5]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: interface Interface { }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface Interface {
  fn F();
}

// --- fail_redefine_with_dependents.carbon

library "[[@TEST_NAME]]";

interface Interface {}

// CHECK:STDERR: fail_redefine_with_dependents.carbon:[[@LINE+7]]:1: error: redefinition of `interface Interface` [RedeclRedef]
// CHECK:STDERR: interface Interface {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefine_with_dependents.carbon:[[@LINE-5]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: interface Interface {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface Interface {
  fn F[self: Self]();
}

// --- fail_name_conflict_with_fn.carbon

library "[[@TEST_NAME]]";

fn Function();

// CHECK:STDERR: fail_name_conflict_with_fn.carbon:[[@LINE+7]]:11: error: duplicate name `Function` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: interface Function;
// CHECK:STDERR:           ^~~~~~~~
// CHECK:STDERR: fail_name_conflict_with_fn.carbon:[[@LINE-5]]:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn Function();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
interface Function;

// --- fail_name_conflict_with_class.carbon

class Class;

// CHECK:STDERR: fail_name_conflict_with_class.carbon:[[@LINE+7]]:11: error: duplicate name `Class` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: interface Class { }
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR: fail_name_conflict_with_class.carbon:[[@LINE-5]]:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
interface Class { }

// CHECK:STDOUT: --- fail_redefine_without_dependents.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type.ba0a47.1: type = facet_type <@Interface.loc4> [concrete]
// CHECK:STDOUT:   %Self.46e08f.1: %Interface.type.ba0a47.1 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Interface.type.ba0a47.2: type = facet_type <@Interface.loc13> [concrete]
// CHECK:STDOUT:   %Self.46e08f.2: %Interface.type.ba0a47.2 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Interface.F.type: type = fn_type @Interface.F [concrete]
// CHECK:STDOUT:   %Interface.F: %Interface.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface.loc13 [concrete]
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, @Interface.loc13.%Interface.F.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Interface = %Interface.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl.loc4: type = interface_decl @Interface.loc4 [concrete = constants.%Interface.type.ba0a47.1] {} {}
// CHECK:STDOUT:   %Interface.decl.loc13: type = interface_decl @Interface.loc13 [concrete = constants.%Interface.type.ba0a47.2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface.loc4 {
// CHECK:STDOUT:   %Self: %Interface.type.ba0a47.1 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.46e08f.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface.loc13 {
// CHECK:STDOUT:   %Self: %Interface.type.ba0a47.2 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.46e08f.2]
// CHECK:STDOUT:   %Interface.F.decl: %Interface.F.type = fn_decl @Interface.F [concrete = constants.%Interface.F] {} {}
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, %Interface.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%Interface.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.F(@Interface.loc13.%Self: %Interface.type.ba0a47.2) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.F(constants.%Self.46e08f.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redefine_with_dependents.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type.ba0a47.1: type = facet_type <@Interface.loc4> [concrete]
// CHECK:STDOUT:   %Self.46e08f.1: %Interface.type.ba0a47.1 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Interface.type.ba0a47.2: type = facet_type <@Interface.loc13> [concrete]
// CHECK:STDOUT:   %Self.46e08f.2: %Interface.type.ba0a47.2 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.46e08f.2 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %Interface.F.type: type = fn_type @Interface.F [concrete]
// CHECK:STDOUT:   %Interface.F: %Interface.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface.loc13 [concrete]
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, @Interface.loc13.%Interface.F.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Interface = %Interface.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl.loc4: type = interface_decl @Interface.loc4 [concrete = constants.%Interface.type.ba0a47.1] {} {}
// CHECK:STDOUT:   %Interface.decl.loc13: type = interface_decl @Interface.loc13 [concrete = constants.%Interface.type.ba0a47.2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface.loc4 {
// CHECK:STDOUT:   %Self: %Interface.type.ba0a47.1 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.46e08f.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface.loc13 {
// CHECK:STDOUT:   %Self: %Interface.type.ba0a47.2 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.46e08f.2]
// CHECK:STDOUT:   %Interface.F.decl: %Interface.F.type = fn_decl @Interface.F [concrete = constants.%Interface.F] {
// CHECK:STDOUT:     %self.patt: @Interface.F.%pattern_type (%pattern_type) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Interface.F.%pattern_type (%pattern_type) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Interface.F.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc14_14.1: type = splice_block %.loc14_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Interface.type.ba0a47.2 = name_ref Self, @Interface.loc13.%Self [symbolic = %Self (constants.%Self.46e08f.2)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc14_14.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Interface.F.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, %Interface.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%Interface.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.F(@Interface.loc13.%Self: %Interface.type.ba0a47.2) {
// CHECK:STDOUT:   %Self: %Interface.type.ba0a47.2 = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.46e08f.2)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Interface.F.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.F(constants.%Self.46e08f.2) {
// CHECK:STDOUT:   %Self => constants.%Self.46e08f.2
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_conflict_with_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Function.type.b14: type = fn_type @Function.loc4 [concrete]
// CHECK:STDOUT:   %Function: %Function.type.b14 = struct_value () [concrete]
// CHECK:STDOUT:   %Function.type.b2d: type = facet_type <@Function.loc13> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Function = %Function.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Function.decl.loc4: %Function.type.b14 = fn_decl @Function.loc4 [concrete = constants.%Function] {} {}
// CHECK:STDOUT:   %Function.decl.loc13: type = interface_decl @Function.loc13 [concrete = constants.%Function.type.b2d] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Function.loc13;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Function.loc4();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_conflict_with_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class.loc2 [concrete]
// CHECK:STDOUT:   %Class.type: type = facet_type <@Class.loc11> [concrete]
// CHECK:STDOUT:   %Self: %Class.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Class = %Class.decl.loc2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.decl.loc2: type = class_decl @Class.loc2 [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %Class.decl.loc11: type = interface_decl @Class.loc11 [concrete = constants.%Class.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Class.loc11 {
// CHECK:STDOUT:   %Self: %Class.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class.loc2;
// CHECK:STDOUT:
