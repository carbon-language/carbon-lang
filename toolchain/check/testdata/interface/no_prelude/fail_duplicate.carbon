// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/fail_duplicate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/fail_duplicate.carbon

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
// CHECK:STDOUT:   %Interface.type.d726c6.1: type = facet_type <@Interface.1> [concrete]
// CHECK:STDOUT:   %Self.719cbb.1: %Interface.type.d726c6.1 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Interface.type.d726c6.2: type = facet_type <@Interface.2> [concrete]
// CHECK:STDOUT:   %Self.719cbb.2: %Interface.type.d726c6.2 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type %Interface.type.d726c6.2 [concrete]
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, @Interface.2.%F.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Interface = %Interface.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl.loc4: type = interface_decl @Interface.1 [concrete = constants.%Interface.type.d726c6.1] {} {}
// CHECK:STDOUT:   %Interface.decl.loc13: type = interface_decl @Interface.2 [concrete = constants.%Interface.type.d726c6.2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface.1 {
// CHECK:STDOUT:   %Self: %Interface.type.d726c6.1 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.719cbb.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface.2 {
// CHECK:STDOUT:   %Self: %Interface.type.d726c6.2 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.719cbb.2]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Interface.2.%Self: %Interface.type.d726c6.2) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.719cbb.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redefine_with_dependents.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type.d726c6.1: type = facet_type <@Interface.1> [concrete]
// CHECK:STDOUT:   %Self.719cbb.1: %Interface.type.d726c6.1 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Interface.type.d726c6.2: type = facet_type <@Interface.2> [concrete]
// CHECK:STDOUT:   %Self.719cbb.2: %Interface.type.d726c6.2 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.719cbb.2 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type %Interface.type.d726c6.2 [concrete]
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, @Interface.2.%F.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Interface = %Interface.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl.loc4: type = interface_decl @Interface.1 [concrete = constants.%Interface.type.d726c6.1] {} {}
// CHECK:STDOUT:   %Interface.decl.loc13: type = interface_decl @Interface.2 [concrete = constants.%Interface.type.d726c6.2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface.1 {
// CHECK:STDOUT:   %Self: %Interface.type.d726c6.1 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.719cbb.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface.2 {
// CHECK:STDOUT:   %Self: %Interface.type.d726c6.2 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.719cbb.2]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: @F.%Self.as_type.loc14_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.%Self.as_type.loc14_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @F.%Self.as_type.loc14_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc14_14.1: type = splice_block %.loc14_14.2 [symbolic = %Self.as_type.loc14_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Interface.type.d726c6.2 = name_ref Self, @Interface.2.%Self [symbolic = %Self (constants.%Self.719cbb.2)]
// CHECK:STDOUT:       %Self.as_type.loc14_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc14_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc14_14.2: type = converted %Self.ref, %Self.as_type.loc14_14.2 [symbolic = %Self.as_type.loc14_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.%Self.as_type.loc14_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Interface.2.%Self: %Interface.type.d726c6.2) {
// CHECK:STDOUT:   %Self: %Interface.type.d726c6.2 = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.719cbb.2)]
// CHECK:STDOUT:   %Self.as_type.loc14_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.%Self.as_type.loc14_14.1 (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.719cbb.2) {
// CHECK:STDOUT:   %Self => constants.%Self.719cbb.2
// CHECK:STDOUT:   %Self.as_type.loc14_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_conflict_with_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Function.type.b14: type = fn_type @Function.1 [concrete]
// CHECK:STDOUT:   %Function: %Function.type.b14 = struct_value () [concrete]
// CHECK:STDOUT:   %Function.type.f90: type = facet_type <@Function.2> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Function = %Function.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Function.decl.loc4: %Function.type.b14 = fn_decl @Function.1 [concrete = constants.%Function] {} {}
// CHECK:STDOUT:   %Function.decl.loc13: type = interface_decl @Function.2 [concrete = constants.%Function.type.f90] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Function.2;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Function.1();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_conflict_with_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class.1 [concrete]
// CHECK:STDOUT:   %Class.type: type = facet_type <@Class.2> [concrete]
// CHECK:STDOUT:   %Self: %Class.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Class = %Class.decl.loc2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.decl.loc2: type = class_decl @Class.1 [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %Class.decl.loc11: type = interface_decl @Class.2 [concrete = constants.%Class.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Class.2 {
// CHECK:STDOUT:   %Self: %Class.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class.1;
// CHECK:STDOUT:
