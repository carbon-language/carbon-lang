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

// CHECK:STDERR: fail_redefine_without_dependents.carbon:[[@LINE+7]]:1: error: redefinition of `interface Interface`
// CHECK:STDERR: interface Interface {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefine_without_dependents.carbon:[[@LINE-5]]:1: note: previously defined here
// CHECK:STDERR: interface Interface { }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface Interface {
  fn F();
}

// --- fail_redefine_with_dependents.carbon

library "[[@TEST_NAME]]";

interface Interface {}

// CHECK:STDERR: fail_redefine_with_dependents.carbon:[[@LINE+7]]:1: error: redefinition of `interface Interface`
// CHECK:STDERR: interface Interface {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefine_with_dependents.carbon:[[@LINE-5]]:1: note: previously defined here
// CHECK:STDERR: interface Interface {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface Interface {
  fn F[self: Self]();
}

// --- fail_name_conflict_with_fn.carbon

library "[[@TEST_NAME]]";

fn Function();

// CHECK:STDERR: fail_name_conflict_with_fn.carbon:[[@LINE+7]]:1: error: duplicate name being declared in the same scope
// CHECK:STDERR: interface Function;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_name_conflict_with_fn.carbon:[[@LINE-5]]:1: note: name is previously declared here
// CHECK:STDERR: fn Function();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
interface Function;

// --- fail_name_conflict_with_class.carbon

class Class;

// CHECK:STDERR: fail_name_conflict_with_class.carbon:[[@LINE+6]]:1: error: duplicate name being declared in the same scope
// CHECK:STDERR: interface Class { }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_name_conflict_with_class.carbon:[[@LINE-5]]:1: note: name is previously declared here
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
interface Class { }

// CHECK:STDOUT: --- fail_redefine_without_dependents.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = interface_type @Interface [template]
// CHECK:STDOUT:   %Self.1: %Interface.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.type: type = interface_type @.1 [template]
// CHECK:STDOUT:   %Self.2: %.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %.type, %F.type [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @.1.%F.decl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %.decl: type = interface_decl @.1 [template = constants.%.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @.1 {
// CHECK:STDOUT:   %Self: %.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %.loc14: %.2 = assoc_entity element0, %F.decl [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc14
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@.1.%Self: %.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redefine_with_dependents.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = interface_type @Interface [template]
// CHECK:STDOUT:   %Self.1: %Interface.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.type: type = interface_type @.1 [template]
// CHECK:STDOUT:   %Self.2: %.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %.type, %F.type [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @.1.%F.decl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %.decl: type = interface_decl @.1 [template = constants.%.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @.1 {
// CHECK:STDOUT:   %Self: %.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: @F.%Self (%Self.2) = binding_pattern self
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: %.type = name_ref Self, @.1.%Self [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:     %.loc14_14.1: type = facet_type_access %Self.ref [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:     %.loc14_14.2: type = converted %Self.ref, %.loc14_14.1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:     %self.param: @F.%Self (%Self.2) = param self, runtime_param0
// CHECK:STDOUT:     %self: @F.%Self (%Self.2) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: %.2 = assoc_entity element0, %F.decl [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc14
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@.1.%Self: %.type) {
// CHECK:STDOUT:   %Self: %.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @F.%Self (%Self.2)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.2) {
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_conflict_with_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Function.type: type = fn_type @Function [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Function: %Function.type = struct_value () [template]
// CHECK:STDOUT:   %.type: type = interface_type @.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Function = %Function.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Function.decl: %Function.type = fn_decl @Function [template = constants.%Function] {} {}
// CHECK:STDOUT:   %.decl: type = interface_decl @.1 [template = constants.%.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @.1;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Function();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_conflict_with_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.type: type = interface_type @.1 [template]
// CHECK:STDOUT:   %Self: %.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT:   %.decl: type = interface_decl @.1 [template = constants.%.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @.1 {
// CHECK:STDOUT:   %Self: %.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class;
// CHECK:STDOUT:
