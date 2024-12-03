// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/fail_modifiers.carbon

// --- fail_abstract.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_abstract.carbon:[[@LINE+4]]:1: error: `abstract` not allowed on `interface` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: abstract interface Abstract {
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract interface Abstract {
}

// --- fail_default.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_default.carbon:[[@LINE+4]]:1: error: `default` not allowed on `interface` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: default interface Default;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default interface Default;

// --- fail_virtual.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_virtual.carbon:[[@LINE+4]]:1: error: `virtual` not allowed on `interface` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: virtual interface Virtual {
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
virtual interface Virtual {
}

// --- fail_protected.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_protected.carbon:[[@LINE+4]]:1: error: `protected` not allowed; requires class scope [ModifierProtectedNotAllowed]
// CHECK:STDERR: protected interface Protected;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
protected interface Protected;

// --- fail_private_member.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_private_member.carbon:[[@LINE+4]]:3: error: `private` not allowed; requires class or file scope [ModifierPrivateNotAllowed]
  // CHECK:STDERR:   private fn F[self: Self]();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  private fn F[self: Self]();
}

// --- fail_protected_member.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_protected_member.carbon:[[@LINE+3]]:3: error: `protected` not allowed; requires class scope [ModifierProtectedNotAllowed]
  // CHECK:STDERR:   protected fn F[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~
  protected fn F[self: Self]();
}

// CHECK:STDOUT: --- fail_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract.type: type = facet_type <@Abstract> [template]
// CHECK:STDOUT:   %Self: %Abstract.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Abstract = %Abstract.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Abstract.decl: type = interface_decl @Abstract [template = constants.%Abstract.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Abstract {
// CHECK:STDOUT:   %Self: %Abstract.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Default = %Default.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Default.decl: type = interface_decl @Default [template = constants.%Default.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Default;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_virtual.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Virtual.type: type = facet_type <@Virtual> [template]
// CHECK:STDOUT:   %Self: %Virtual.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Virtual = %Virtual.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Virtual.decl: type = interface_decl @Virtual [template = constants.%Virtual.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Virtual {
// CHECK:STDOUT:   %Self: %Virtual.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_protected.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Protected.type: type = facet_type <@Protected> [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Protected = %Protected.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Protected.decl: type = interface_decl @Protected [template = constants.%Protected.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Protected;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_private_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %I.type, %F.type [template]
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: @F.%Self.as_type.loc9_22.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.%Self.as_type.loc9_22.1 (%Self.as_type) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc9_22.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc9_22.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc9: type = converted %Self.ref, %Self.as_type.loc9_22.2 [symbolic = %Self.as_type.loc9_22.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %self.param: @F.%Self.as_type.loc9_22.1 (%Self.as_type) = value_param runtime_param0
// CHECK:STDOUT:     %self: @F.%Self.as_type.loc9_22.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, %F.decl [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc9_22.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc9_22.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.%Self.as_type.loc9_22.1 (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc9_22.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_protected_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %I.type, %F.type [template]
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: @F.%Self.as_type.loc8_24.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.%Self.as_type.loc8_24.1 (%Self.as_type) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc8_24.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc8_24.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc8: type = converted %Self.ref, %Self.as_type.loc8_24.2 [symbolic = %Self.as_type.loc8_24.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %self.param: @F.%Self.as_type.loc8_24.1 (%Self.as_type) = value_param runtime_param0
// CHECK:STDOUT:     %self: @F.%Self.as_type.loc8_24.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, %F.decl [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc8_24.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_24.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.%Self.as_type.loc8_24.1 (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc8_24.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
