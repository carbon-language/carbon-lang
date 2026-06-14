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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_modifiers.carbon

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
  // CHECK:STDERR:   private fn F(self);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  private fn F(self);
}

// --- fail_protected_member.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_protected_member.carbon:[[@LINE+4]]:3: error: `protected` not allowed; requires class scope [ModifierProtectedNotAllowed]
  // CHECK:STDERR:   protected fn F(self);
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  protected fn F(self);
}

// CHECK:STDOUT: --- fail_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract.type: type = facet_type <@Abstract> [concrete]
// CHECK:STDOUT:   %Self: %Abstract.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Abstract = %Abstract.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Abstract.decl: type = interface_decl @Abstract [concrete = constants.%Abstract.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Abstract {
// CHECK:STDOUT:   %Self: %Abstract.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Abstract.WithSelf.decl = interface_with_self_decl @Abstract [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Abstract.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Default = %Default.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Default.decl: type = interface_decl @Default [concrete = constants.%Default.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Default;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_virtual.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Virtual.type: type = facet_type <@Virtual> [concrete]
// CHECK:STDOUT:   %Self: %Virtual.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Virtual = %Virtual.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Virtual.decl: type = interface_decl @Virtual [concrete = constants.%Virtual.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Virtual {
// CHECK:STDOUT:   %Self: %Virtual.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Virtual.WithSelf.decl = interface_with_self_decl @Virtual [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Virtual.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_protected.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Protected.type: type = facet_type <@Protected> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Protected = %Protected.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Protected.decl: type = interface_decl @Protected [concrete = constants.%Protected.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Protected;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_private_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt: %pattern_type = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt: %pattern_type = at_binding_pattern self, %self.param_patt [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F: %I.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F)] {
// CHECK:STDOUT:     %self.param_patt.loc9_16.1: @I.WithSelf.F.%pattern_type (%pattern_type) = value_param_pattern [symbolic = %self.param_patt.loc9_16.2 (constants.%self.param_patt)]
// CHECK:STDOUT:     %self.patt.loc9_16.1: @I.WithSelf.F.%pattern_type (%pattern_type) = at_binding_pattern self, %self.param_patt.loc9_16.1 [symbolic = %self.patt.loc9_16.2 (constants.%self.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @I.WithSelf.F.%Self.as_type.loc9_16.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_16.1: type = splice_block %.loc9_16.2 [symbolic = %Self.as_type.loc9_16.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc9_16.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc9_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc9_16.2: type = converted %Self.ref, %Self.as_type.loc9_16.2 [symbolic = %Self.as_type.loc9_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.WithSelf.F.%Self.as_type.loc9_16.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @I.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc9_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc9_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc9_16.1 [symbolic = %pattern_type (constants.%pattern_type)]
// CHECK:STDOUT:   %self.param_patt.loc9_16.2: @I.WithSelf.F.%pattern_type (%pattern_type) = value_param_pattern [symbolic = %self.param_patt.loc9_16.2 (constants.%self.param_patt)]
// CHECK:STDOUT:   %self.patt.loc9_16.2: @I.WithSelf.F.%pattern_type (%pattern_type) = at_binding_pattern self, %self.param_patt.loc9_16.2 [symbolic = %self.patt.loc9_16.2 (constants.%self.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.F.%Self.as_type.loc9_16.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc9_16.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type
// CHECK:STDOUT:   %self.param_patt.loc9_16.2 => constants.%self.param_patt
// CHECK:STDOUT:   %self.patt.loc9_16.2 => constants.%self.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_protected_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt: %pattern_type = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt: %pattern_type = at_binding_pattern self, %self.param_patt [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F: %I.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F)] {
// CHECK:STDOUT:     %self.param_patt.loc9_18.1: @I.WithSelf.F.%pattern_type (%pattern_type) = value_param_pattern [symbolic = %self.param_patt.loc9_18.2 (constants.%self.param_patt)]
// CHECK:STDOUT:     %self.patt.loc9_18.1: @I.WithSelf.F.%pattern_type (%pattern_type) = at_binding_pattern self, %self.param_patt.loc9_18.1 [symbolic = %self.patt.loc9_18.2 (constants.%self.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @I.WithSelf.F.%Self.as_type.loc9_18.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_18.1: type = splice_block %.loc9_18.2 [symbolic = %Self.as_type.loc9_18.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc9_18.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc9_18.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc9_18.2: type = converted %Self.ref, %Self.as_type.loc9_18.2 [symbolic = %Self.as_type.loc9_18.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.WithSelf.F.%Self.as_type.loc9_18.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @I.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc9_18.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc9_18.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc9_18.1 [symbolic = %pattern_type (constants.%pattern_type)]
// CHECK:STDOUT:   %self.param_patt.loc9_18.2: @I.WithSelf.F.%pattern_type (%pattern_type) = value_param_pattern [symbolic = %self.param_patt.loc9_18.2 (constants.%self.param_patt)]
// CHECK:STDOUT:   %self.patt.loc9_18.2: @I.WithSelf.F.%pattern_type (%pattern_type) = at_binding_pattern self, %self.param_patt.loc9_18.2 [symbolic = %self.patt.loc9_18.2 (constants.%self.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.F.%Self.as_type.loc9_18.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc9_18.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type
// CHECK:STDOUT:   %self.param_patt.loc9_18.2 => constants.%self.param_patt
// CHECK:STDOUT:   %self.patt.loc9_18.2 => constants.%self.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
