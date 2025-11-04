// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/require.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/require.carbon

// --- fail_todo_extend.carbon
library "[[@TEST_NAME]]";

interface Y {
  fn YY();
}

//@dump-sem-ir-begin
interface Z {
  extend require impls Y;
}
//@dump-sem-ir-end

fn F(T:! Z) {
  // CHECK:STDERR: fail_todo_extend.carbon:[[@LINE+4]]:3: error: member name `YY` not found in `Z` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   T.YY();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  T.YY();
  // CHECK:STDERR: fail_todo_extend.carbon:[[@LINE+4]]:3: error: cannot convert type `T` that implements `Z` into type implementing `Y` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   T.(Y.YY)();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  T.(Y.YY)();
}

// --- fail_todo_implicit_self_impls.carbon
library "[[@TEST_NAME]]";

interface Y {
  fn YY();
}

//@dump-sem-ir-begin
interface Z {
  require impls Y;
}
//@dump-sem-ir-end

fn F(T:! Z) {
  // CHECK:STDERR: fail_todo_implicit_self_impls.carbon:[[@LINE+4]]:3: error: cannot convert type `T` that implements `Z` into type implementing `Y` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   T.(Y.YY)();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  T.(Y.YY)();
}

// --- fail_todo_explicit_self_impls.carbon
library "[[@TEST_NAME]]";

interface Y {
  fn YY();
}

//@dump-sem-ir-begin
interface Z {
  require Self impls Y;
}
//@dump-sem-ir-end

fn F(T:! Z) {
  // CHECK:STDERR: fail_todo_explicit_self_impls.carbon:[[@LINE+4]]:3: error: cannot convert type `T` that implements `Z` into type implementing `Y` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   T.(Y.YY)();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  T.(Y.YY)();
}

// --- fail_implicit_self_no_extend_name_lookup_fails.carbon
library "[[@TEST_NAME]]";

interface Y {
  fn YY();
}

interface Z {
  require impls Y;
}

fn F(T:! Z) {
  // This should fail name lookup since Z does not extend Y.
  //
  // CHECK:STDERR: fail_implicit_self_no_extend_name_lookup_fails.carbon:[[@LINE+4]]:3: error: member name `YY` not found in `Z` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   T.YY();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  T.YY();
}

// --- fail_explicit_self_no_extend_name_lookup_fails.carbon
library "[[@TEST_NAME]]";

interface Y {
  fn YY();
}

interface Z {
  require Self impls Y;
}

fn F(T:! Z) {
  // This should fail name lookup since Z does not extend Y.
  //
  // CHECK:STDERR: fail_explicit_self_no_extend_name_lookup_fails.carbon:[[@LINE+4]]:3: error: member name `YY` not found in `Z` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   T.YY();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  T.YY();
}

// --- explicit_self_specific_impls.carbon
library "[[@TEST_NAME]]";

interface Y {}

class C(T:! type);

//@dump-sem-ir-begin
interface Z {
  require C(Self) impls Y;
}
//@dump-sem-ir-end

// --- require_impls_where.carbon
library "[[@TEST_NAME]]";

interface Y { let Y1:! type; }

//@dump-sem-ir-begin
interface Z {
  require impls Y where .Y1 = ();
}
//@dump-sem-ir-end

// --- require_impls_self_specific.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
interface Z(T:! type) {
  require T impls Z(Self);
}
//@dump-sem-ir-end

// --- fail_require_impls_without_self.carbon
library "[[@TEST_NAME]]";

interface Y {
  let Y1:! type;
}

interface Z(T:! type) {
  // Either the type `T` or the facet type `Y` must mention `Self` in a way that
  // it would appear in the type structure used for impl lookup (so inside a
  // `where` does not count). But they don't.
  //
  // CHECK:STDERR: fail_require_impls_without_self.carbon:[[@LINE+4]]:3: error: no `Self` reference found in `require` declaration; `Self` must appear in the self-type or as a generic parameter for each `interface` or `constraint` [RequireImplsMissingSelf]
  // CHECK:STDERR:   require T impls Y where .Y1 = Self;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  require T impls Y where .Y1 = Self;
}

// --- fail_require_impls_without_self_in_one_interface.carbon
library "[[@TEST_NAME]]";

interface Y(T:! type) {}

interface Z(T:! type) {
  // Self is in one interface but not the other.
  //
  // CHECK:STDERR: fail_require_impls_without_self_in_one_interface.carbon:[[@LINE+4]]:3: error: no `Self` reference found in `require` declaration; `Self` must appear in the self-type or as a generic parameter for each `interface` or `constraint` [RequireImplsMissingSelf]
  // CHECK:STDERR:   require T impls Y(Self) & Y({});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  require T impls Y(Self) & Y({});
}

// --- fail_self_impls_self.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  // CHECK:STDERR: fail_self_impls_self.carbon:[[@LINE+4]]:17: error: `require` declaration constrained by a non-facet type; expected an `interface` or `constraint` name after `impls` [RequireImplsMissingFacetType]
  // CHECK:STDERR:   require impls Self;
  // CHECK:STDERR:                 ^~~~
  // CHECK:STDERR:
  require impls Self;
}

// --- fail_impls_type.carbon
library "[[@TEST_NAME]]";

class C(T:! type) {}

interface Z(T:! type) {
  // CHECK:STDERR: fail_impls_type.carbon:[[@LINE+4]]:19: error: `require` declaration constrained by a non-facet type; expected an `interface` or `constraint` name after `impls` [RequireImplsMissingFacetType]
  // CHECK:STDERR:   require T impls C(Self);
  // CHECK:STDERR:                   ^~~~~~~
  // CHECK:STDERR:
  require T impls C(Self);
}

// --- fail_non_type_impls.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z(T:! type) {
  // CHECK:STDERR: fail_non_type_impls.carbon:[[@LINE+7]]:11: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   require 1 impls Y;
  // CHECK:STDERR:           ^
  // CHECK:STDERR: fail_non_type_impls.carbon:[[@LINE+4]]:11: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   require 1 impls Y;
  // CHECK:STDERR:           ^
  // CHECK:STDERR:
  require 1 impls Y;
}

// --- fail_impls_non_type.carbon
library "[[@TEST_NAME]]";

interface Y {}

interface Z(T:! type) {
  // CHECK:STDERR: fail_impls_non_type.carbon:[[@LINE+4]]:17: error: `require` declaration constrained by a non-facet type; expected an `interface` or `constraint` name after `impls` [RequireImplsMissingFacetType]
  // CHECK:STDERR:   require impls 1;
  // CHECK:STDERR:                 ^
  // CHECK:STDERR:
  require impls 1;
}

// --- require_self_in_requirement.carbon
library "[[@TEST_NAME]]";

interface Y {
  let Y1:! type;
}

//@dump-sem-ir-begin
interface Z {
  // Self can appear in a requirement.
  require impls Y where .Y1 = Self;
}
//@dump-sem-ir-end

// --- require_same.carbon
library "[[@TEST_NAME]]";

interface Y {
  require impls Y;
}

//@dump-sem-ir-begin
interface Z(T:! type) {
  // This is okay because an interface Z can be identified before it's complete,
  // unlike a named constraint.
  require impls Z(T);
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- fail_todo_extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.3b3: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.3b3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .YY = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_implicit_self_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.3b3: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.3b3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_explicit_self_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.3b3: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.3b3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- explicit_self_specific_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.3b3: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.3b3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- require_impls_where.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.3b3: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.3b3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- require_impls_self_specific.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Z.type.9fb: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.9fb = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.4d7: type = facet_type <@Z, @Z(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Z.type.4d7 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic]
// CHECK:STDOUT:   %Z.type.49d: type = facet_type <@Z, @Z(%Self.binding.as_type)> [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: %Z.type.9fb = interface_decl @Z [concrete = constants.%Z.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %T.loc4_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T.loc4_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type.loc4: type = facet_type <@Z, @Z(%T.loc4_13.1)> [symbolic = %Z.type.loc4 (constants.%Z.type.4d7)]
// CHECK:STDOUT:   %Self.loc4_23.2: @Z.%Z.type.loc4 (%Z.type.4d7) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.loc4_23.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %Z.type.loc5: type = facet_type <@Z, @Z(%Self.binding.as_type)> [symbolic = %Z.type.loc5 (constants.%Z.type.49d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_23.1: @Z.%Z.type.loc4 (%Z.type.4d7) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_23.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Z = <poisoned>
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self.binding.as_type) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%Self.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- require_self_in_requirement.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.3b3: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.3b3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- require_same.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Z.type.9fb: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.9fb = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.4d7: type = facet_type <@Z, @Z(%T)> [symbolic]
// CHECK:STDOUT:   %Self.c41: %Z.type.4d7 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.6a8: type = symbolic_binding_type Self, 1, %Self.c41 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: %Z.type.9fb = interface_decl @Z [concrete = constants.%Z.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %T.loc8_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(%T.loc8_13.2: type) {
// CHECK:STDOUT:   %T.loc8_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc8_13.1)> [symbolic = %Z.type (constants.%Z.type.4d7)]
// CHECK:STDOUT:   %Self.loc8_23.2: @Z.%Z.type (%Z.type.4d7) = symbolic_binding Self, 1 [symbolic = %Self.loc8_23.2 (constants.%Self.c41)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.loc8_23.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6a8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc8_23.1: @Z.%Z.type (%Z.type.4d7) = symbolic_binding Self, 1 [symbolic = %Self.loc8_23.2 (constants.%Self.c41)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc8_23.1
// CHECK:STDOUT:     .Z = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T) {
// CHECK:STDOUT:   %T.loc8_13.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
