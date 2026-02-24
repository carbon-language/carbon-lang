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

fn F(T:! Z, t:! T) {
  // TODO: We find the name `YY`, but then fail to do impl lookup for `Y` in
  // `T:! Z` since the identified facet type doesn't include `Y`. The
  // identified facet type needs to include extends inside interfaces?

  // CHECK:STDERR: fail_todo_extend.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Y` in type `T` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   T.YY();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  T.YY();
  // CHECK:STDERR: fail_todo_extend.carbon:[[@LINE+4]]:3: error: cannot convert type `T` that implements `Z` into type implementing `Y` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   T.(Y.YY)();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  T.(Y.YY)();

  // CHECK:STDERR: fail_todo_extend.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Y` in type `T` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   t.YY();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  t.YY();
}

// --- fail_todo_extend_with_self.carbon
library "[[@TEST_NAME]]";

interface Y(T:! type) {
  fn YY();
}

//@dump-sem-ir-begin
interface Z {
  extend require impls Y(Self);
}
//@dump-sem-ir-end

fn F(T:! Z) {
  // TODO: We find the name `YY`, but then fail to do impl lookup for `Y` in
  // `T:! Z` since the identified facet type doesn't include `Y`. The
  // identified facet type needs to include extends inside interfaces?
  //
  // CHECK:STDERR: fail_todo_extend_with_self.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Y(T)` in type `T` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   T.YY();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  T.YY();
  // CHECK:STDERR: fail_todo_extend_with_self.carbon:[[@LINE+4]]:3: error: cannot convert type `T` that implements `Z` into type implementing `Y(T)` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   T.(Y(T).YY)();
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  T.(Y(T).YY)();
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
  // CHECK:STDERR: fail_implicit_self_no_extend_name_lookup_fails.carbon:[[@LINE+4]]:3: error: member name `YY` not found in `Z` [MemberNameNotFoundInSpecificScope]
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
  // CHECK:STDERR: fail_explicit_self_no_extend_name_lookup_fails.carbon:[[@LINE+4]]:3: error: member name `YY` not found in `Z` [MemberNameNotFoundInSpecificScope]
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

// --- fail_non_self_doesnt_provide_witness_for_self.carbon
library "[[@TEST_NAME]]";

interface Y {
  fn YY();
}

class C(T:! type);

interface Z {
  require C(Self) impls Y;
}

fn F(T:! Z) {
  // This fails because `T impls Z` implies that `C(T) impls Y`, not that
  // `T impls Y`.
  // CHECK:STDERR: fail_non_self_doesnt_provide_witness_for_self.carbon:[[@LINE+4]]:3: error: cannot convert type `T` that implements `Z` into type implementing `Y` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   T.(Y.YY)();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  T.(Y.YY)();
}

// --- fail_todo_non_self_provide_witness_for_non_self.carbon
library "[[@TEST_NAME]]";

interface Y {
  fn YY();
}

class C(T:! type);

interface Z {
  require C(Self) impls Y;
}

fn F(T:! Z) {
  // CHECK:STDERR: fail_todo_non_self_provide_witness_for_non_self.carbon:[[@LINE+4]]:3: error: cannot convert type `C(T)` into type implementing `Y` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   C(T).(Y.YY)();
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  C(T).(Y.YY)();
}

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
  // CHECK:STDERR: fail_require_impls_without_self.carbon:[[@LINE+4]]:3: error: no `Self` reference found in `require` declaration; `Self` must appear in the self-type or as a generic argument for each required interface, but found interface `Y` without a `Self` argument [RequireImplsMissingSelf]
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
  // CHECK:STDERR: fail_require_impls_without_self_in_one_interface.carbon:[[@LINE+4]]:3: error: no `Self` reference found in `require` declaration; `Self` must appear in the self-type or as a generic argument for each required interface, but found interface `Y({})` without a `Self` argument [RequireImplsMissingSelf]
  // CHECK:STDERR:   require T impls Y(Self) & Y({});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  require T impls Y(Self) & Y({});
}

// --- fail_require_impls_without_self_in_multiple_interfaces.carbon
library "[[@TEST_NAME]]";

interface Y(T:! type) {}

interface Z(T:! type) {
  // Self does not appear in either interface, we should get an error for each.
  //
  // CHECK:STDERR: fail_require_impls_without_self_in_multiple_interfaces.carbon:[[@LINE+8]]:3: error: no `Self` reference found in `require` declaration; `Self` must appear in the self-type or as a generic argument for each required interface, but found interface `Y(())` without a `Self` argument [RequireImplsMissingSelf]
  // CHECK:STDERR:   require T impls Y(()) & Y({});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_require_impls_without_self_in_multiple_interfaces.carbon:[[@LINE+4]]:3: error: no `Self` reference found in `require` declaration; `Self` must appear in the self-type or as a generic argument for each required interface, but found interface `Y({})` without a `Self` argument [RequireImplsMissingSelf]
  // CHECK:STDERR:   require T impls Y(()) & Y({});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  require T impls Y(()) & Y({});
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
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.c59: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.c59 [symbolic]
// CHECK:STDOUT:   %T: %Z.type = symbolic_binding T, 0 [symbolic]
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
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c59]
// CHECK:STDOUT:   interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Z.WithSelf.Self.binding.as_type.impls.Y.type.require.decl = require_decl @Z.WithSelf.Self.binding.as_type.impls.Y.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type impls %Y.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @Z.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: type = specific_constant @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%Y.ref, @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(constants.%Self.c59) [concrete = constants.%Y.type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .YY = <poisoned>
// CHECK:STDOUT:   extend %.loc9
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.WithSelf.Self.binding.as_type.impls.Y.type.require {
// CHECK:STDOUT:     require @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%Self.as_type impls @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%Y.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.c59)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self.c59) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(constants.%Self.c59) {
// CHECK:STDOUT:   %Self => constants.%Self.c59
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_extend_with_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Y.type.82b: type = generic_interface_type @Y [concrete]
// CHECK:STDOUT:   %Y.generic: %Y.type.82b = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.c59: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.c59 [symbolic]
// CHECK:STDOUT:   %Y.type.432: type = facet_type <@Y, @Y(%Self.binding.as_type)> [symbolic]
// CHECK:STDOUT:   %require_complete.88f: <witness> = require_complete_type %Y.type.432 [symbolic]
// CHECK:STDOUT:   %T.ea3: %Z.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.ea3 [symbolic]
// CHECK:STDOUT:   %Y.type.112: type = facet_type <@Y, @Y(%T.binding.as_type)> [symbolic]
// CHECK:STDOUT:   %require_complete.dc8: <witness> = require_complete_type %Y.type.112 [symbolic]
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
// CHECK:STDOUT:   %Self.loc8: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c59]
// CHECK:STDOUT:   interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Z.WithSelf.Self.binding.as_type.impls.Y.type.require.decl = require_decl @Z.WithSelf.Self.binding.as_type.impls.Y.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type.loc9_18 impls %Y.type.loc9_30.1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type.loc9_18: type = facet_access_type @Z.%Self.loc8 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %Y.ref: %Y.type.82b = name_ref Y, file.%Y.decl [concrete = constants.%Y.generic]
// CHECK:STDOUT:     %Self.ref: %Z.type = name_ref Self, @Z.%Self.loc8 [symbolic = %Self (constants.%Self.c59)]
// CHECK:STDOUT:     %Self.as_type.loc9_30: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %.loc9: type = converted %Self.ref, %Self.as_type.loc9_30 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %Y.type.loc9_30.1: type = facet_type <@Y, @Y(constants.%Self.binding.as_type)> [symbolic = %Y.type.loc9_30.2 (constants.%Y.type.432)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: type = specific_constant @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%Y.type.loc9_30.1, @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(constants.%Self.c59) [symbolic = %Y.type (constants.%Y.type.432)]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc8
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .YY = <poisoned>
// CHECK:STDOUT:   extend %.loc9
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.WithSelf.Self.binding.as_type.impls.Y.type.require {
// CHECK:STDOUT:     require @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%Self.as_type.loc9_18 impls @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%Y.type.loc9_30.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(@Z.%Self.loc8: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.c59)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %Y.type.loc9_30.2: type = facet_type <@Y, @Y(%Self.binding.as_type)> [symbolic = %Y.type.loc9_30.2 (constants.%Y.type.432)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self.c59) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc9 => constants.%Self.c59
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %Y.type => constants.%Y.type.432
// CHECK:STDOUT:   %require_complete => constants.%require_complete.88f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(constants.%Self.c59) {
// CHECK:STDOUT:   %Self => constants.%Self.c59
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %Y.type.loc9_30.2 => constants.%Y.type.432
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T.ea3) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc9 => constants.%T.ea3
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %Y.type => constants.%Y.type.112
// CHECK:STDOUT:   %require_complete => constants.%require_complete.dc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_implicit_self_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.c59: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.c59 [symbolic]
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
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c59]
// CHECK:STDOUT:   interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Z.WithSelf.Self.binding.as_type.impls.Y.type.require.decl = require_decl @Z.WithSelf.Self.binding.as_type.impls.Y.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type impls %Y.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @Z.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.WithSelf.Self.binding.as_type.impls.Y.type.require {
// CHECK:STDOUT:     require @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%Self.as_type impls @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%Y.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.c59)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self.c59) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(constants.%Self.c59) {
// CHECK:STDOUT:   %Self => constants.%Self.c59
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_explicit_self_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.c59: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.c59 [symbolic]
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
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c59]
// CHECK:STDOUT:   interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Z.WithSelf.Self.binding.as_type.impls.Y.type.require.decl = require_decl @Z.WithSelf.Self.binding.as_type.impls.Y.type.require [concrete] {
// CHECK:STDOUT:     require %.loc9 impls %Y.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: %Z.type = name_ref Self, @Z.%Self [symbolic = %Self (constants.%Self.c59)]
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %.loc9: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.WithSelf.Self.binding.as_type.impls.Y.type.require {
// CHECK:STDOUT:     require @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%.loc9 impls @Z.WithSelf.Self.binding.as_type.impls.Y.type.require.%Y.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.c59)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self.c59) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Self.binding.as_type.impls.Y.type.require(constants.%Self.c59) {
// CHECK:STDOUT:   %Self => constants.%Self.c59
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- explicit_self_specific_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.c59: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.c59 [symbolic]
// CHECK:STDOUT:   %C.42e: type = class_type @C, @C(%Self.binding.as_type) [symbolic]
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
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c59]
// CHECK:STDOUT:   interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Z.WithSelf.C.impls.Y.type.require.decl = require_decl @Z.WithSelf.C.impls.Y.type.require [concrete] {
// CHECK:STDOUT:     require %C.loc9_17.1 impls %Y.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %Self.ref: %Z.type = name_ref Self, @Z.%Self [symbolic = %Self (constants.%Self.c59)]
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %.loc9: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %C.loc9_17.1: type = class_type @C, @C(constants.%Self.binding.as_type) [symbolic = %C.loc9_17.2 (constants.%C.42e)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.WithSelf.C.impls.Y.type.require {
// CHECK:STDOUT:     require @Z.WithSelf.C.impls.Y.type.require.%C.loc9_17.1 impls @Z.WithSelf.C.impls.Y.type.require.%Y.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.WithSelf.C.impls.Y.type.require(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.c59)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %C.loc9_17.2: type = class_type @C, @C(%Self.binding.as_type) [symbolic = %C.loc9_17.2 (constants.%C.42e)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self.c59) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.C.impls.Y.type.require(constants.%Self.c59) {
// CHECK:STDOUT:   %Self => constants.%Self.c59
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %C.loc9_17.2 => constants.%C.42e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- require_impls_where.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0: %Y.assoc_type = assoc_entity element0, @Y.%Y1 [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.c59: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.c59 [symbolic]
// CHECK:STDOUT:   %.Self: %Y.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @Y [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Y_where.type: type = facet_type <@Y where %impl.elem0 = %empty_tuple.type> [concrete]
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
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c59]
// CHECK:STDOUT:   interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require.decl = require_decl @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type impls %.loc7_19
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @Z.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.Self.ref: %Y.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc7_25: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %Y1.ref: %Y.assoc_type = name_ref Y1, @Y1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Y.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc7_32.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_32.2: type = converted %.loc7_32.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc7_19: type = where_expr %.Self [concrete = constants.%Y_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Y.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc7_32.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require {
// CHECK:STDOUT:     require @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require.%Self.as_type impls @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require.%.loc7_19
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.c59)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self.c59) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require(constants.%Self.c59) {
// CHECK:STDOUT:   %Self => constants.%Self.c59
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- require_impls_self_specific.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Z.type.352: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.352 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.0ed: type = facet_type <@Z, @Z(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %Z.type.0ed = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic]
// CHECK:STDOUT:   %Z.type.6aa: type = facet_type <@Z, @Z(%Self.binding.as_type)> [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: %Z.type.352 = interface_decl @Z [concrete = constants.%Z.generic] {
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
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc4_13.1)> [symbolic = %Z.type (constants.%Z.type.0ed)]
// CHECK:STDOUT:   %Self.loc4_23.2: @Z.%Z.type (%Z.type.0ed) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_23.1: @Z.%Z.type (%Z.type.0ed) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self)]
// CHECK:STDOUT:     interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Z.WithSelf.T.impls.Z.type.require.decl = require_decl @Z.WithSelf.T.impls.Z.type.require [concrete] {
// CHECK:STDOUT:       require %T.ref impls %Z.type.loc5_25.1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Z.%T.loc4_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Z.ref: %Z.type.352 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:       %.loc5_21: @Z.WithSelf.T.impls.Z.type.require.%Z.type.loc5_21 (%Z.type.0ed) = specific_constant @Z.%Self.loc4_23.1, @Z(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref: @Z.WithSelf.T.impls.Z.type.require.%Z.type.loc5_21 (%Z.type.0ed) = name_ref Self, %.loc5_21 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc5_25: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %Z.type.loc5_25.1: type = facet_type <@Z, @Z(constants.%Self.binding.as_type)> [symbolic = %Z.type.loc5_25.2 (constants.%Z.type.6aa)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_23.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Z = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Z = <poisoned>
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @Z.WithSelf.T.impls.Z.type.require {
// CHECK:STDOUT:       require @Z.WithSelf.T.impls.Z.type.require.%T.ref impls @Z.WithSelf.T.impls.Z.type.require.%Z.type.loc5_25.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.WithSelf.T.impls.Z.type.require(@Z.%T.loc4_13.2: type, @Z.%Self.loc4_23.1: @Z.%Z.type (%Z.type.0ed)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Z.type.loc5_21: type = facet_type <@Z, @Z(%T)> [symbolic = %Z.type.loc5_21 (constants.%Z.type.0ed)]
// CHECK:STDOUT:   %Self: @Z.WithSelf.T.impls.Z.type.require.%Z.type.loc5_21 (%Z.type.0ed) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %Z.type.loc5_25.2: type = facet_type <@Z, @Z(%Self.binding.as_type)> [symbolic = %Z.type.loc5_25.2 (constants.%Z.type.6aa)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self.binding.as_type) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%Self.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.T.impls.Z.type.require(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Z.type.loc5_21 => constants.%Z.type.0ed
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %Z.type.loc5_25.2 => constants.%Z.type.6aa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- require_self_in_requirement.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0: %Y.assoc_type = assoc_entity element0, @Y.%Y1 [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.c59: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self.c59 [symbolic]
// CHECK:STDOUT:   %.Self: %Y.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @Y [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %Y_where.type: type = facet_type <@Y where %impl.elem0 = %Self.binding.as_type> [symbolic]
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
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c59]
// CHECK:STDOUT:   interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require.decl = require_decl @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type.loc10_11 impls %.loc10_19
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type.loc10_11: type = facet_access_type @Z.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.Self.ref: %Y.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc10_25: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %Y1.ref: %Y.assoc_type = name_ref Y1, @Y1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Y.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %Self.ref: %Z.type = name_ref Self, @Z.%Self [symbolic = %Self (constants.%Self.c59)]
// CHECK:STDOUT:     %Self.as_type.loc10_31: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %.loc10_31: type = converted %Self.ref, %Self.as_type.loc10_31 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %.loc10_19: type = where_expr %.Self [symbolic = %Y_where.type (constants.%Y_where.type)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Y.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc10_31
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require {
// CHECK:STDOUT:     require @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require.%Self.as_type.loc10_11 impls @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require.%.loc10_19
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.c59)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %Y_where.type: type = facet_type <@Y where constants.%impl.elem0 = %Self.binding.as_type> [symbolic = %Y_where.type (constants.%Y_where.type)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self.c59) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Self.binding.as_type.impls.Y_where.type.require(constants.%Self.c59) {
// CHECK:STDOUT:   %Self => constants.%Self.c59
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %Y_where.type => constants.%Y_where.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- require_same.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Z.type.352: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.352 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.0ed: type = facet_type <@Z, @Z(%T)> [symbolic]
// CHECK:STDOUT:   %Self.822: %Z.type.0ed = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.b37: type = symbolic_binding_type Self, 1, %Self.822 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Z.decl: %Z.type.352 = interface_decl @Z [concrete = constants.%Z.generic] {
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
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc8_13.1)> [symbolic = %Z.type (constants.%Z.type.0ed)]
// CHECK:STDOUT:   %Self.loc8_23.2: @Z.%Z.type (%Z.type.0ed) = symbolic_binding Self, 1 [symbolic = %Self.loc8_23.2 (constants.%Self.822)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc8_23.1: @Z.%Z.type (%Z.type.0ed) = symbolic_binding Self, 1 [symbolic = %Self.loc8_23.2 (constants.%Self.822)]
// CHECK:STDOUT:     interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Z.WithSelf.Self.binding.as_type.impls.Z.type.require.decl = require_decl @Z.WithSelf.Self.binding.as_type.impls.Z.type.require [concrete] {
// CHECK:STDOUT:       require %Self.as_type impls %Z.type.loc11_20
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type @Z.%Self.loc8_23.1 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.b37)]
// CHECK:STDOUT:       %Z.ref: %Z.type.352 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Z.%T.loc8_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Z.type.loc11_20: type = facet_type <@Z, @Z(constants.%T)> [symbolic = %Z.type.loc11_11 (constants.%Z.type.0ed)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc8_23.1
// CHECK:STDOUT:     .Z = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Z = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @Z.WithSelf.Self.binding.as_type.impls.Z.type.require {
// CHECK:STDOUT:       require @Z.WithSelf.Self.binding.as_type.impls.Z.type.require.%Self.as_type impls @Z.WithSelf.Self.binding.as_type.impls.Z.type.require.%Z.type.loc11_20
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @Z.WithSelf.Self.binding.as_type.impls.Z.type.require(@Z.%T.loc8_13.2: type, @Z.%Self.loc8_23.1: @Z.%Z.type (%Z.type.0ed)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Z.type.loc11_11: type = facet_type <@Z, @Z(%T)> [symbolic = %Z.type.loc11_11 (constants.%Z.type.0ed)]
// CHECK:STDOUT:   %Self: @Z.WithSelf.Self.binding.as_type.impls.Z.type.require.%Z.type.loc11_11 (%Z.type.0ed) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.822)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.b37)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T) {
// CHECK:STDOUT:   %T.loc8_13.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T, constants.%Self.822) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Self.binding.as_type.impls.Z.type.require(constants.%T, constants.%Self.822) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Z.type.loc11_11 => constants.%Z.type.0ed
// CHECK:STDOUT:   %Self => constants.%Self.822
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.b37
// CHECK:STDOUT: }
// CHECK:STDOUT:
