// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/forward_decls.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/forward_decls.carbon

// --- empty.carbon
library "[[@TEST_NAME]]";

interface I;
impl {} as I;

interface I {}
impl {} as I {}

// --- method.carbon
library "[[@TEST_NAME]]";

interface I;
impl {} as I;

interface I {
  fn G();
}
impl {} as I {
  fn G() {}
}

// --- combine.carbon
library "[[@TEST_NAME]]";

interface I;
impl {} as I & I;

interface I {}
impl {} as I & I {}

// --- associated_const.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
}
impl {} as I where .T = ();

impl {} as I where .T = () {}

// --- associated_const_compound_member_access.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
}
class C {}
impl C as I where .T = ();

let x: C.(I.T) = ();

impl C as I where .T = () {}

// --- associated_const_of_facet.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
}
class C {}
impl C as I where .T = ();

let x: (C as I).T = ();

impl C as I where .T = () {}

// --- fail_unset_associated_const.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
}
class C {}
impl C as I;

// CHECK:STDERR: fail_unset_associated_const.carbon:[[@LINE+4]]:8: error: accessing member from impl before it has a defined value [ImplAccessMemberBeforeSet]
// CHECK:STDERR: let x: (C as I).T = ();
// CHECK:STDERR:        ^~~~~~~~~~
// CHECK:STDERR:
let x: (C as I).T = ();

// CHECK:STDERR: fail_unset_associated_const.carbon:[[@LINE+7]]:1: error: associated constant T not given a value in impl of interface I [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: impl C as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_unset_associated_const.carbon:[[@LINE-14]]:7: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR:   let T:! type;
// CHECK:STDERR:       ^~~~~~~~
// CHECK:STDERR:
impl C as I {}

// --- fail_associated_const_before_interface_definition.carbon
library "[[@TEST_NAME]]";

interface I;
class C;
class D {}
// CHECK:STDERR: fail_associated_const_before_interface_definition.carbon:[[@LINE+7]]:19: error: member access into facet of incomplete type `I` [IncompleteTypeInMemberAccessOfFacet]
// CHECK:STDERR: impl D as I where .T = C;
// CHECK:STDERR:                   ^~
// CHECK:STDERR: fail_associated_const_before_interface_definition.carbon:[[@LINE-6]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
impl D as I where .T = C;

interface I {}
class C {}
// CHECK:STDERR: fail_associated_const_before_interface_definition.carbon:[[@LINE+4]]:19: error: member name `T` not found in `I` [MemberNameNotFoundInSpecificScope]
// CHECK:STDERR: impl D as I where .T = C {}
// CHECK:STDERR:                   ^~
// CHECK:STDERR:
impl D as I where .T = C {}

// --- associated_const_of_parameterized.carbon
library "[[@TEST_NAME]]";

interface I(U:! type) {
  let T:! type;
}
class C;
class D {}
impl D as I(C) where .T = C;

class C {}
impl D as I(C) where .T = C {}

// --- find_incomplete_impl.carbon
library "[[@TEST_NAME]]";

interface I;
class D {}
impl D as I;
class C(T:! I);

fn F(x: C(D));

interface I {}
impl D as I {}
class C(T:! I) {}

fn F(unused x: C(D)) {}

// --- fail_todo_two_interfaces.carbon
library "[[@TEST_NAME]]";

interface I;
interface J;

// CHECK:STDERR: fail_todo_two_interfaces.carbon:[[@LINE+4]]:1: error: impl as 2 interfaces, expected 1 [ImplOfNotOneInterface]
// CHECK:STDERR: impl {} as I & J;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl {} as I & J;

interface I {}
interface J {}

// --- fail_never_assigned_associated_const.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
  let U:! type;
}
class C {}

impl C as I where .T = ();

// CHECK:STDERR: fail_never_assigned_associated_const.carbon:[[@LINE+7]]:1: error: associated constant U not given a value in impl of interface I [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: impl C as I where .T = () {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_never_assigned_associated_const.carbon:[[@LINE-9]]:7: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR:   let U:! type;
// CHECK:STDERR:       ^~~~~~~~
// CHECK:STDERR:
impl C as I where .T = () {}

// --- example_from_proposal_5168.carbon
library "[[@TEST_NAME]]";

interface X;

// Allowed to use incomplete interfaces in function declarations.
fn F(_:! X);
fn G[U:! X](u: U);

class C;

// Allowed to use incomplete types and interfaces in impl declarations.
impl C as X;

interface Y;

interface X {
  // TODO: Should be allowed to use an incomplete interface here,
  // but `require Self impls` is not supported yet.
  // require Self impls Y;
}

// Classes must be defined before being used in a function definition.
class C {}

fn H(c: C) {
  // Allowed since C is complete and we have a declaration `impl C as X;`
  F(C);
  G(c);
}

// The above declarations require that `interface Y`, `fn F` (since it is
// generic), and `impl C as X` are defined in the same file.
interface Y {}
fn F(_:! X) {}
fn G[U:! X](unused u: U) {}
impl C as Y;
impl C as X {}
impl C as Y {}

// --- fail_todo_impl_in_interface_definition.carbon
library "[[@TEST_NAME]]";

// This test uses many unsupported features, and is expected to change.
interface I {
  // CHECK:STDERR: fail_todo_impl_in_interface_definition.carbon:[[@LINE+4]]:3: error: semantics TODO: `interface modifier` [SemanticsTodo]
  // CHECK:STDERR:   default fn F() {
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn F() {
    class C {}
    impl C as I;

    // CHECK:STDERR: fail_todo_impl_in_interface_definition.carbon:[[@LINE+7]]:5: error: missing implementation of F in impl of interface I [ImplMissingFunction]
    // CHECK:STDERR:     impl C as I {}
    // CHECK:STDERR:     ^~~~~~~~~~~~~
    // CHECK:STDERR: fail_todo_impl_in_interface_definition.carbon:[[@LINE-7]]:3: note: associated function F declared here [AssociatedFunctionHere]
    // CHECK:STDERR:   default fn F() {
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    impl C as I {}
  }
}

// --- fail_todo_impl_in_interface_definition_with_associated.carbon
library "[[@TEST_NAME]]";

// This test uses many unsupported features, and is expected to change.
interface I {
  let U:! type;
  // CHECK:STDERR: fail_todo_impl_in_interface_definition_with_associated.carbon:[[@LINE+4]]:3: error: semantics TODO: `interface modifier` [SemanticsTodo]
  // CHECK:STDERR:   default fn F() {
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn F() {
    class C {}
    impl C as I where .U = C;

    // CHECK:STDERR: fail_todo_impl_in_interface_definition_with_associated.carbon:[[@LINE+7]]:5: error: missing implementation of F in impl of interface I [ImplMissingFunction]
    // CHECK:STDERR:     impl C as I where .U = C {}
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_todo_impl_in_interface_definition_with_associated.carbon:[[@LINE-7]]:3: note: associated function F declared here [AssociatedFunctionHere]
    // CHECK:STDERR:   default fn F() {
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    impl C as I where .U = C {}
  }
}

// CHECK:STDOUT: --- empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @empty_struct_type.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %empty_struct_type, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl.loc3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl.loc3: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %.loc4_7.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc4_7.2: type = converted %.loc4_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc4: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc6: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %.loc7_7.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc7_7.2: type = converted %.loc7_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc7: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_struct_type.as.I.impl: %.loc4_7.2 as %I.ref.loc4 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @empty_struct_type.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @empty_struct_type.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.G.type.047: type = fn_type @I.WithSelf.G, @I.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.G.8c4: %I.WithSelf.G.type.047 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.G.decl [concrete]
// CHECK:STDOUT:   %empty_struct_type.as.I.impl.G.type: type = fn_type @empty_struct_type.as.I.impl.G [concrete]
// CHECK:STDOUT:   %empty_struct_type.as.I.impl.G: %empty_struct_type.as.I.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %empty_struct_type, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.WithSelf.G.type.ed6: type = fn_type @I.WithSelf.G, @I.WithSelf(%I.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.G.42b: %I.WithSelf.G.type.ed6 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl.loc3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl.loc3: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %.loc4_7.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc4_7.2: type = converted %.loc4_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc4: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc6: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %.loc9_7.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc9_7.2: type = converted %.loc9_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc9: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.G.decl: @I.WithSelf.%I.WithSelf.G.type (%I.WithSelf.G.type.047) = fn_decl @I.WithSelf.G [symbolic = @I.WithSelf.%I.WithSelf.G (constants.%I.WithSelf.G.8c4)] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.G.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = @I.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_struct_type.as.I.impl: %.loc4_7.2 as %I.ref.loc4 {
// CHECK:STDOUT:   %empty_struct_type.as.I.impl.G.decl: %empty_struct_type.as.I.impl.G.type = fn_decl @empty_struct_type.as.I.impl.G [concrete = constants.%empty_struct_type.as.I.impl.G] {} {}
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%empty_struct_type.as.I.impl.G.decl), @empty_struct_type.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %empty_struct_type.as.I.impl.G.decl
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.G(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_struct_type.as.I.impl.G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %I.WithSelf.G.type => constants.%I.WithSelf.G.type.047
// CHECK:STDOUT:   %I.WithSelf.G => constants.%I.WithSelf.G.8c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.G(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.G.type => constants.%I.WithSelf.G.type.ed6
// CHECK:STDOUT:   %I.WithSelf.G => constants.%I.WithSelf.G.42b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.G(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- combine.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.b10: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.b10 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.4bd: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.d15: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.4bd, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %I.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @empty_struct_type.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.ab9: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %empty_struct_type, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.8d3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.8d3), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl.loc3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl.loc3: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %.loc4_14.1: type = value_of_initializer @empty_struct_type.as.I.impl.%type.as.BitAndWith.impl.Op.call.loc4 [concrete = constants.%I.type]
// CHECK:STDOUT:   %.loc4_14.2: type = converted @empty_struct_type.as.I.impl.%type.as.BitAndWith.impl.Op.call.loc4, %.loc4_14.1 [concrete = constants.%I.type]
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %.loc4_7.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc4_7.2: type = converted %.loc4_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc4_12: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:     %I.ref.loc4_16: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:     %impl.elem0.loc4: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:     %bound_method.loc4: <bound method> = bound_method %I.ref.loc4_12, %impl.elem0.loc4 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:     %type.as.BitAndWith.impl.Op.call.loc4: init type = call %bound_method.loc4(%I.ref.loc4_12, %I.ref.loc4_16) [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc6: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %.loc7_14.1: type = value_of_initializer @empty_struct_type.as.I.impl.%type.as.BitAndWith.impl.Op.call.loc7 [concrete = constants.%I.type]
// CHECK:STDOUT:   %.loc7_14.2: type = converted @empty_struct_type.as.I.impl.%type.as.BitAndWith.impl.Op.call.loc7, %.loc7_14.1 [concrete = constants.%I.type]
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %.loc7_7.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc7_7.2: type = converted %.loc7_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc7_12: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:     %I.ref.loc7_16: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:     %impl.elem0.loc7: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:     %bound_method.loc7: <bound method> = bound_method %I.ref.loc7_12, %impl.elem0.loc7 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:     %type.as.BitAndWith.impl.Op.call.loc7: init type = call %bound_method.loc7(%I.ref.loc7_12, %I.ref.loc7_16) [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.ab9]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_struct_type.as.I.impl: %.loc4_7.2 as file.%.loc4_14.2 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @empty_struct_type.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.ab9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%T [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @empty_struct_type.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %empty_struct_type, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %.loc6_7.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc6: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.2: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc6: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc6: type = facet_access_type %.Self.ref.loc6 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc6_20: type = converted %.Self.ref.loc6, %.Self.as_type.loc6 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %T.ref.loc6: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc6: type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc6_26.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc6_26.2: type = converted %.loc6_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc6_14: type = where_expr %.Self.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc6, %.loc6_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc8: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.1: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc8: %I.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc8: type = facet_access_type %.Self.ref.loc8 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref.loc8, %.Self.as_type.loc8 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %T.ref.loc8: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc8: type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc8_26.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_26.2: type = converted %.loc8_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self.1 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc8, %.loc8_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%T [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%T)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_struct_type.as.I.impl: %.loc6_7.2 as %.loc6_14 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @empty_struct_type.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_const_compound_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%T [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc7: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.2: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc7: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc7: type = facet_access_type %.Self.ref.loc7 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc7_19: type = converted %.Self.ref.loc7, %.Self.as_type.loc7 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %T.ref.loc7: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc7: type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc7_25.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_25.2: type = converted %.loc7_25.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc7_13: type = where_expr %.Self.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc7, %.loc7_25.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_9.1: type = splice_block %impl.elem0 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, %I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.ref: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:     %.loc9_9.2: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_19: %empty_tuple.type = converted @__global_init.%.loc9, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %x: %empty_tuple.type = value_binding x, %.loc9_19
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc11: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc11: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.1: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc11: %I.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc11: type = facet_access_type %.Self.ref.loc11 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc11_19: type = converted %.Self.ref.loc11, %.Self.as_type.loc11 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %T.ref.loc11: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc11: type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc11_25.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc11_25.2: type = converted %.loc11_25.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc11_13: type = where_expr %.Self.1 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc11, %.loc11_25.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%T [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%T)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref.loc7 as %.loc7_13 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_const_of_facet.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%T [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc7: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.2: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc7: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc7: type = facet_access_type %.Self.ref.loc7 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc7_19: type = converted %.Self.ref.loc7, %.Self.as_type.loc7 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %T.ref.loc7: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc7: type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc7_25.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_25.2: type = converted %.loc7_25.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc7_13: type = where_expr %.Self.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc7, %.loc7_25.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_16.1: type = splice_block %impl.elem0 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, %I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:     %.loc9_11: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:     %as_type: type = facet_access_type %.loc9_11 [concrete = constants.%C]
// CHECK:STDOUT:     %.loc9_16.2: type = converted %.loc9_11, %as_type [concrete = constants.%C]
// CHECK:STDOUT:     %T.ref: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_22: %empty_tuple.type = converted @__global_init.%.loc9, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %x: %empty_tuple.type = value_binding x, %.loc9_22
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc11: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc11: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.1: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc11: %I.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc11: type = facet_access_type %.Self.ref.loc11 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc11_19: type = converted %.Self.ref.loc11, %.Self.as_type.loc11 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %T.ref.loc11: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc11: type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc11_25.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc11_25.2: type = converted %.loc11_25.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc11_13: type = where_expr %.Self.1 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc11, %.loc11_25.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%T [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%T)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref.loc7 as %.loc7_13 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_unset_associated_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%T [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc7: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: <error> = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_16.1: type = splice_block %impl.elem0 [concrete = <error>] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, %I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:     %.loc13_11: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:     %as_type: type = facet_access_type %.loc13_11 [concrete = constants.%C]
// CHECK:STDOUT:     %.loc13_16.2: type = converted %.loc13_11, %as_type [concrete = constants.%C]
// CHECK:STDOUT:     %T.ref: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%I.impl_witness, element0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: <error> = value_binding x, <error> [concrete = <error>]
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc22: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc22: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%T [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%T)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref.loc7 as %I.ref.loc7 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_associated_const_before_interface_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl.loc3
// CHECK:STDOUT:     .C = %C.decl.loc4
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl.loc3: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %C.decl.loc4: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @D.as.<error>.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref.loc13: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %I.ref.loc13: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.2: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc13: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc13: type = facet_access_type %.Self.ref.loc13 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc13_19: type = converted %.Self.ref.loc13, %.Self.as_type.loc13 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %T.ref.loc13: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %C.ref.loc13: type = name_ref C, file.%C.decl.loc4 [concrete = constants.%C]
// CHECK:STDOUT:     %.loc13_13: type = where_expr %.Self.2 [concrete = <error>] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_rewrite %T.ref.loc13, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc15: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %C.decl.loc16: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @D.as.<error>.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref.loc21: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %I.ref.loc21: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.1: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc21: %I.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc21: type = facet_access_type %.Self.ref.loc21 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc21_19: type = converted %.Self.ref.loc21, %.Self.as_type.loc21 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %T.ref.loc21: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %C.ref.loc21: type = name_ref C, file.%C.decl.loc4 [concrete = constants.%C]
// CHECK:STDOUT:     %.loc21_13: type = where_expr %.Self.1 [concrete = <error>] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_rewrite %T.ref.loc21, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.<error>.impl: %D.ref.loc13 as %.loc13_13 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_const_of_parameterized.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %I.type.609: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.609 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.1ab: type = facet_type <@I, @I(%U)> [symbolic]
// CHECK:STDOUT:   %Self.fdb: %I.type.1ab = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.76c: type = assoc_entity_type @I, @I(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.eea: %I.assoc_type.76c = assoc_entity element0, @I.WithSelf.%T [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type.807: type = facet_type <@I, @I(%C)> [concrete]
// CHECK:STDOUT:   %.Self.9dc: %I.type.807 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.b60: %I.type.807 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.2e0: type = assoc_entity_type @I, @I(%C) [concrete]
// CHECK:STDOUT:   %assoc0.501: %I.assoc_type.2e0 = assoc_entity element0, @I.WithSelf.%T [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.9dc [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.9dc, @I, @I(%C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I, @I(%C) where %impl.elem0 = %C> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @D.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.807 = facet_value %D, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl.loc6
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: %I.type.609 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %U.patt: %pattern_type = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_17.1: type = splice_block %.loc3_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc3_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc3_13.2: type = symbolic_binding U, 0 [symbolic = %U.loc3_13.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc6: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @D.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref.loc8: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %I.ref.loc8: %I.type.609 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %C.ref.loc8_13: type = name_ref C, file.%C.decl.loc6 [concrete = constants.%C]
// CHECK:STDOUT:     %I.type.loc8: type = facet_type <@I, @I(constants.%C)> [concrete = constants.%I.type.807]
// CHECK:STDOUT:     %.Self.2: %I.type.807 = symbolic_binding .Self [symbolic_self = constants.%.Self.9dc]
// CHECK:STDOUT:     %.Self.ref.loc8: %I.type.807 = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.9dc]
// CHECK:STDOUT:     %.Self.as_type.loc8: type = facet_access_type %.Self.ref.loc8 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc8_22.1: type = converted %.Self.ref.loc8, %.Self.as_type.loc8 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc8_22.2: %I.assoc_type.2e0 = specific_constant @T.%assoc0, @I.WithSelf(constants.%C, constants.%.Self.9dc) [concrete = constants.%assoc0.501]
// CHECK:STDOUT:     %T.ref.loc8: %I.assoc_type.2e0 = name_ref T, %.loc8_22.2 [concrete = constants.%assoc0.501]
// CHECK:STDOUT:     %impl.elem0.loc8: type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %C.ref.loc8_27: type = name_ref C, file.%C.decl.loc6 [concrete = constants.%C]
// CHECK:STDOUT:     %.loc8_16: type = where_expr %.Self.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type.807
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc8, %C.ref.loc8_27
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc10: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @D.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref.loc11: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %I.ref.loc11: %I.type.609 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %C.ref.loc11_13: type = name_ref C, file.%C.decl.loc6 [concrete = constants.%C]
// CHECK:STDOUT:     %I.type.loc11: type = facet_type <@I, @I(constants.%C)> [concrete = constants.%I.type.807]
// CHECK:STDOUT:     %.Self.1: %I.type.807 = symbolic_binding .Self [symbolic_self = constants.%.Self.9dc]
// CHECK:STDOUT:     %.Self.ref.loc11: %I.type.807 = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.9dc]
// CHECK:STDOUT:     %.Self.as_type.loc11: type = facet_access_type %.Self.ref.loc11 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc11_22.1: type = converted %.Self.ref.loc11, %.Self.as_type.loc11 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc11_22.2: %I.assoc_type.2e0 = specific_constant @T.%assoc0, @I.WithSelf(constants.%C, constants.%.Self.9dc) [concrete = constants.%assoc0.501]
// CHECK:STDOUT:     %T.ref.loc11: %I.assoc_type.2e0 = name_ref T, %.loc11_22.2 [concrete = constants.%assoc0.501]
// CHECK:STDOUT:     %impl.elem0.loc11: type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %C.ref.loc11_27: type = name_ref C, file.%C.decl.loc6 [concrete = constants.%C]
// CHECK:STDOUT:     %.loc11_16: type = where_expr %.Self.1 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type.807
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc11, %C.ref.loc11_27
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%U.loc3_13.2: type) {
// CHECK:STDOUT:   %U.loc3_13.1: type = symbolic_binding U, 0 [symbolic = %U.loc3_13.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%U.loc3_13.1)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self.loc3_23.2: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.fdb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_23.1: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.fdb)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:       %assoc0: @I.WithSelf.%I.assoc_type (%I.assoc_type.76c) = assoc_entity element0, @I.WithSelf.%T [symbolic = @I.WithSelf.%assoc0 (constants.%assoc0.eea)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_23.1
// CHECK:STDOUT:     .T = @T.%assoc0
// CHECK:STDOUT:     witness = (@I.WithSelf.%T)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.I.impl: %D.ref.loc8 as %.loc8_16 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @D.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%C [concrete = constants.%C]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%U) {
// CHECK:STDOUT:   %U.loc3_13.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%U, constants.%Self.fdb) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%C) {
// CHECK:STDOUT:   %U.loc3_13.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.807
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.b60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%C, constants.%Self.fdb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %U => constants.%C
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.2e0
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.501
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%C, constants.%.Self.9dc) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %U => constants.%C
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.2e0
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.501
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%C, constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %U => constants.%C
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.2e0
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.501
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- find_incomplete_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @D.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.bba: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %D, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %C.229: type = class_type @C, @C(%I.facet) [concrete]
// CHECK:STDOUT:   %pattern_type.b40: type = pattern_type %C.229 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl.loc3
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .C = %C.decl.loc6
// CHECK:STDOUT:     .F = %F.decl.loc8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl.loc3: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @D.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref.loc5: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %I.ref.loc5: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc6: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.9d9 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %I.ref.loc6 [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref.loc6: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_9.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc6_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc8: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.b40 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.b40 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param.loc8: %C.229 = value_param call_param0
// CHECK:STDOUT:     %.loc8_12.1: type = splice_block %C.loc8 [concrete = constants.%C.229] {
// CHECK:STDOUT:       %C.ref.loc8: %C.type = name_ref C, file.%C.decl.loc6 [concrete = constants.%C.generic]
// CHECK:STDOUT:       %D.ref.loc8: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %I.facet.loc8: %I.type = facet_value %D.ref.loc8, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:       %.loc8_12.2: %I.type = converted %D.ref.loc8, %I.facet.loc8 [concrete = constants.%I.facet]
// CHECK:STDOUT:       %C.loc8: type = class_type @C, @C(constants.%I.facet) [concrete = constants.%C.229]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.loc8: %C.229 = value_binding x, %x.param.loc8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc10: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @D.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref.loc11: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %I.ref.loc11: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc12: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.9d9 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12: type = splice_block %I.ref.loc12 [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref.loc12: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12: %I.type = symbolic_binding T, 0 [symbolic = %T.loc6_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc14: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.b40 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.b40 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param.loc14: %C.229 = value_param call_param0
// CHECK:STDOUT:     %.loc14_19.1: type = splice_block %C.loc14 [concrete = constants.%C.229] {
// CHECK:STDOUT:       %C.ref.loc14: %C.type = name_ref C, file.%C.decl.loc6 [concrete = constants.%C.generic]
// CHECK:STDOUT:       %D.ref.loc14: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %I.facet.loc14: %I.type = facet_value %D.ref.loc14, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:       %.loc14_19.2: %I.type = converted %D.ref.loc14, %I.facet.loc14 [concrete = constants.%I.facet]
// CHECK:STDOUT:       %C.loc14: type = class_type @C, @C(constants.%I.facet) [concrete = constants.%C.229]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.loc14: %C.229 = value_binding x, %x.param.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.I.impl: %D.ref.loc5 as %I.ref.loc5 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @D.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc6_9.2: %I.type) {
// CHECK:STDOUT:   %T.loc6_9.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc6_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.bba
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param.loc14: %C.229) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc6_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%I.facet) {
// CHECK:STDOUT:   %T.loc6_9.1 => constants.%I.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_two_interfaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.b10: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.b10 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.4bd: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.d15: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.4bd, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %I.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@I & @J> [concrete]
// CHECK:STDOUT:   %Self.ab9: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.8a1: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.8d3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.8d3), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl.loc3
// CHECK:STDOUT:     .J = %J.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl.loc3: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl.loc4: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %.loc10_14.1: type = value_of_initializer @empty_struct_type.as.<error>.impl.%type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc10_14.2: type = converted @empty_struct_type.as.<error>.impl.%type.as.BitAndWith.impl.Op.call, %.loc10_14.1 [concrete = constants.%facet_type]
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.<error>.impl [concrete] {} {
// CHECK:STDOUT:     %.loc10_7.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc10_7.2: type = converted %.loc10_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl.loc4 [concrete = constants.%J.type]
// CHECK:STDOUT:     %impl.elem0: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %I.ref, %impl.elem0 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:     %type.as.BitAndWith.impl.Op.call: init type = call %bound_method(%I.ref, %J.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc12: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl.loc13: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.ab9]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.8a1]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_struct_type.as.<error>.impl: %.loc10_7.2 as file.%.loc10_14.2;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.ab9) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.8a1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_never_assigned_associated_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%T [concrete]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.WithSelf.%U [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc9: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.2: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc9: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc9: type = facet_access_type %.Self.ref.loc9 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc9_19: type = converted %.Self.ref.loc9, %.Self.as_type.loc9 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %T.ref.loc9: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc9: type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc9_25.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc9_25.2: type = converted %.loc9_25.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc9_13: type = where_expr %.Self.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc9, %.loc9_25.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc18: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.1: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc18: %I.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc18: type = facet_access_type %.Self.ref.loc18 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc18_19: type = converted %.Self.ref.loc18, %.Self.as_type.loc18 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %T.ref.loc18: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc18: type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc18_25.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc18_25.2: type = converted %.loc18_25.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc18_13: type = where_expr %.Self.1 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc18, %.loc18_25.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%T [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc1: %I.assoc_type = assoc_entity element1, @I.WithSelf.%U [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   .U = @U.%assoc1
// CHECK:STDOUT:   witness = (@I.WithSelf.%T, @I.WithSelf.%U)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref.loc9 as %.loc9_13 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, <error>), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- example_from_proposal_5168.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X.type: type = facet_type <@X> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.b4b: type = pattern_type %X.type [concrete]
// CHECK:STDOUT:   %_: %X.type = symbolic_binding _, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %U: %X.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 0, %U [symbolic]
// CHECK:STDOUT:   %pattern_type.b36: type = pattern_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %X.impl_witness: <witness> = impl_witness @C.as.X.impl.%X.impl_witness_table [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.e52: %X.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %X.facet: %X.type = facet_value %C, (%X.impl_witness) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%X.facet) [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%X.facet) [concrete]
// CHECK:STDOUT:   %Self.162: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness @C.as.Y.impl.%Y.impl_witness_table [concrete]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %C, (%Y.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .X = %X.decl.loc3
// CHECK:STDOUT:     .F = %F.decl.loc6
// CHECK:STDOUT:     .G = %G.decl.loc7
// CHECK:STDOUT:     .C = %C.decl.loc9
// CHECK:STDOUT:     .Y = %Y.decl.loc14
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %X.decl.loc3: type = interface_decl @X [concrete = constants.%X.type] {} {}
// CHECK:STDOUT:   %F.decl.loc6: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %_.patt: %pattern_type.b4b = symbolic_binding_pattern _, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %X.ref.loc6 [concrete = constants.%X.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %X.ref.loc6: type = name_ref X, file.%X.decl.loc3 [concrete = constants.%X.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %_.loc6_6.2: %X.type = symbolic_binding _, 0 [symbolic = %_.loc6_6.1 (constants.%_)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl.loc7: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %U.patt: %pattern_type.b4b = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %u.param_patt: @G.%pattern_type (%pattern_type.b36) = value_param_pattern [concrete]
// CHECK:STDOUT:     %u.patt: @G.%pattern_type (%pattern_type.b36) = at_binding_pattern u, %u.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_10: type = splice_block %X.ref.loc7 [concrete = constants.%X.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %X.ref.loc7: type = name_ref X, file.%X.decl.loc3 [concrete = constants.%X.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_6.2: %X.type = symbolic_binding U, 0 [symbolic = %U.loc7_6.1 (constants.%U)]
// CHECK:STDOUT:     %u.param.loc7: @G.%U.binding.as_type (%U.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_16.1: type = splice_block %.loc7_16.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:       %U.ref.loc7: %X.type = name_ref U, %U.loc7_6.2 [symbolic = %U.loc7_6.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type.loc7: type = facet_access_type %U.ref.loc7 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       %.loc7_16.2: type = converted %U.ref.loc7, %U.as_type.loc7 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u.loc7: @G.%U.binding.as_type (%U.binding.as_type) = value_binding u, %u.param.loc7
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc9: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.X.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc12: type = name_ref C, file.%C.decl.loc9 [concrete = constants.%C]
// CHECK:STDOUT:     %X.ref.loc12: type = name_ref X, file.%X.decl.loc3 [concrete = constants.%X.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.decl.loc14: type = interface_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT:   %X.decl.loc16: type = interface_decl @X [concrete = constants.%X.type] {} {}
// CHECK:STDOUT:   %C.decl.loc23: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.7c7 = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.7c7 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc25: type = name_ref C, file.%C.decl.loc9 [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.decl.loc33: type = interface_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT:   %F.decl.loc34: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %_.patt: %pattern_type.b4b = symbolic_binding_pattern _, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc34: type = splice_block %X.ref.loc34 [concrete = constants.%X.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %X.ref.loc34: type = name_ref X, file.%X.decl.loc3 [concrete = constants.%X.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %_.loc34: %X.type = symbolic_binding _, 0 [symbolic = %_.loc6_6.1 (constants.%_)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl.loc35: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %U.patt: %pattern_type.b4b = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %u.param_patt: @G.%pattern_type (%pattern_type.b36) = value_param_pattern [concrete]
// CHECK:STDOUT:     %u.patt: @G.%pattern_type (%pattern_type.b36) = at_binding_pattern u, %u.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc35_10: type = splice_block %X.ref.loc35 [concrete = constants.%X.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %X.ref.loc35: type = name_ref X, file.%X.decl.loc3 [concrete = constants.%X.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc35: %X.type = symbolic_binding U, 0 [symbolic = %U.loc7_6.1 (constants.%U)]
// CHECK:STDOUT:     %u.param.loc35: @G.%U.binding.as_type (%U.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc35_23.1: type = splice_block %.loc35_23.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:       %U.ref.loc35: %X.type = name_ref U, %U.loc35 [symbolic = %U.loc7_6.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type.loc35: type = facet_access_type %U.ref.loc35 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       %.loc35_23.2: type = converted %U.ref.loc35, %U.as_type.loc35 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u.loc35: @G.%U.binding.as_type (%U.binding.as_type) = value_binding u, %u.param.loc35
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.Y.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc36: type = name_ref C, file.%C.decl.loc9 [concrete = constants.%C]
// CHECK:STDOUT:     %Y.ref.loc36: type = name_ref Y, file.%Y.decl.loc14 [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.X.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc37: type = name_ref C, file.%C.decl.loc9 [concrete = constants.%C]
// CHECK:STDOUT:     %X.ref.loc37: type = name_ref X, file.%X.decl.loc3 [concrete = constants.%X.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.Y.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc38: type = name_ref C, file.%C.decl.loc9 [concrete = constants.%C]
// CHECK:STDOUT:     %Y.ref.loc38: type = name_ref Y, file.%Y.decl.loc14 [concrete = constants.%Y.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @X {
// CHECK:STDOUT:   %Self: %X.type = symbolic_binding Self, 0 [symbolic = constants.%Self.e52]
// CHECK:STDOUT:   %X.WithSelf.decl = interface_with_self_decl @X [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = constants.%Self.162]
// CHECK:STDOUT:   %Y.WithSelf.decl = interface_with_self_decl @Y [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.X.impl: %C.ref.loc12 as %X.ref.loc12 {
// CHECK:STDOUT:   %X.impl_witness_table = impl_witness_table (), @C.as.X.impl [concrete]
// CHECK:STDOUT:   %X.impl_witness: <witness> = impl_witness %X.impl_witness_table [concrete = constants.%X.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Y.impl: %C.ref.loc36 as %Y.ref.loc36 {
// CHECK:STDOUT:   %Y.impl_witness_table = impl_witness_table (), @C.as.Y.impl [concrete]
// CHECK:STDOUT:   %Y.impl_witness: <witness> = impl_witness %Y.impl_witness_table [concrete = constants.%Y.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %Y.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%_.loc6_6.2: %X.type) {
// CHECK:STDOUT:   %_.loc6_6.1: %X.type = symbolic_binding _, 0 [symbolic = %_.loc6_6.1 (constants.%_)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%U.loc7_6.2: %X.type) {
// CHECK:STDOUT:   %U.loc7_6.1: %X.type = symbolic_binding U, 0 [symbolic = %U.loc7_6.1 (constants.%U)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 0, %U.loc7_6.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.b36)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param.loc35: @G.%U.binding.as_type (%U.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl.loc6 [concrete = constants.%F]
// CHECK:STDOUT:   %C.ref.loc27: type = name_ref C, file.%C.decl.loc9 [concrete = constants.%C]
// CHECK:STDOUT:   %X.facet.loc27: %X.type = facet_value %C.ref.loc27, (constants.%X.impl_witness) [concrete = constants.%X.facet]
// CHECK:STDOUT:   %.loc27: %X.type = converted %C.ref.loc27, %X.facet.loc27 [concrete = constants.%X.facet]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%X.facet) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl.loc7 [concrete = constants.%G]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %X.facet.loc28: %X.type = facet_value constants.%C, (constants.%X.impl_witness) [concrete = constants.%X.facet]
// CHECK:STDOUT:   %.loc28: %X.type = converted constants.%C, %X.facet.loc28 [concrete = constants.%X.facet]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%X.facet) [concrete = constants.%G.specific_fn]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.specific_fn(%c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%_) {
// CHECK:STDOUT:   %_.loc6_6.1 => constants.%_
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%U) {
// CHECK:STDOUT:   %U.loc7_6.1 => constants.%U
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.WithSelf(constants.%Self.e52) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%X.facet) {
// CHECK:STDOUT:   %_.loc6_6.1 => constants.%X.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%X.facet) {
// CHECK:STDOUT:   %U.loc7_6.1 => constants.%X.facet
// CHECK:STDOUT:   %U.binding.as_type => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7c7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%Self.162) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.WithSelf(constants.%X.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%Y.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_impl_in_interface_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.08c: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.705: %I.WithSelf.F.type.08c = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Self) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table, @C.as.I.impl(%Self) [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.5dd: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.f5e: %I.WithSelf.F.type.5dd = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.08c) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.705)] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .F = @I.WithSelf.%assoc0
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Self) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:   %I.impl_witness.loc11_16.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(%Self) [symbolic = %I.impl_witness.loc11_16.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref.loc11 as %I.ref.loc11 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (<error>), @C.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc11_16.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(constants.%Self) [symbolic = %I.impl_witness.loc11_16.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = <poisoned>
// CHECK:STDOUT:     witness = %I.impl_witness.loc11_16.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(@I.%Self: %I.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Self) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.decl: type = class_decl @C [symbolic = @I.WithSelf.F.%C (constants.%C)] {} {}
// CHECK:STDOUT:     impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:       %C.ref.loc11: type = name_ref C, @I.WithSelf.F.%C.decl [symbolic = %C (constants.%C)]
// CHECK:STDOUT:       %I.ref.loc11: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:       %C.ref.loc20: type = name_ref C, @I.WithSelf.F.%C.decl [symbolic = %C (constants.%C)]
// CHECK:STDOUT:       %I.ref.loc20: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.08c
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.705
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %C => constants.%C
// CHECK:STDOUT:   %I.impl_witness.loc11_16.2 => constants.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.5dd
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.f5e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_impl_in_interface_definition_with_associated.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%U [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.08c: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.705: %I.WithSelf.F.type.08c = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Self) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %I.WithSelf.F.type.29d: type = fn_type @I.WithSelf.F, @I.WithSelf(%.Self) [symbolic_self]
// CHECK:STDOUT:   %I.WithSelf.F.8ec: %I.WithSelf.F.type.29d = struct_value () [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %C> [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table, @C.as.I.impl(%Self) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I_where.type [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.000: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.a8c: %I.WithSelf.F.type.000 = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%U [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.08c) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.705)] {} {}
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, %I.WithSelf.F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = @I.WithSelf.%assoc1
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   witness = (@I.WithSelf.%U, @I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Self) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where constants.%impl.elem0 = %C> [symbolic = %I_where.type (constants.%I_where.type)]
// CHECK:STDOUT:   %I.impl_witness.loc12_29.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(%Self) [symbolic = %I.impl_witness.loc12_29.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I_where.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref.loc12_10 as %.loc12_17 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, <error>), @C.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc12_29.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(constants.%Self) [symbolic = %I.impl_witness.loc12_29.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%C [symbolic = %C (constants.%C)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = <poisoned>
// CHECK:STDOUT:     witness = %I.impl_witness.loc12_29.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(@I.%Self: %I.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Self) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.decl: type = class_decl @C [symbolic = @I.WithSelf.F.%C (constants.%C)] {} {}
// CHECK:STDOUT:     impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:       %C.ref.loc12_10: type = name_ref C, @I.WithSelf.F.%C.decl [symbolic = %C (constants.%C)]
// CHECK:STDOUT:       %I.ref.loc12: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self.2: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref.loc12: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.as_type.loc12: type = facet_access_type %.Self.ref.loc12 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc12_23: type = converted %.Self.ref.loc12, %.Self.as_type.loc12 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %U.ref.loc12: %I.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc12: type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %C.ref.loc12_28: type = name_ref C, @I.WithSelf.F.%C.decl [symbolic = %C (constants.%C)]
// CHECK:STDOUT:       %.loc12_17: type = where_expr %.Self.2 [symbolic = %I_where.type (constants.%I_where.type)] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc12, %C.ref.loc12_28
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:       %C.ref.loc21_10: type = name_ref C, @I.WithSelf.F.%C.decl [symbolic = %C (constants.%C)]
// CHECK:STDOUT:       %I.ref.loc21: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self.1: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref.loc21: %I.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.as_type.loc21: type = facet_access_type %.Self.ref.loc21 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc21_23: type = converted %.Self.ref.loc21, %.Self.as_type.loc21 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %U.ref.loc21: %I.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc21: type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %C.ref.loc21_28: type = name_ref C, @I.WithSelf.F.%C.decl [symbolic = %C (constants.%C)]
// CHECK:STDOUT:       %.loc21_17: type = where_expr %.Self.1 [symbolic = %I_where.type (constants.%I_where.type)] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc21, %C.ref.loc21_28
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.08c
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.705
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%.Self
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.29d
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.8ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %C => constants.%C
// CHECK:STDOUT:   %I_where.type => constants.%I_where.type
// CHECK:STDOUT:   %I.impl_witness.loc12_29.2 => constants.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.000
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.a8c
// CHECK:STDOUT: }
// CHECK:STDOUT:
