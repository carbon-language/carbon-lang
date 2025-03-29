// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/min_prelude/forward_declarations.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/min_prelude/forward_declarations.carbon

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

// --- fail_todo_associated_const_compound_member_access.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
}
impl {} as I where .T = ();

// CHECK:STDERR: fail_todo_associated_const_compound_member_access.carbon:[[@LINE+7]]:8: error: cannot implicitly convert non-type value of type `{}` into type implementing `I` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: let x: {}.(I.T) = ();
// CHECK:STDERR:        ^~~~~~~~
// CHECK:STDERR: fail_todo_associated_const_compound_member_access.carbon:[[@LINE+4]]:8: note: type `{}` does not implement interface `Core.ImplicitAs(I)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let x: {}.(I.T) = ();
// CHECK:STDERR:        ^~~~~~~~
// CHECK:STDERR:
let x: {}.(I.T) = ();

impl {} as I where .T = () {}

// --- fail_todo_associated_const_of_facet.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
}
impl {} as I where .T = ();

// CHECK:STDERR: fail_todo_associated_const_of_facet.carbon:[[@LINE+7]]:9: error: cannot convert non-type value of type `{}` into type implementing `I` with `as` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: let x: ({} as I).T = ();
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR: fail_todo_associated_const_of_facet.carbon:[[@LINE+4]]:9: note: type `{}` does not implement interface `Core.As(I)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let x: ({} as I).T = ();
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR:
let x: ({} as I).T = ();

impl {} as I where .T = () {}

// --- fail_todo_unset_associated_const.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
}
impl {} as I;

// TODO: Should fail because the associated constant is not set,
// not because it can't find the impl.

// CHECK:STDERR: fail_todo_unset_associated_const.carbon:[[@LINE+14]]:9: error: cannot convert non-type value of type `{}` into type implementing `I` with `as` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: let x: ({} as I).T = ();
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR: fail_todo_unset_associated_const.carbon:[[@LINE+11]]:9: note: type `{}` does not implement interface `Core.As(I)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let x: ({} as I).T = ();
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_unset_associated_const.carbon:[[@LINE-12]]:12: error: associated constant T not given a value in impl of interface I [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: impl {} as I;
// CHECK:STDERR:            ^
// CHECK:STDERR: fail_todo_unset_associated_const.carbon:[[@LINE-17]]:7: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR:   let T:! type;
// CHECK:STDERR:       ^~~~~~~~
// CHECK:STDERR:
let x: ({} as I).T = ();

impl {} as I {}

// --- fail_associated_const_before_interface_definition.carbon
library "[[@TEST_NAME]]";

interface I;
class C;
// CHECK:STDERR: fail_associated_const_before_interface_definition.carbon:[[@LINE+7]]:20: error: member access into object of incomplete type `I` [IncompleteTypeInMemberAccess]
// CHECK:STDERR: impl {} as I where .T = C;
// CHECK:STDERR:                    ^~
// CHECK:STDERR: fail_associated_const_before_interface_definition.carbon:[[@LINE-5]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
impl {} as I where .T = C;

interface I {}
class C {}
// CHECK:STDERR: fail_associated_const_before_interface_definition.carbon:[[@LINE+4]]:20: error: member name `T` not found in `I` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: impl {} as I where .T = C {}
// CHECK:STDERR:                    ^~
// CHECK:STDERR:
impl {} as I where .T = C {}

// --- associated_const_of_parameterized.carbon
library "[[@TEST_NAME]]";

interface I(U:! type) {
  let T:! type;
}
class C;
impl {} as I(C) where .T = C;

class C {}
impl {} as I(C) where .T = C {}

// --- fail_todo_find_incomplete_impl.carbon
library "[[@TEST_NAME]]";

interface I;
impl {} as I;
class C(T:! I);

// CHECK:STDERR: fail_todo_find_incomplete_impl.carbon:[[@LINE+10]]:9: error: cannot implicitly convert non-type value of type `{}` into type implementing `I` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: fn F(x: C({}));
// CHECK:STDERR:         ^~~~~
// CHECK:STDERR: fail_todo_find_incomplete_impl.carbon:[[@LINE+7]]:9: note: type `{}` does not implement interface `Core.ImplicitAs(I)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn F(x: C({}));
// CHECK:STDERR:         ^~~~~
// CHECK:STDERR: fail_todo_find_incomplete_impl.carbon:[[@LINE-8]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(T:! I);
// CHECK:STDERR:         ^
// CHECK:STDERR:
fn F(x: C({}));

interface I {}
impl {} as I {}
class C(T:! I) {}

// CHECK:STDERR: fail_todo_find_incomplete_impl.carbon:[[@LINE+10]]:9: error: cannot implicitly convert non-type value of type `{}` into type implementing `I` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: fn F(x: C({})) {}
// CHECK:STDERR:         ^~~~~
// CHECK:STDERR: fail_todo_find_incomplete_impl.carbon:[[@LINE+7]]:9: note: type `{}` does not implement interface `Core.ImplicitAs(I)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn F(x: C({})) {}
// CHECK:STDERR:         ^~~~~
// CHECK:STDERR: fail_todo_find_incomplete_impl.carbon:[[@LINE-8]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(T:! I) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
fn F(x: C({})) {}

// --- fail_two_interfaces.carbon
library "[[@TEST_NAME]]";

interface I;
interface J;

// CHECK:STDERR: fail_two_interfaces.carbon:[[@LINE+8]]:1: error: impl as 2 interfaces, expected 1 [ImplOfNotOneInterface]
// CHECK:STDERR: impl {} as I & J;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_interfaces.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl {} as I & J;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl {} as I & J;

interface I {}
interface J {}

// --- fail_todo_never_assigned_associated_const.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
  let U:! type;
}

// TODO: Should accept this, once we support setting
// associated constants in the body of the impl definition.

// CHECK:STDERR: fail_todo_never_assigned_associated_const.carbon:[[@LINE+7]]:12: error: associated constant U not given a value in impl of interface I [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: impl {} as I where .T = ();
// CHECK:STDERR:            ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_never_assigned_associated_const.carbon:[[@LINE-9]]:7: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR:   let U:! type;
// CHECK:STDERR:       ^~~~~~~~
// CHECK:STDERR:
impl {} as I where .T = ();

// TODO: This should fail instead, unless there is a
// `where U = ...` declaration in the body of the impl.
impl {} as I where .T = () {}

// CHECK:STDOUT: --- empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc4_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc4_7.2: type = converted %.loc4_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc4: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %I.decl.loc6: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc7_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc7_7.2: type = converted %.loc7_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc7: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc4_7.2 as %I.ref.loc4 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%G.decl) [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type.ccd: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.600: %G.type.ccd = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%G.decl [concrete]
// CHECK:STDOUT:   %G.type.d6f: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.d07: %G.type.d6f = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %empty_struct_type, (%impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc4_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc4_7.2: type = converted %.loc4_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc4: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%G.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %I.decl.loc6: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc9_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc9_7.2: type = converted %.loc9_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc9: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %G.decl: %G.type.ccd = fn_decl @G.1 [concrete = constants.%G.600] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %G.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc4_7.2 as %I.ref.loc4 {
// CHECK:STDOUT:   %G.decl: %G.type.d6f = fn_decl @G.2 [concrete = constants.%G.d07] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- combine.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.25f: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type %BitAnd.type [concrete]
// CHECK:STDOUT:   %assoc0: %BitAnd.assoc_type = assoc_entity element0, imports.%Core.import_ref.a93 [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.25f [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness.db8: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.f99: type = fn_type @Op.2, @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op.05a: %Op.type.f99 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %impl_witness.3ea: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl.f92(type) [concrete]
// CHECK:STDOUT:   %impl_witness.b81: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.eb8: type = fn_type @Op.2, @impl.f92(type) [concrete]
// CHECK:STDOUT:   %Op.444: %Op.type.eb8 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness type [concrete]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value type, (%impl_witness.3ea) [concrete]
// CHECK:STDOUT:   %.2ac: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %I.type, %Op.444 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.444, @Op.2(type) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %I.type, %Op.specific_fn [concrete]
// CHECK:STDOUT:   %impl_witness.1bc: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//prelude, inst101 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.08d: %BitAnd.assoc_type = import_ref Core//prelude, loc18_41, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.040: %BitAnd.type = import_ref Core//prelude, inst101 [no loc], loaded [symbolic = constants.%Self.25f]
// CHECK:STDOUT:   %Core.import_ref.51c: <witness> = import_ref Core//prelude, loc21_36, loaded [symbolic = @impl.f92.%impl_witness (constants.%impl_witness.b81)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc21_24, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.9c1: type = import_ref Core//prelude, loc21_29, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.1e6: @impl.f92.%Op.type (%Op.type.f99) = import_ref Core//prelude, loc22_42, loaded [symbolic = @impl.f92.%Op (constants.%Op.05a)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl.loc3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl.loc3: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %.loc4_14.1: type = value_of_initializer @impl.8dc.%type.and.loc4 [concrete = constants.%I.type]
// CHECK:STDOUT:   %.loc4_14.2: type = converted @impl.8dc.%type.and.loc4, %.loc4_14.1 [concrete = constants.%I.type]
// CHECK:STDOUT:   impl_decl @impl.8dc [concrete] {} {
// CHECK:STDOUT:     %.loc4_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc4_7.2: type = converted %.loc4_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc4_12: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:     %I.ref.loc4_16: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:     %impl.elem0.loc4: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:     %bound_method.loc4_14.1: <bound method> = bound_method %I.ref.loc4_12, %impl.elem0.loc4 [concrete = constants.%Op.bound]
// CHECK:STDOUT:     %specific_fn.loc4: <specific function> = specific_function %impl.elem0.loc4, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc4_14.2: <bound method> = bound_method %I.ref.loc4_12, %specific_fn.loc4 [concrete = constants.%bound_method]
// CHECK:STDOUT:     %type.and.loc4: init type = call %bound_method.loc4_14.2(%I.ref.loc4_12, %I.ref.loc4_16) [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete = constants.%impl_witness.1bc]
// CHECK:STDOUT:   %I.decl.loc6: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %.loc7_14.1: type = value_of_initializer @impl.8dc.%type.and.loc7 [concrete = constants.%I.type]
// CHECK:STDOUT:   %.loc7_14.2: type = converted @impl.8dc.%type.and.loc7, %.loc7_14.1 [concrete = constants.%I.type]
// CHECK:STDOUT:   impl_decl @impl.8dc [concrete] {} {
// CHECK:STDOUT:     %.loc7_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc7_7.2: type = converted %.loc7_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc7_12: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:     %I.ref.loc7_16: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:     %impl.elem0.loc7: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:     %bound_method.loc7_14.1: <bound method> = bound_method %I.ref.loc7_12, %impl.elem0.loc7 [concrete = constants.%Op.bound]
// CHECK:STDOUT:     %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc7_14.2: <bound method> = bound_method %I.ref.loc7_12, %specific_fn.loc7 [concrete = constants.%bound_method]
// CHECK:STDOUT:     %type.and.loc7: init type = call %bound_method.loc7_14.2(%I.ref.loc7_12, %I.ref.loc7_16) [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.ad0
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.08d
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.f92(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl.f92(%T) [symbolic = %impl_witness (constants.%impl_witness.db8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl.f92(%T) [symbolic = %Op.type (constants.%Op.type.f99)]
// CHECK:STDOUT:   %Op: @impl.f92.%Op.type (%Op.type.f99) = struct_value () [symbolic = %Op (constants.%Op.05a)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.f92.%T (%T) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.9c1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.51c
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8dc: %.loc4_7.2 as file.%.loc4_14.2 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.040: %BitAnd.type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.25f)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type (%Self.as_type)](%other.param_patt: @Op.1.%Self.as_type (%Self.as_type)) -> @Op.1.%Self.as_type (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Core.import_ref.5ab3ec.2: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%T (%T)](%other.param_patt: @Op.2.%T (%T)) -> @Op.2.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.25f) {
// CHECK:STDOUT:   %Self => constants.%Self.25f
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.db8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:   %T.patt => type
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.3ea
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.eb8
// CHECK:STDOUT:   %Op => constants.%Op.444
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_tuple.type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc6_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc6: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.1: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc6: %I.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc6: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc6: type = facet_access_type %.Self.ref.loc6 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc6_20: type = converted %.Self.ref.loc6, %.Self.as_type.loc6 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc6: <witness> = facet_access_witness %.Self.ref.loc6, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc6: type = impl_witness_access %.Self.as_wit.iface0.loc6, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc6_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_26.2: type = converted %.loc6_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc6_14: type = where_expr %.Self.1 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc6, %.loc6_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_tuple.type) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc8: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.2: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc8: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc8: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc8: type = facet_access_type %.Self.ref.loc8 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref.loc8, %.Self.as_type.loc8 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc8: <witness> = facet_access_witness %.Self.ref.loc8, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc8: type = impl_witness_access %.Self.as_wit.iface0.loc8, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc8_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_26.2: type = converted %.loc8_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc8, %.loc8_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc6_7.2 as %.loc6_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_associated_const_compound_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0.c7e: %I.assoc_type = assoc_entity element0, @I.%T [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %impl_witness.2a6: <witness> = impl_witness (%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.837: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0.02f: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.c42: type = facet_type <@ImplicitAs, @ImplicitAs(%I.type)> [concrete]
// CHECK:STDOUT:   %Self.acc: %ImplicitAs.type.c42 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.7ef: type = fn_type @Convert, @ImplicitAs(%I.type) [concrete]
// CHECK:STDOUT:   %Convert.c77: %Convert.type.7ef = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.167: type = assoc_entity_type %ImplicitAs.type.c42 [concrete]
// CHECK:STDOUT:   %assoc0.0e0: %ImplicitAs.assoc_type.167 = assoc_entity element0, imports.%Core.import_ref.1c7 [concrete]
// CHECK:STDOUT:   %assoc0.43d: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness.b81: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op: %Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc13_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//prelude, inst66 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.630: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = import_ref Core//prelude, loc14_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.43d)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//prelude, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc13_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//prelude, inst66 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//prelude, loc14_35, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//prelude, inst101 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.3bf = import_ref Core//prelude, loc18_41, unloaded
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.f80 = import_ref Core//prelude, loc21_36, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.3: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc21_24, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.9c1: type = import_ref Core//prelude, loc21_29, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.bd4 = import_ref Core//prelude, loc22_42, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.4: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.280 [concrete] {} {
// CHECK:STDOUT:     %.loc6_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc6: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.1: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc6: %I.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc6: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0.c7e]
// CHECK:STDOUT:     %.Self.as_type.loc6: type = facet_access_type %.Self.ref.loc6 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc6_20: type = converted %.Self.ref.loc6, %.Self.as_type.loc6 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc6: <witness> = facet_access_witness %.Self.ref.loc6, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc6: type = impl_witness_access %.Self.as_wit.iface0.loc6, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc6_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_26.2: type = converted %.loc6_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc6_14: type = where_expr %.Self.1 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc6, %.loc6_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_tuple.type) [concrete = constants.%impl_witness.2a6]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: <error> = binding_pattern x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %.loc15_9: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %I.ref: type = name_ref I, %I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.ref: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0.c7e]
// CHECK:STDOUT:     %.loc15_10: %I.type = converted %.loc15_9, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: <error> = bind_name x, <error>
// CHECK:STDOUT:   impl_decl @impl.280 [concrete] {} {
// CHECK:STDOUT:     %.loc17_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc17_7.2: type = converted %.loc17_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc17: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.2: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc17: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc17: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0.c7e]
// CHECK:STDOUT:     %.Self.as_type.loc17: type = facet_access_type %.Self.ref.loc17 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc17_20: type = converted %.Self.ref.loc17, %.Self.as_type.loc17 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc17: <witness> = facet_access_witness %.Self.ref.loc17, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc17: type = impl_witness_access %.Self.as_wit.iface0.loc17, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc17_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc17_26.2: type = converted %.loc17_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc17_14: type = where_expr %.Self.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc17, %.loc17_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete = constants.%assoc0.c7e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.837)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.02f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.630
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.ad0
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.3bf
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.280: %.loc6_7.2 as %.loc6_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.f92(imports.%Core.import_ref.5ab3ec.3: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl.f92(%T) [symbolic = %impl_witness (constants.%impl_witness.b81)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @impl.f92(%T) [symbolic = %Op.type (constants.%Op.type)]
// CHECK:STDOUT:   %Op: @impl.f92.%Op.type (%Op.type) = struct_value () [symbolic = %Op (constants.%Op)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.f92.%T (%T) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.9c1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.f80
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: @Convert.%ImplicitAs.type (%ImplicitAs.type.d62) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(imports.%Core.import_ref.5ab3ec.4: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.%T (%T)](%other.param_patt: @Op.%T (%T)) -> @Op.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.519) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%I.type) {
// CHECK:STDOUT:   %Dest => constants.%I.type
// CHECK:STDOUT:   %Dest.patt => constants.%I.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.c42
// CHECK:STDOUT:   %Self => constants.%Self.acc
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.7ef
// CHECK:STDOUT:   %Convert => constants.%Convert.c77
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.167
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.b81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_associated_const_of_facet.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0.c7e: %I.assoc_type = assoc_entity element0, @I.%T [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %impl_witness.2a6: <witness> = impl_witness (%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %As.type.eed: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.65a: %As.type.eed = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.843: type = fn_type @Convert, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.95f: %Convert.type.843 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.65a [symbolic]
// CHECK:STDOUT:   %As.assoc_type.a44: type = assoc_entity_type %As.type.eed [symbolic]
// CHECK:STDOUT:   %assoc0.ea3: %As.assoc_type.a44 = assoc_entity element0, imports.%Core.import_ref.708 [symbolic]
// CHECK:STDOUT:   %As.type.0da: type = facet_type <@As, @As(%I.type)> [concrete]
// CHECK:STDOUT:   %Self.744: %As.type.0da = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.542: type = fn_type @Convert, @As(%I.type) [concrete]
// CHECK:STDOUT:   %Convert.04e: %Convert.type.542 = struct_value () [concrete]
// CHECK:STDOUT:   %As.assoc_type.8f1: type = assoc_entity_type %As.type.0da [concrete]
// CHECK:STDOUT:   %assoc0.bb5: %As.assoc_type.8f1 = assoc_entity element0, imports.%Core.import_ref.708 [concrete]
// CHECK:STDOUT:   %assoc0.5bc: %As.assoc_type.a44 = assoc_entity element0, imports.%Core.import_ref.4e8 [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness.b81: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op: %Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc9_14, loaded [symbolic = @As.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.a7c = import_ref Core//prelude, inst25 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.713: @As.%As.assoc_type (%As.assoc_type.a44) = import_ref Core//prelude, loc10_35, loaded [symbolic = @As.%assoc0 (constants.%assoc0.5bc)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//prelude, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc9_14, loaded [symbolic = @As.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.996: @As.%As.type (%As.type.eed) = import_ref Core//prelude, inst25 [no loc], loaded [symbolic = @As.%Self (constants.%Self.65a)]
// CHECK:STDOUT:   %Core.import_ref.708: @As.%Convert.type (%Convert.type.843) = import_ref Core//prelude, loc10_35, loaded [symbolic = @As.%Convert (constants.%Convert.95f)]
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//prelude, inst101 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.3bf = import_ref Core//prelude, loc18_41, unloaded
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.f80 = import_ref Core//prelude, loc21_36, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.3: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc21_24, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.9c1: type = import_ref Core//prelude, loc21_29, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.bd4 = import_ref Core//prelude, loc22_42, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.4: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.280 [concrete] {} {
// CHECK:STDOUT:     %.loc6_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc6: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.1: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc6: %I.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc6: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0.c7e]
// CHECK:STDOUT:     %.Self.as_type.loc6: type = facet_access_type %.Self.ref.loc6 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc6_20: type = converted %.Self.ref.loc6, %.Self.as_type.loc6 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc6: <witness> = facet_access_witness %.Self.ref.loc6, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc6: type = impl_witness_access %.Self.as_wit.iface0.loc6, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc6_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_26.2: type = converted %.loc6_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc6_14: type = where_expr %.Self.1 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc6, %.loc6_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_tuple.type) [concrete = constants.%impl_witness.2a6]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: <error> = binding_pattern x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %.loc15_10: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %I.ref: type = name_ref I, %I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc15_12: %I.type = converted %.loc15_10, <error> [concrete = <error>]
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: <error> = bind_name x, <error>
// CHECK:STDOUT:   impl_decl @impl.280 [concrete] {} {
// CHECK:STDOUT:     %.loc17_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc17_7.2: type = converted %.loc17_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc17: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.2: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc17: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc17: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0.c7e]
// CHECK:STDOUT:     %.Self.as_type.loc17: type = facet_access_type %.Self.ref.loc17 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc17_20: type = converted %.Self.ref.loc17, %.Self.as_type.loc17 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc17: <witness> = facet_access_witness %.Self.ref.loc17, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc17: type = impl_witness_access %.Self.as_wit.iface0.loc17, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc17_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc17_26.2: type = converted %.loc17_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc17_14: type = where_expr %.Self.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc17, %.loc17_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete = constants.%assoc0.c7e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.eed)]
// CHECK:STDOUT:   %Self: @As.%As.type (%As.type.eed) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.65a)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.843)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.843) = struct_value () [symbolic = %Convert (constants.%Convert.95f)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As.%As.type (%As.type.eed) [symbolic = %As.assoc_type (constants.%As.assoc_type.a44)]
// CHECK:STDOUT:   %assoc0: @As.%As.assoc_type (%As.assoc_type.a44) = assoc_entity element0, imports.%Core.import_ref.708 [symbolic = %assoc0 (constants.%assoc0.ea3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.a7c
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.713
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.ad0
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.3bf
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.280: %.loc6_7.2 as %.loc6_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.f92(imports.%Core.import_ref.5ab3ec.3: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl.f92(%T) [symbolic = %impl_witness (constants.%impl_witness.b81)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @impl.f92(%T) [symbolic = %Op.type (constants.%Op.type)]
// CHECK:STDOUT:   %Op: @impl.f92.%Op.type (%Op.type) = struct_value () [symbolic = %Op (constants.%Op)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.f92.%T (%T) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.9c1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.f80
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.996: @As.%As.type (%As.type.eed)) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.eed)]
// CHECK:STDOUT:   %Self: @Convert.%As.type (%As.type.eed) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.65a)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(imports.%Core.import_ref.5ab3ec.4: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.%T (%T)](%other.param_patt: @Op.%T (%T)) -> @Op.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.65a) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.eed
// CHECK:STDOUT:   %Self => constants.%Self.65a
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%I.type) {
// CHECK:STDOUT:   %Dest => constants.%I.type
// CHECK:STDOUT:   %Dest.patt => constants.%I.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.0da
// CHECK:STDOUT:   %Self => constants.%Self.744
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.542
// CHECK:STDOUT:   %Convert => constants.%Convert.04e
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.8f1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.bb5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.b81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_unset_associated_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0.c7e: %I.assoc_type = assoc_entity element0, @I.%T [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %impl_witness.85b: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %As.type.eed: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.65a: %As.type.eed = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.843: type = fn_type @Convert, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.95f: %Convert.type.843 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.65a [symbolic]
// CHECK:STDOUT:   %As.assoc_type.a44: type = assoc_entity_type %As.type.eed [symbolic]
// CHECK:STDOUT:   %assoc0.ea3: %As.assoc_type.a44 = assoc_entity element0, imports.%Core.import_ref.708 [symbolic]
// CHECK:STDOUT:   %As.type.0da: type = facet_type <@As, @As(%I.type)> [concrete]
// CHECK:STDOUT:   %Self.744: %As.type.0da = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.542: type = fn_type @Convert, @As(%I.type) [concrete]
// CHECK:STDOUT:   %Convert.04e: %Convert.type.542 = struct_value () [concrete]
// CHECK:STDOUT:   %As.assoc_type.8f1: type = assoc_entity_type %As.type.0da [concrete]
// CHECK:STDOUT:   %assoc0.bb5: %As.assoc_type.8f1 = assoc_entity element0, imports.%Core.import_ref.708 [concrete]
// CHECK:STDOUT:   %assoc0.5bc: %As.assoc_type.a44 = assoc_entity element0, imports.%Core.import_ref.4e8 [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness.b81: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op: %Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc9_14, loaded [symbolic = @As.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.a7c = import_ref Core//prelude, inst25 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.713: @As.%As.assoc_type (%As.assoc_type.a44) = import_ref Core//prelude, loc10_35, loaded [symbolic = @As.%assoc0 (constants.%assoc0.5bc)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//prelude, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc9_14, loaded [symbolic = @As.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.996: @As.%As.type (%As.type.eed) = import_ref Core//prelude, inst25 [no loc], loaded [symbolic = @As.%Self (constants.%Self.65a)]
// CHECK:STDOUT:   %Core.import_ref.708: @As.%Convert.type (%Convert.type.843) = import_ref Core//prelude, loc10_35, loaded [symbolic = @As.%Convert (constants.%Convert.95f)]
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//prelude, inst101 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.3bf = import_ref Core//prelude, loc18_41, unloaded
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.f80 = import_ref Core//prelude, loc21_36, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.3: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc21_24, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.9c1: type = import_ref Core//prelude, loc21_29, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.bd4 = import_ref Core//prelude, loc22_42, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.4: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.8dc [concrete] {} {
// CHECK:STDOUT:     %.loc6_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc6: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness.85b]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: <error> = binding_pattern x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %.loc25_10: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %I.ref: type = name_ref I, %I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc25_12: %I.type = converted %.loc25_10, <error> [concrete = <error>]
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: <error> = bind_name x, <error>
// CHECK:STDOUT:   impl_decl @impl.8dc [concrete] {} {
// CHECK:STDOUT:     %.loc27_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc27_7.2: type = converted %.loc27_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc27: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete = constants.%assoc0.c7e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.eed)]
// CHECK:STDOUT:   %Self: @As.%As.type (%As.type.eed) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.65a)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.843)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.843) = struct_value () [symbolic = %Convert (constants.%Convert.95f)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As.%As.type (%As.type.eed) [symbolic = %As.assoc_type (constants.%As.assoc_type.a44)]
// CHECK:STDOUT:   %assoc0: @As.%As.assoc_type (%As.assoc_type.a44) = assoc_entity element0, imports.%Core.import_ref.708 [symbolic = %assoc0 (constants.%assoc0.ea3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.a7c
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.713
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.ad0
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.3bf
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8dc: %.loc6_7.2 as %I.ref.loc6 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.f92(imports.%Core.import_ref.5ab3ec.3: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl.f92(%T) [symbolic = %impl_witness (constants.%impl_witness.b81)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @impl.f92(%T) [symbolic = %Op.type (constants.%Op.type)]
// CHECK:STDOUT:   %Op: @impl.f92.%Op.type (%Op.type) = struct_value () [symbolic = %Op (constants.%Op)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.f92.%T (%T) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.9c1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.f80
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.996: @As.%As.type (%As.type.eed)) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.eed)]
// CHECK:STDOUT:   %Self: @Convert.%As.type (%As.type.eed) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.65a)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(imports.%Core.import_ref.5ab3ec.4: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.%T (%T)](%other.param_patt: @Op.%T (%T)) -> @Op.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc25: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.65a) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.eed
// CHECK:STDOUT:   %Self => constants.%Self.65a
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%I.type) {
// CHECK:STDOUT:   %Dest => constants.%I.type
// CHECK:STDOUT:   %Dest.patt => constants.%I.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.0da
// CHECK:STDOUT:   %Self => constants.%Self.744
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.542
// CHECK:STDOUT:   %Convert => constants.%Convert.04e
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.8f1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.bb5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.b81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_associated_const_before_interface_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl.loc3
// CHECK:STDOUT:     .C = %C.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl.loc3: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %C.decl.loc4: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc12_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc12_7.2: type = converted %.loc12_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc12: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.1: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc12: %I.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %C.ref.loc12: type = name_ref C, file.%C.decl.loc4 [concrete = constants.%C]
// CHECK:STDOUT:     %.loc12_14: type = where_expr %.Self.1 [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite <error>, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc14: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %C.decl.loc15: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc20_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc20_7.2: type = converted %.loc20_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc20: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.2: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc20: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %C.ref.loc20: type = name_ref C, file.%C.decl.loc4 [concrete = constants.%C]
// CHECK:STDOUT:     %.loc20_14: type = where_expr %.Self.2 [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc12_7.2 as %.loc12_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- associated_const_of_parameterized.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.325: type = facet_type <@I, @I(%U)> [symbolic]
// CHECK:STDOUT:   %Self.209: %I.type.325 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.955: type = assoc_entity_type %I.type.325 [symbolic]
// CHECK:STDOUT:   %assoc0.807: %I.assoc_type.955 = assoc_entity element0, @I.%T [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %I.type.e7a: type = facet_type <@I, @I(%C)> [concrete]
// CHECK:STDOUT:   %.Self: %I.type.e7a = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Self.b1d: %I.type.e7a = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.aa7: type = assoc_entity_type %I.type.e7a [concrete]
// CHECK:STDOUT:   %assoc0.3ba: %I.assoc_type.aa7 = assoc_entity element0, @I.%T [concrete]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %I.facet: %I.type.e7a = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I, @I(%C) where %impl.elem0 = %C> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%C) [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %U.patt.loc3_13.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc3_13.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.loc3_13.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc3_13.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc6: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc7_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc7_7.2: type = converted %.loc7_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc7: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %C.ref.loc7_14: type = name_ref C, file.%C.decl.loc6 [concrete = constants.%C]
// CHECK:STDOUT:     %I.type.loc7: type = facet_type <@I, @I(constants.%C)> [concrete = constants.%I.type.e7a]
// CHECK:STDOUT:     %.Self.1: %I.type.e7a = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc7: %I.type.e7a = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.loc7_23.1: %I.assoc_type.aa7 = specific_constant @T.%assoc0, @I(constants.%C) [concrete = constants.%assoc0.3ba]
// CHECK:STDOUT:     %T.ref.loc7: %I.assoc_type.aa7 = name_ref T, %.loc7_23.1 [concrete = constants.%assoc0.3ba]
// CHECK:STDOUT:     %.Self.as_type.loc7: type = facet_access_type %.Self.ref.loc7 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc7_23.2: type = converted %.Self.ref.loc7, %.Self.as_type.loc7 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc7: <witness> = facet_access_witness %.Self.ref.loc7, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc7: type = impl_witness_access %.Self.as_wit.iface0.loc7, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %C.ref.loc7_28: type = name_ref C, file.%C.decl.loc6 [concrete = constants.%C]
// CHECK:STDOUT:     %.loc7_17: type = where_expr %.Self.1 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc7, %C.ref.loc7_28
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%C) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %C.decl.loc9: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc10_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc10_7.2: type = converted %.loc10_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc10: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %C.ref.loc10_14: type = name_ref C, file.%C.decl.loc6 [concrete = constants.%C]
// CHECK:STDOUT:     %I.type.loc10: type = facet_type <@I, @I(constants.%C)> [concrete = constants.%I.type.e7a]
// CHECK:STDOUT:     %.Self.2: %I.type.e7a = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc10: %I.type.e7a = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.loc10_23.1: %I.assoc_type.aa7 = specific_constant @T.%assoc0, @I(constants.%C) [concrete = constants.%assoc0.3ba]
// CHECK:STDOUT:     %T.ref.loc10: %I.assoc_type.aa7 = name_ref T, %.loc10_23.1 [concrete = constants.%assoc0.3ba]
// CHECK:STDOUT:     %.Self.as_type.loc10: type = facet_access_type %.Self.ref.loc10 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc10_23.2: type = converted %.Self.ref.loc10, %.Self.as_type.loc10 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc10: <witness> = facet_access_witness %.Self.ref.loc10, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc10: type = impl_witness_access %.Self.as_wit.iface0.loc10, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %C.ref.loc10_28: type = name_ref C, file.%C.decl.loc6 [concrete = constants.%C]
// CHECK:STDOUT:     %.loc10_17: type = where_expr %.Self.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10, %C.ref.loc10_28
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%U.loc3_13.1: type) {
// CHECK:STDOUT:   %U.loc3_13.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc3_13.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc3_13.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc3_13.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%U.loc3_13.2)> [symbolic = %I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %Self.2: @I.%I.type (%I.type.325) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.209)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.325) [symbolic = %I.assoc_type (constants.%I.assoc_type.955)]
// CHECK:STDOUT:   %assoc0: @I.%I.assoc_type (%I.assoc_type.955) = assoc_entity element0, %T [symbolic = %assoc0 (constants.%assoc0.807)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.325) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.209)]
// CHECK:STDOUT:     %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:       %assoc0: @I.%I.assoc_type (%I.assoc_type.955) = assoc_entity element0, @I.%T [symbolic = @I.%assoc0 (constants.%assoc0.807)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = @T.%assoc0
// CHECK:STDOUT:     witness = (%T)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@I.%U.loc3_13.1: type, @I.%Self.1: @I.%I.type (%I.type.325)) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc7_7.2 as %.loc7_17 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%U) {
// CHECK:STDOUT:   %U.loc3_13.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc3_13.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%U, constants.%Self.209) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%U.loc3_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%C) {
// CHECK:STDOUT:   %U.loc3_13.2 => constants.%C
// CHECK:STDOUT:   %U.patt.loc3_13.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.e7a
// CHECK:STDOUT:   %Self.2 => constants.%Self.b1d
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.aa7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.3ba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%C, constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_find_incomplete_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %impl_witness.1bc: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %T.826: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.3ad: %I.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.826) [symbolic]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.837: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0.02f: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.c42: type = facet_type <@ImplicitAs, @ImplicitAs(%I.type)> [concrete]
// CHECK:STDOUT:   %Self.acc: %ImplicitAs.type.c42 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.7ef: type = fn_type @Convert, @ImplicitAs(%I.type) [concrete]
// CHECK:STDOUT:   %Convert.c77: %Convert.type.7ef = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.167: type = assoc_entity_type %ImplicitAs.type.c42 [concrete]
// CHECK:STDOUT:   %assoc0.0e0: %ImplicitAs.assoc_type.167 = assoc_entity element0, imports.%Core.import_ref.1c7 [concrete]
// CHECK:STDOUT:   %assoc0.43d: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness.b81: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl.f92(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @impl.f92(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op: %Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %F.type.b25846.1: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.c41931.1: %F.type.b25846.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type.b25846.2: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41931.2: %F.type.b25846.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc13_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//prelude, inst66 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.630: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = import_ref Core//prelude, loc14_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.43d)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//prelude, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc13_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//prelude, inst66 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//prelude, loc14_35, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//prelude, inst101 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.3bf = import_ref Core//prelude, loc18_41, unloaded
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.f80 = import_ref Core//prelude, loc21_36, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.3: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc21_24, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.9c1: type = import_ref Core//prelude, loc21_29, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.bd4 = import_ref Core//prelude, loc22_42, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.4: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl.loc3
// CHECK:STDOUT:     .C = %C.decl.loc5
// CHECK:STDOUT:     .F = %F.decl.loc17
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl.loc3: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.8dc [concrete] {} {
// CHECK:STDOUT:     %.loc4_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc4_7.2: type = converted %.loc4_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc4: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete = constants.%impl_witness.1bc]
// CHECK:STDOUT:   %C.decl.loc5: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc21: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5 (constants.%T.patt.3ad)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref.loc5: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc5_9.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc5_9.2 (constants.%T.826)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc17: %F.type.b25846.1 = fn_decl @F.1 [concrete = constants.%F.c41931.1] {
// CHECK:STDOUT:     %x.patt: <error> = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: <error> = value_param_pattern %x.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: <error> = value_param call_param0
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl.loc5 [concrete = constants.%C.generic]
// CHECK:STDOUT:       %.loc17_12: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc17_13: %I.type = converted %.loc17_12, <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: <error> = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc19: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.8dc [concrete] {} {
// CHECK:STDOUT:     %.loc20_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc20_7.2: type = converted %.loc20_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc20: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc21: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc21: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5 (constants.%T.patt.3ad)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref.loc21: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc21: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc5_9.2 (constants.%T.826)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc33: %F.type.b25846.2 = fn_decl @F.2 [concrete = constants.%F.c41931.2] {
// CHECK:STDOUT:     %x.patt: <error> = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: <error> = value_param_pattern %x.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: <error> = value_param call_param0
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl.loc5 [concrete = constants.%C.generic]
// CHECK:STDOUT:       %.loc33_12: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc33_13: %I.type = converted %.loc33_12, <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: <error> = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.837)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.02f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.630
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.ad0
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.3bf
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8dc: %.loc4_7.2 as %I.ref.loc4 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.f92(imports.%Core.import_ref.5ab3ec.3: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.e01)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl.f92(%T) [symbolic = %impl_witness (constants.%impl_witness.b81)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @impl.f92(%T) [symbolic = %Op.type (constants.%Op.type)]
// CHECK:STDOUT:   %Op: @impl.f92.%Op.type (%Op.type) = struct_value () [symbolic = %Op (constants.%Op)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.f92.%T (%T.8b3) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.9c1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.f80
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc5_9.1: %I.type) {
// CHECK:STDOUT:   %T.loc5_9.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc5_9.2 (constants.%T.826)]
// CHECK:STDOUT:   %T.patt.loc5: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5 (constants.%T.patt.3ad)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: @Convert.%ImplicitAs.type (%ImplicitAs.type.d62) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(imports.%Core.import_ref.5ab3ec.4: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.%T (%T.8b3)](%other.param_patt: @Op.%T (%T.8b3)) -> @Op.%T (%T.8b3) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1(%x.param_patt: <error>);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%x.param_patt: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T.826) {
// CHECK:STDOUT:   %T.loc5_9.2 => constants.%T.826
// CHECK:STDOUT:   %T.patt.loc5 => constants.%T.826
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.519) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%I.type) {
// CHECK:STDOUT:   %Dest => constants.%I.type
// CHECK:STDOUT:   %Dest.patt => constants.%I.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.c42
// CHECK:STDOUT:   %Self => constants.%Self.acc
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.7ef
// CHECK:STDOUT:   %Convert => constants.%Convert.c77
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.167
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %T.patt => constants.%T.8b3
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.b81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_interfaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.25f: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type %BitAnd.type [concrete]
// CHECK:STDOUT:   %assoc0: %BitAnd.assoc_type = assoc_entity element0, imports.%Core.import_ref.a93 [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.25f [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness.db8: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.f99: type = fn_type @Op.2, @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op.05a: %Op.type.f99 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %impl_witness.3ea: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl.f92(type) [concrete]
// CHECK:STDOUT:   %impl_witness.b81: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl.f92(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.eb8: type = fn_type @Op.2, @impl.f92(type) [concrete]
// CHECK:STDOUT:   %Op.444: %Op.type.eb8 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness type [concrete]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value type, (%impl_witness.3ea) [concrete]
// CHECK:STDOUT:   %.2ac: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %I.type, %Op.444 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.444, @Op.2(type) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %I.type, %Op.specific_fn [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@I & @J> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//prelude, inst101 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.08d: %BitAnd.assoc_type = import_ref Core//prelude, loc18_41, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.040: %BitAnd.type = import_ref Core//prelude, inst101 [no loc], loaded [symbolic = constants.%Self.25f]
// CHECK:STDOUT:   %Core.import_ref.51c: <witness> = import_ref Core//prelude, loc21_36, loaded [symbolic = @impl.f92.%impl_witness (constants.%impl_witness.b81)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc21_24, loaded [symbolic = @impl.f92.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.9c1: type = import_ref Core//prelude, loc21_29, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.1e6: @impl.f92.%Op.type (%Op.type.f99) = import_ref Core//prelude, loc22_42, loaded [symbolic = @impl.f92.%Op (constants.%Op.05a)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T)]
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
// CHECK:STDOUT:   %.loc14_14.1: type = value_of_initializer @impl.be3.%type.and [concrete = constants.%facet_type]
// CHECK:STDOUT:   %.loc14_14.2: type = converted @impl.be3.%type.and, %.loc14_14.1 [concrete = constants.%facet_type]
// CHECK:STDOUT:   impl_decl @impl.be3 [concrete] {} {
// CHECK:STDOUT:     %.loc14_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc14_7.2: type = converted %.loc14_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl.loc3 [concrete = constants.%I.type]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl.loc4 [concrete = constants.%J.type]
// CHECK:STDOUT:     %impl.elem0: %.2ac = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:     %bound_method.loc14_14.1: <bound method> = bound_method %I.ref, %impl.elem0 [concrete = constants.%Op.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc14_14.2: <bound method> = bound_method %I.ref, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:     %type.and: init type = call %bound_method.loc14_14.2(%I.ref, %J.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc16: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl.loc17: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.ad0
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.08d
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.f92(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl.f92(%T) [symbolic = %impl_witness (constants.%impl_witness.db8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl.f92(%T) [symbolic = %Op.type (constants.%Op.type.f99)]
// CHECK:STDOUT:   %Op: @impl.f92.%Op.type (%Op.type.f99) = struct_value () [symbolic = %Op (constants.%Op.05a)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.f92.%T (%T) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.9c1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.51c
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.be3: %.loc14_7.2 as file.%.loc14_14.2;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.040: %BitAnd.type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.25f)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type (%Self.as_type)](%other.param_patt: @Op.1.%Self.as_type (%Self.as_type)) -> @Op.1.%Self.as_type (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Core.import_ref.5ab3ec.2: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%T (%T)](%other.param_patt: @Op.2.%T (%T)) -> @Op.2.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.25f) {
// CHECK:STDOUT:   %Self => constants.%Self.25f
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.db8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:   %T.patt => type
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.3ea
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.eb8
// CHECK:STDOUT:   %Op => constants.%Op.444
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_never_assigned_associated_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.%U [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_tuple.type, <error>) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc18_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc18_7.2: type = converted %.loc18_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc18: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.1: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc18: %I.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc18: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc18: type = facet_access_type %.Self.ref.loc18 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc18_20: type = converted %.Self.ref.loc18, %.Self.as_type.loc18 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc18: <witness> = facet_access_witness %.Self.ref.loc18, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc18: type = impl_witness_access %.Self.as_wit.iface0.loc18, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc18_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc18_26.2: type = converted %.loc18_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc18_14: type = where_expr %.Self.1 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc18, %.loc18_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_tuple.type, <error>) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc22_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc22_7.2: type = converted %.loc22_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref.loc22: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.2: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc22: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc22: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc22: type = facet_access_type %.Self.ref.loc22 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc22_20: type = converted %.Self.ref.loc22, %.Self.as_type.loc22 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc22: <witness> = facet_access_witness %.Self.ref.loc22, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc22: type = impl_witness_access %.Self.as_wit.iface0.loc22, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc22_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc22_26.2: type = converted %.loc22_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc22_14: type = where_expr %.Self.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc22, %.loc22_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc1: %I.assoc_type = assoc_entity element1, @I.%U [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   .U = @U.%assoc1
// CHECK:STDOUT:   witness = (%T, %U)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc18_7.2 as %.loc18_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- include_files/facet_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %As.type.b51: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.b51 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.8ba: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.b4e: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.7f0: type = facet_access_type %Self.b4e [symbolic]
// CHECK:STDOUT:   %Convert.type.ad1: type = fn_type @Convert.1, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.0ed: %Convert.type.ad1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type %As.type.8ba [symbolic]
// CHECK:STDOUT:   %assoc0.ac5: %As.assoc_type = assoc_entity element0, @As.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.0f3: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.419: type = facet_access_type %Self.0f3 [symbolic]
// CHECK:STDOUT:   %Convert.type.4cf: type = fn_type @Convert.2, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.147: %Convert.type.4cf = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type %ImplicitAs.type.07f [symbolic]
// CHECK:STDOUT:   %assoc0.a50: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.e44: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.560: type = facet_access_type %Self.e44 [symbolic]
// CHECK:STDOUT:   %Op.type.613: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.d98: %Op.type.613 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type %BitAnd.type [concrete]
// CHECK:STDOUT:   %assoc0.121: %BitAnd.assoc_type = assoc_entity element0, @BitAnd.%Op.decl [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl), @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.28d: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.902: %Op.type.28d = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value %T, (%impl_witness) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:     .BitAnd = %BitAnd.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.b51 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %Dest.patt.loc9_14.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc9_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc9_14.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc9_14.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt.loc13_22.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc13_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc13_22.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc13_22.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BitAnd.decl: type = interface_decl @BitAnd [concrete = constants.%BitAnd.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc21_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc21_14.1 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:     %BitAnd.ref: type = name_ref BitAnd, file.%BitAnd.decl [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:     %T.loc21_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl), @impl(constants.%T) [symbolic = @impl.%impl_witness (constants.%impl_witness)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%Dest.loc9_14.1: type) {
// CHECK:STDOUT:   %Dest.loc9_14.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc9_14.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc9_14.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc9_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest.loc9_14.2)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self.2: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%Dest.loc9_14.2) [symbolic = %Convert.type (constants.%Convert.type.ad1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.ad1) = struct_value () [symbolic = %Convert (constants.%Convert.0ed)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As.%As.type (%As.type.8ba) [symbolic = %As.assoc_type (constants.%As.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc10_35.2: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc10_35.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:     %Convert.decl: @As.%Convert.type (%Convert.type.ad1) = fn_decl @Convert.1 [symbolic = @As.%Convert (constants.%Convert.0ed)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @As.%Dest.loc9_14.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = value_param call_param0
// CHECK:STDOUT:       %.loc10_20.1: type = splice_block %.loc10_20.3 [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)] {
// CHECK:STDOUT:         %.loc10_20.2: @Convert.1.%As.type (%As.type.8ba) = specific_constant @As.%Self.1, @As(constants.%Dest) [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%As.type (%As.type.8ba) = name_ref Self, %.loc10_20.2 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.as_type.loc10_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:         %.loc10_20.3: type = converted %Self.ref, %Self.as_type.loc10_20.2 [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc10_35.1: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc10_35.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc10_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc13_22.1: type) {
// CHECK:STDOUT:   %Dest.loc13_22.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc13_22.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc13_22.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc13_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc13_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%Dest.loc13_22.2) [symbolic = %Convert.type (constants.%Convert.type.4cf)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.4cf) = struct_value () [symbolic = %Convert (constants.%Convert.147)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc14_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.4cf) = fn_decl @Convert.2 [symbolic = @ImplicitAs.%Convert (constants.%Convert.147)] {
// CHECK:STDOUT:       %self.patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.2.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.2.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc13_22.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = value_param call_param0
// CHECK:STDOUT:       %.loc14_20.1: type = splice_block %.loc14_20.3 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)] {
// CHECK:STDOUT:         %.loc14_20.2: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.ref: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc14_20.2 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.as_type.loc14_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:         %.loc14_20.3: type = converted %Self.ref, %Self.as_type.loc14_20.2 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.2.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.2.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc14_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc14_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.e44]
// CHECK:STDOUT:   %Op.decl: %Op.type.613 = fn_decl @Op.1 [concrete = constants.%Op.d98] {
// CHECK:STDOUT:     %self.patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %other.patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:     %return.patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc18_37: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:     %Self.as_type.loc18_37: type = facet_access_type %Self.ref.loc18_37 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %.loc18_37: type = converted %Self.ref.loc18_37, %Self.as_type.loc18_37 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %self.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param call_param0
// CHECK:STDOUT:     %.loc18_15.1: type = splice_block %.loc18_15.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc18_15: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc18_15.2: type = facet_access_type %Self.ref.loc18_15 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc18_15.2: type = converted %Self.ref.loc18_15, %Self.as_type.loc18_15.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param call_param1
// CHECK:STDOUT:     %.loc18_28.1: type = splice_block %.loc18_28.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc18_28: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc18_28: type = facet_access_type %Self.ref.loc18_28 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc18_28.2: type = converted %Self.ref.loc18_28, %Self.as_type.loc18_28 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %BitAnd.assoc_type = assoc_entity element0, %Op.decl [concrete = constants.%assoc0.121]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc21_14.1: type) {
// CHECK:STDOUT:   %T.loc21_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc21_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Op.decl), @impl(%T.loc21_14.2) [symbolic = %impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T.loc21_14.2) [symbolic = %Op.type (constants.%Op.type.28d)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.28d) = struct_value () [symbolic = %Op (constants.%Op.902)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%T.loc21_14.2 (%T) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %BitAnd.ref {
// CHECK:STDOUT:     %Op.decl: @impl.%Op.type (%Op.type.28d) = fn_decl @Op.2 [symbolic = @impl.%Op (constants.%Op.902)] {
// CHECK:STDOUT:       %self.patt: @Op.2.%T (%T) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Op.2.%T (%T) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %other.patt: @Op.2.%T (%T) = binding_pattern other
// CHECK:STDOUT:       %other.param_patt: @Op.2.%T (%T) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:       %return.patt: @Op.2.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Op.2.%T (%T) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref.loc22_37: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Op.2.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref.loc22_15: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @Op.2.%T (%T) = bind_name self, %self.param
// CHECK:STDOUT:       %other.param: @Op.2.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %Self.ref.loc22_28: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %other: @Op.2.%T (%T) = bind_name other, %other.param
// CHECK:STDOUT:       %return.param: ref @Op.2.%T (%T) = out_param call_param2
// CHECK:STDOUT:       %return: ref @Op.2.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Op.decl
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@As.%Dest.loc9_14.1: type, @As.%Self.1: @As.%As.type (%As.type.8ba)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self: @Convert.1.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:   %Self.as_type.loc10_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(@ImplicitAs.%Dest.loc13_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:   %Self.as_type.loc14_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419)]() -> @Convert.2.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@BitAnd.%Self: %BitAnd.type) {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:   %Self.as_type.loc18_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560)](%other.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560)) -> @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(@impl.%T.loc21_14.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%T (%T)](%other.param_patt: @Op.2.%T (%T)) -> @Op.2.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc9_14.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc9_14.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.b4e) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.8ba
// CHECK:STDOUT:   %Self => constants.%Self.b4e
// CHECK:STDOUT:   %Self.as_type.loc10_20.1 => constants.%Self.as_type.7f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%Dest.loc9_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc13_22.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc13_22.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%Dest, constants.%Self.0f3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self.0f3
// CHECK:STDOUT:   %Self.as_type.loc14_20.1 => constants.%Self.as_type.419
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest.loc13_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.e44) {
// CHECK:STDOUT:   %Self => constants.%Self.e44
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%Self.as_type.560
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc21_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc21_14.2 => constants.%T
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc21_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%BitAnd.facet) {
// CHECK:STDOUT:   %Self => constants.%BitAnd.facet
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
