// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/assoc_const_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/assoc_const_self.carbon

// --- assoc_const_self.carbon

library "[[@TEST_NAME]]";

interface I {
  let V:! Self;
}

impl {} as I where .V = {} {}

impl i32 as I where .V = 0 {}

// --- fail_assoc_const_mismatch.carbon

library "[[@TEST_NAME]]";

interface I {
  let V:! Self;
}

// CHECK:STDERR: fail_assoc_const_mismatch.carbon:[[@LINE+7]]:12: error: cannot implicitly convert expression of type `Core.IntLiteral` to `{}` [ConversionFailure]
// CHECK:STDERR: impl {} as I where .V = 0 {}
// CHECK:STDERR:            ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_assoc_const_mismatch.carbon:[[@LINE+4]]:12: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs({})` [MissingImplInMemberAccessNote]
// CHECK:STDERR: impl {} as I where .V = 0 {}
// CHECK:STDERR:            ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl {} as I where .V = 0 {}

// --- fail_assoc_const_not_constant_after_conversion.carbon

library "[[@TEST_NAME]]";

interface I {
  let V:! Self;
}

class C {}

impl () as Core.ImplicitAs(C) {
  fn Convert[self: ()]() -> C { return {}; }
}

// CHECK:STDERR: fail_assoc_const_not_constant_after_conversion.carbon:[[@LINE+4]]:11: error: associated constant V given value that is not constant after conversion to `C` [AssociatedConstantNotConstantAfterConversion]
// CHECK:STDERR: impl C as I where .V = () {}
// CHECK:STDERR:           ^~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C as I where .V = () {}

// --- fail_monomorphization_failure.carbon

library "[[@TEST_NAME]]";

interface I(N:! Core.IntLiteral()) {
  // CHECK:STDERR: fail_monomorphization_failure.carbon:[[@LINE+3]]:17: error: array bound of -1 is negative [ArrayBoundNegative]
  // CHECK:STDERR:   let V:! array(Self, N);
  // CHECK:STDERR:                 ^~~~
  let V:! array(Self, N);
}

// CHECK:STDERR: fail_monomorphization_failure.carbon:[[@LINE+4]]:24: note: in `V` used here [ResolvingSpecificHere]
// CHECK:STDERR: impl {} as I(-1) where .V = () {}
// CHECK:STDERR:                        ^~
// CHECK:STDERR:
impl {} as I(-1) where .V = () {}

// --- fail_todo_constrained_fn.carbon

library "[[@TEST_NAME]]";

interface I {
  let V:! Self;
}

fn F(T:! I where {} impls Core.ImplicitAs(.Self) and .V = {});

fn CallF() {
  // TODO: This call should eventually work.
  // CHECK:STDERR: fail_todo_constrained_fn.carbon:[[@LINE+10]]:3: error: cannot implicitly convert non-type value of type `{}` into type implementing `I where .(I.V) = {} and...` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   F({});
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_todo_constrained_fn.carbon:[[@LINE+7]]:3: note: type `{}` does not implement interface `Core.ImplicitAs(I where .(I.V) = {} and...)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   F({});
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_todo_constrained_fn.carbon:[[@LINE-10]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I where {} impls Core.ImplicitAs(.Self) and .V = {});
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F({});
}

// CHECK:STDOUT: --- assoc_const_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.b70: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %require_complete.9b1: <witness> = require_complete_type %Self.as_type.b70 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.45f: %I.assoc_type = assoc_entity element0, @I.%V [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %I.facet.d87: %I.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %require_complete.d0c: <witness> = require_complete_type %.Self.as_type [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %.Self.as_type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %I_where.type.9d4: type = facet_type <@I where %impl.elem0 = %empty_struct> [concrete]
// CHECK:STDOUT:   %impl_witness.313: <witness> = impl_witness (file.%impl_witness_assoc_constant.loc8) [concrete]
// CHECK:STDOUT:   %I.facet.012: %I.type = facet_value %empty_struct_type, (%impl_witness.313) [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %I_where.type.b5a: type = facet_type <@I where %impl.elem0 = %int_0.5c6> [concrete]
// CHECK:STDOUT:   %impl_witness.2ac: <witness> = impl_witness (file.%impl_witness_assoc_constant.loc10) [concrete]
// CHECK:STDOUT:   %I.facet.255: %I.type = facet_value %i32, (%impl_witness.2ac) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
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
// CHECK:STDOUT:   impl_decl @impl.77e [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.45f]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: %.Self.as_type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc8_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_26.2: %empty_struct_type = converted %.loc8_26.1, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [concrete = constants.%I_where.type.9d4] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc8_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc8: <witness> = impl_witness (%impl_witness_assoc_constant.loc8) [concrete = constants.%impl_witness.313]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc8: %empty_struct_type = impl_witness_assoc_constant constants.%empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   impl_decl @impl.a12 [concrete] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.45f]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc10_21: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: %.Self.as_type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %.loc10_15: type = where_expr %.Self [concrete = constants.%I_where.type.b5a] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %int_0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc10: <witness> = impl_witness (%impl_witness_assoc_constant.loc10) [concrete = constants.%impl_witness.2ac]
// CHECK:STDOUT:   %impl.elem0: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc10_15.1: <bound method> = bound_method constants.%int_0.5c6, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_15.2: <bound method> = bound_method constants.%int_0.5c6, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc10_15.2(constants.%int_0.5c6) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc10_15.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc10_15.2: %i32 = converted constants.%int_0.5c6, %.loc10_15.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc10: %i32 = impl_witness_assoc_constant constants.%int_0.6a9 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %V: @V.%Self.as_type (%Self.as_type.b70) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%V [concrete = constants.%assoc0.45f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .V = @V.%assoc0
// CHECK:STDOUT:   witness = (%V)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @V(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.b70)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @V.%Self.as_type (%Self.as_type.b70) [symbolic = %require_complete (constants.%require_complete.9b1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const V:! @V.%Self.as_type (%Self.as_type.b70);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.77e: %.loc8_7.2 as %.loc8_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.a12: %i32 as %.loc10_15 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.b70
// CHECK:STDOUT:   %require_complete => constants.%require_complete.9b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%I.facet.d87) {
// CHECK:STDOUT:   %Self => constants.%I.facet.d87
// CHECK:STDOUT:   %Self.as_type => constants.%.Self.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d0c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%I.facet.012) {
// CHECK:STDOUT:   %Self => constants.%I.facet.012
// CHECK:STDOUT:   %Self.as_type => constants.%empty_struct_type
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%I.facet.255) {
// CHECK:STDOUT:   %Self => constants.%I.facet.255
// CHECK:STDOUT:   %Self.as_type => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_assoc_const_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.b70: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %require_complete.9b1: <witness> = require_complete_type %Self.as_type.b70 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.45f: %I.assoc_type = assoc_entity element0, @I.%V [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %I.facet.d87: %I.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %require_complete.d0c: <witness> = require_complete_type %.Self.as_type [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %.Self.as_type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %int_0> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (file.%impl_witness_assoc_constant) [concrete]
// CHECK:STDOUT:   %I.facet.5b8: %I.type = facet_value %empty_struct_type, (%impl_witness) [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc15_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc15_7.2: type = converted %.loc15_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.45f]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc15_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: %.Self.as_type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %.loc15_14: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %int_0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%impl_witness_assoc_constant) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %.loc15: %empty_struct_type = converted constants.%int_0, <error> [concrete = <error>]
// CHECK:STDOUT:   %impl_witness_assoc_constant: <error> = impl_witness_assoc_constant <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %V: @V.%Self.as_type (%Self.as_type.b70) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%V [concrete = constants.%assoc0.45f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .V = @V.%assoc0
// CHECK:STDOUT:   witness = (%V)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @V(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.b70)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @V.%Self.as_type (%Self.as_type.b70) [symbolic = %require_complete (constants.%require_complete.9b1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const V:! @V.%Self.as_type (%Self.as_type.b70);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc15_7.2 as %.loc15_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.b70
// CHECK:STDOUT:   %require_complete => constants.%require_complete.9b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%I.facet.d87) {
// CHECK:STDOUT:   %Self => constants.%I.facet.d87
// CHECK:STDOUT:   %Self.as_type => constants.%.Self.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d0c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%I.facet.5b8) {
// CHECK:STDOUT:   %Self => constants.%I.facet.5b8
// CHECK:STDOUT:   %Self.as_type => constants.%empty_struct_type
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_assoc_const_not_constant_after_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.b70: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %require_complete.9b1: <witness> = require_complete_type %Self.as_type.b70 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.45f: %I.assoc_type = assoc_entity element0, @I.%V [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.15e: type = facet_type <@ImplicitAs, @ImplicitAs(%C)> [concrete]
// CHECK:STDOUT:   %Convert.type.56d: type = fn_type @Convert.1, @ImplicitAs(%C) [concrete]
// CHECK:STDOUT:   %impl_witness.3db: <witness> = impl_witness (@impl.e86.%Convert.decl) [concrete]
// CHECK:STDOUT:   %Convert.type.721: type = fn_type @Convert.2 [concrete]
// CHECK:STDOUT:   %Convert.155: %Convert.type.721 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.15e = facet_value %empty_tuple.type, (%impl_witness.3db) [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %I.facet.d87: %I.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %require_complete.d0c: <witness> = require_complete_type %.Self.as_type [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %.Self.as_type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_tuple> [concrete]
// CHECK:STDOUT:   %impl_witness.880: <witness> = impl_witness (file.%impl_witness_assoc_constant) [concrete]
// CHECK:STDOUT:   %I.facet.3ae: %I.type = facet_value %C, (%impl_witness.880) [concrete]
// CHECK:STDOUT:   %.32d: type = fn_type_with_self_type %Convert.type.56d, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %empty_tuple, %Convert.155 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:   impl_decl @impl.e86 [concrete] {} {
// CHECK:STDOUT:     %.loc10_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_7.2: type = converted %.loc10_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%C)> [concrete = constants.%ImplicitAs.type.15e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc10: <witness> = impl_witness (@impl.e86.%Convert.decl) [concrete = constants.%impl_witness.3db]
// CHECK:STDOUT:   impl_decl @impl.bb1 [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.45f]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc18_19: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: %.Self.as_type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc18_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc18_25.2: %empty_tuple.type = converted %.loc18_25.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc18_13: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc18_25.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc18: <witness> = impl_witness (%impl_witness_assoc_constant) [concrete = constants.%impl_witness.880]
// CHECK:STDOUT:   %impl.elem0: %.32d = impl_witness_access constants.%impl_witness.3db, element0 [concrete = constants.%Convert.155]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method constants.%empty_tuple, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %.loc18_13.1: ref %C = temporary_storage
// CHECK:STDOUT:   %Convert.call: init %C = call %bound_method(constants.%empty_tuple) to %.loc18_13.1
// CHECK:STDOUT:   %.loc18_13.2: init %C = converted constants.%empty_tuple, %Convert.call
// CHECK:STDOUT:   %.loc18_13.3: ref %C = temporary %.loc18_13.1, %.loc18_13.2
// CHECK:STDOUT:   %.loc18_13.4: %C = bind_value %.loc18_13.3
// CHECK:STDOUT:   %impl_witness_assoc_constant: <error> = impl_witness_assoc_constant <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %V: @V.%Self.as_type (%Self.as_type.b70) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%V [concrete = constants.%assoc0.45f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .V = @V.%assoc0
// CHECK:STDOUT:   witness = (%V)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @V(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.b70)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @V.%Self.as_type (%Self.as_type.b70) [symbolic = %require_complete (constants.%require_complete.9b1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const V:! @V.%Self.as_type (%Self.as_type.b70);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.e86: %.loc10_7.2 as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.721 = fn_decl @Convert.2 [concrete = constants.%Convert.155] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc11_21.1: type = splice_block %.loc11_21.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc11_21.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc11_21.3: type = converted %.loc11_21.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.bb1: %C.ref as %.loc18_13 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc18
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
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: %empty_tuple.type]() -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_41.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc11_41.2: init %C = class_init (), %return [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc11_42: init %C = converted %.loc11_41.1, %.loc11_41.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   return %.loc11_42 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.b70
// CHECK:STDOUT:   %require_complete => constants.%require_complete.9b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%I.facet.d87) {
// CHECK:STDOUT:   %Self => constants.%I.facet.d87
// CHECK:STDOUT:   %Self.as_type => constants.%.Self.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d0c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%I.facet.3ae) {
// CHECK:STDOUT:   %Self => constants.%I.facet.3ae
// CHECK:STDOUT:   %Self.as_type => constants.%C
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_monomorphization_failure.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.8a1: type = facet_type <@I, @I(%N)> [symbolic]
// CHECK:STDOUT:   %Self.3a0: %I.type.8a1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.72b: type = facet_access_type %Self.3a0 [symbolic]
// CHECK:STDOUT:   %array_type: type = array_type %N, %Self.as_type.72b [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type [symbolic]
// CHECK:STDOUT:   %I.assoc_type.9df: type = assoc_entity_type @I, @I(%N) [symbolic]
// CHECK:STDOUT:   %assoc0.293: %I.assoc_type.9df = assoc_entity element0, @I.%V [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Op.type.e42: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.c15) [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value Core.IntLiteral, (%impl_witness) [concrete]
// CHECK:STDOUT:   %.690: type = fn_type_with_self_type %Op.type.e42, %Negate.facet [concrete]
// CHECK:STDOUT:   %Op.type.1be: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.bba: %Op.type.1be = struct_value () [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_1, %Op.bba [concrete]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [concrete]
// CHECK:STDOUT:   %I.type.057: type = facet_type <@I, @I(%int_-1)> [concrete]
// CHECK:STDOUT:   %.Self: %I.type.057 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Self.e2e: %I.type.057 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.247: type = assoc_entity_type @I, @I(%int_-1) [concrete]
// CHECK:STDOUT:   %assoc0.350: %I.assoc_type.247 = assoc_entity element0, @I.%V [concrete]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %I.facet: %I.type.057 = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Negate = %Core.Negate
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %N.patt.loc4_13.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_13.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_33.1: type = splice_block %.loc4_33.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_33.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_33.3: type = converted %int_literal.make_type, %.loc4_33.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_13.1: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc4_13.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.19e [concrete] {} {
// CHECK:STDOUT:     %.loc15_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc15_7.2: type = converted %.loc15_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:     %impl.elem0.loc15_14: %.690 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%Op.bba]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_1, %impl.elem0.loc15_14 [concrete = constants.%Op.bound]
// CHECK:STDOUT:     %int.snegate: init Core.IntLiteral = call %bound_method(%int_1) [concrete = constants.%int_-1]
// CHECK:STDOUT:     %.loc15_16.1: Core.IntLiteral = value_of_initializer %int.snegate [concrete = constants.%int_-1]
// CHECK:STDOUT:     %.loc15_16.2: Core.IntLiteral = converted %int.snegate, %.loc15_16.1 [concrete = constants.%int_-1]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%int_-1)> [concrete = constants.%I.type.057]
// CHECK:STDOUT:     %.Self: %I.type.057 = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type.057 = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.loc15_24.1: %I.assoc_type.247 = specific_constant @V.%assoc0, @I(constants.%int_-1) [concrete = constants.%assoc0.350]
// CHECK:STDOUT:     %V.ref: %I.assoc_type.247 = name_ref V, %.loc15_24.1 [concrete = constants.%assoc0.350]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc15_24.2: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc15_24: <error> = impl_witness_access %.Self.as_wit.iface0, element0 [concrete = <error>]
// CHECK:STDOUT:     %.loc15_30: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_18: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc15_24, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%N.loc4_13.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc4_13.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc4_13.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc4_13.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_13.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%N.loc4_13.2)> [symbolic = %I.type (constants.%I.type.8a1)]
// CHECK:STDOUT:   %Self.2: @I.%I.type (%I.type.8a1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.3a0)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%N.loc4_13.2) [symbolic = %I.assoc_type (constants.%I.assoc_type.9df)]
// CHECK:STDOUT:   %assoc0: @I.%I.assoc_type (%I.assoc_type.9df) = assoc_entity element0, %V [symbolic = %assoc0 (constants.%assoc0.293)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.8a1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.3a0)]
// CHECK:STDOUT:     %V: @V.%array_type (%array_type) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:       %assoc0: @I.%I.assoc_type (%I.assoc_type.9df) = assoc_entity element0, @I.%V [symbolic = @I.%assoc0 (constants.%assoc0.293)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .V = @V.%assoc0
// CHECK:STDOUT:     witness = (%V)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @V(@I.%N.loc4_13.1: Core.IntLiteral, @I.%Self.1: @I.%I.type (%I.type.8a1)) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%N)> [symbolic = %I.type (constants.%I.type.8a1)]
// CHECK:STDOUT:   %Self: @V.%I.type (%I.type.8a1) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3a0)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.72b)]
// CHECK:STDOUT:   %array_type: type = array_type %N, %Self.as_type [symbolic = %array_type (constants.%array_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @V.%array_type (%array_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const V:! @V.%array_type (%array_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.19e: %.loc15_7.2 as %.loc15_18 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%N) {
// CHECK:STDOUT:   %N.loc4_13.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc4_13.2 => constants.%N.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%N, constants.%Self.3a0) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %I.type => constants.%I.type.8a1
// CHECK:STDOUT:   %Self => constants.%Self.3a0
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.72b
// CHECK:STDOUT:   %array_type => constants.%array_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@V.%N) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%N.loc4_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%int_-1) {
// CHECK:STDOUT:   %N.loc4_13.2 => constants.%int_-1
// CHECK:STDOUT:   %N.patt.loc4_13.2 => constants.%N.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.057
// CHECK:STDOUT:   %Self.2 => constants.%Self.e2e
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.247
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.350
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%int_-1, constants.%I.facet) {
// CHECK:STDOUT:   %N => constants.%int_-1
// CHECK:STDOUT:   %I.type => constants.%I.type.057
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type => constants.%.Self.as_type
// CHECK:STDOUT:   %array_type => <error>
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_constrained_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.b70: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %require_complete.9b1: <witness> = require_complete_type %Self.as_type.b70 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.45f: %I.assoc_type = assoc_entity element0, @I.%V [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %ImplicitAs.type.2a8: type = facet_type <@ImplicitAs, @ImplicitAs(%.Self.as_type)> [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %require_complete.d0c: <witness> = require_complete_type %.Self.as_type [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %.Self.as_type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_struct and TODO> [concrete]
// CHECK:STDOUT:   %T: %I_where.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %I_where.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallF.type: type = fn_type @CallF [concrete]
// CHECK:STDOUT:   %CallF: %CallF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .CallF = %CallF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_6.1: %I_where.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_12.1: type = splice_block %.loc8_12.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:       %.Self.ref.loc8_43: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.as_type.loc8_48: type = facet_access_type %.Self.ref.loc8_43 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc8_48: type = converted %.Self.ref.loc8_43, %.Self.as_type.loc8_48 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%.Self.as_type)> [symbolic_self = constants.%ImplicitAs.type.2a8]
// CHECK:STDOUT:       %.loc8_19.2: type = converted %.loc8_19.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.Self.ref.loc8_54: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.45f]
// CHECK:STDOUT:       %.Self.as_type.loc8_54: type = facet_access_type %.Self.ref.loc8_54 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc8_54: type = converted %.Self.ref.loc8_54, %.Self.as_type.loc8_54 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref.loc8_54, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:       %impl.elem0: %.Self.as_type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc8_60.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc8_60.2: %empty_struct_type = converted %.loc8_60.1, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc8_12.2: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:         requirement_impls %.loc8_19.2, %ImplicitAs.type
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %.loc8_60.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_6.1: %I_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_6.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallF.decl: %CallF.type = fn_decl @CallF [concrete = constants.%CallF] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %V: @V.%Self.as_type (%Self.as_type.b70) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%V [concrete = constants.%assoc0.45f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .V = @V.%assoc0
// CHECK:STDOUT:   witness = (%V)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @V(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.b70)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @V.%Self.as_type (%Self.as_type.b70) [symbolic = %require_complete (constants.%require_complete.9b1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const V:! @V.%Self.as_type (%Self.as_type.b70);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_6.1: %I_where.type) {
// CHECK:STDOUT:   %T.loc8_6.2: %I_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_6.2: %I_where.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_6.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc8_6.1: %I_where.type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc22_6: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc22_7: %I_where.type = converted %.loc22_6, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.b70
// CHECK:STDOUT:   %require_complete => constants.%require_complete.9b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type => constants.%.Self.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d0c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc8_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_6.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
