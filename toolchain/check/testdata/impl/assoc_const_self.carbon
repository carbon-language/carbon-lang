// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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

// CHECK:STDERR: fail_assoc_const_not_constant_after_conversion.carbon:[[@LINE+4]]:11: error: associated constant V given value `()` that is not constant after conversion to `C` [AssociatedConstantNotConstantAfterConversion]
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
  // CHECK:STDERR: fail_todo_constrained_fn.carbon:[[@LINE+7]]:3: error: cannot convert type `{}` into type implementing `I where .(I.V) = {} and...` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   F({});
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_todo_constrained_fn.carbon:[[@LINE-7]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! I where {} impls Core.ImplicitAs(.Self) and .V = {});
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F({});
}

// CHECK:STDOUT: --- assoc_const_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.9f2: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.bcb: type = symbolic_binding_type Self, 0, %Self.9f2 [symbolic]
// CHECK:STDOUT:   %require_complete.e1b: <witness> = require_complete_type %Self.binding.as_type.bcb [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.1aa: %I.assoc_type = assoc_entity element0, @I.%V [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %require_complete.d2d: <witness> = require_complete_type %.Self.binding.as_type [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %.Self.binding.as_type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %I_where.type.df2: type = facet_type <@I where %impl.elem0 = %empty_struct> [concrete]
// CHECK:STDOUT:   %I.impl_witness.0bd: <witness> = impl_witness file.%I.impl_witness_table.loc8 [concrete]
// CHECK:STDOUT:   %I.facet.c9b: %I.type = facet_value %empty_struct_type, (%I.impl_witness.0bd) [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %I_where.type.e54: type = facet_type <@I where %impl.elem0 = %int_0.5c6> [concrete]
// CHECK:STDOUT:   %I.impl_witness.bdd: <witness> = impl_witness file.%I.impl_witness_table.loc10 [concrete]
// CHECK:STDOUT:   %I.facet.5dc: %I.type = facet_value %i32, (%I.impl_witness.bdd) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %.loc8_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.1aa]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: %.Self.binding.as_type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc8_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_26.2: %empty_struct_type = converted %.loc8_26.1, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [concrete = constants.%I_where.type.df2] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc8_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table.loc8 = impl_witness_table (%impl_witness_assoc_constant.loc8), @empty_struct_type.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness.loc8: <witness> = impl_witness %I.impl_witness_table.loc8 [concrete = constants.%I.impl_witness.0bd]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc8: %empty_struct_type = impl_witness_assoc_constant constants.%empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   impl_decl @i32.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.1aa]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc10_21: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: %.Self.binding.as_type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %.loc10_15: type = where_expr %.Self [concrete = constants.%I_where.type.e54] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %int_0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table.loc10 = impl_witness_table (%impl_witness_assoc_constant.loc10), @i32.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness.loc10: <witness> = impl_witness %I.impl_witness_table.loc10 [concrete = constants.%I.impl_witness.bdd]
// CHECK:STDOUT:   %impl.elem0: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc10_15.1: <bound method> = bound_method constants.%int_0.5c6, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_15.2: <bound method> = bound_method constants.%int_0.5c6, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc10_15.2(constants.%int_0.5c6) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc10_15.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc10_15.2: %i32 = converted constants.%int_0.5c6, %.loc10_15.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc10: %i32 = impl_witness_assoc_constant constants.%int_0.6a9 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.9f2]
// CHECK:STDOUT:   %V: @V.%Self.binding.as_type (%Self.binding.as_type.bcb) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%V [concrete = constants.%assoc0.1aa]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .V = @V.%assoc0
// CHECK:STDOUT:   witness = (%V)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @V(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.9f2)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bcb)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type [symbolic = %require_complete (constants.%require_complete.e1b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const V:! @V.%Self.binding.as_type (%Self.binding.as_type.bcb);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_struct_type.as.I.impl: %.loc8_7.2 as %.loc8_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%I.impl_witness.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.as.I.impl: %i32 as %.loc10_15 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%I.impl_witness.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%Self.9f2) {
// CHECK:STDOUT:   %Self => constants.%Self.9f2
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.bcb
// CHECK:STDOUT:   %require_complete => constants.%require_complete.e1b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%.Self) {
// CHECK:STDOUT:   %Self => constants.%.Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%.Self.binding.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d2d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%I.facet.c9b) {
// CHECK:STDOUT:   %Self => constants.%I.facet.c9b
// CHECK:STDOUT:   %Self.binding.as_type => constants.%empty_struct_type
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%I.facet.5dc) {
// CHECK:STDOUT:   %Self => constants.%I.facet.5dc
// CHECK:STDOUT:   %Self.binding.as_type => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_assoc_const_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.9f2: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.bcb: type = symbolic_binding_type Self, 0, %Self.9f2 [symbolic]
// CHECK:STDOUT:   %require_complete.e1b: <witness> = require_complete_type %Self.binding.as_type.bcb [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.1aa: %I.assoc_type = assoc_entity element0, @I.%V [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %require_complete.d2d: <witness> = require_complete_type %.Self.binding.as_type [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %.Self.binding.as_type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %int_0> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %empty_struct_type, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %.loc15_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc15_7.2: type = converted %.loc15_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.1aa]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc15_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: %.Self.binding.as_type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %.loc15_14: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %int_0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @empty_struct_type.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %.loc15: %empty_struct_type = converted constants.%int_0, <error> [concrete = <error>]
// CHECK:STDOUT:   %impl_witness_assoc_constant: <error> = impl_witness_assoc_constant <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.9f2]
// CHECK:STDOUT:   %V: @V.%Self.binding.as_type (%Self.binding.as_type.bcb) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%V [concrete = constants.%assoc0.1aa]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .V = @V.%assoc0
// CHECK:STDOUT:   witness = (%V)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @V(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.9f2)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bcb)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type [symbolic = %require_complete (constants.%require_complete.e1b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const V:! @V.%Self.binding.as_type (%Self.binding.as_type.bcb);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_struct_type.as.I.impl: %.loc15_7.2 as %.loc15_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%Self.9f2) {
// CHECK:STDOUT:   %Self => constants.%Self.9f2
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.bcb
// CHECK:STDOUT:   %require_complete => constants.%require_complete.e1b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%.Self) {
// CHECK:STDOUT:   %Self => constants.%.Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%.Self.binding.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d2d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%empty_struct_type
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_assoc_const_not_constant_after_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.9f2: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.bcb: type = symbolic_binding_type Self, 0, %Self.9f2 [symbolic]
// CHECK:STDOUT:   %require_complete.e1b: <witness> = require_complete_type %Self.binding.as_type.bcb [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.1aa: %I.assoc_type = assoc_entity element0, @I.%V [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.396: type = facet_type <@ImplicitAs, @ImplicitAs(%C)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.56d: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%C) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness file.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert.type: type = fn_type @empty_tuple.type.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert: %empty_tuple.type.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.396 = facet_value %empty_tuple.type, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %require_complete.d2d: <witness> = require_complete_type %.Self.binding.as_type [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %.Self.binding.as_type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_tuple> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %.77a: type = fn_type_with_self_type %ImplicitAs.Convert.type.56d, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %empty_tuple, %empty_tuple.type.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.ImplicitAs.impl [concrete] {} {
// CHECK:STDOUT:     %.loc10_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_7.2: type = converted %.loc10_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%C)> [concrete = constants.%ImplicitAs.type.396]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (@empty_tuple.type.as.ImplicitAs.impl.%empty_tuple.type.as.ImplicitAs.impl.Convert.decl), @empty_tuple.type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness]
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.1aa]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc18_19: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: %.Self.binding.as_type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc18_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc18_25.2: %empty_tuple.type = converted %.loc18_25.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc18_13: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc18_25.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %impl.elem0: %.77a = impl_witness_access constants.%ImplicitAs.impl_witness, element0 [concrete = constants.%empty_tuple.type.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method constants.%empty_tuple, %impl.elem0 [concrete = constants.%empty_tuple.type.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %.loc18_13.1: ref %C = temporary_storage
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert.call: init %C = call %bound_method(constants.%empty_tuple) to %.loc18_13.1
// CHECK:STDOUT:   %.loc18_13.2: init %C = converted constants.%empty_tuple, %empty_tuple.type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc18_13.3: ref %C = temporary %.loc18_13.1, %.loc18_13.2
// CHECK:STDOUT:   %.loc18_13.4: %C = acquire_value %.loc18_13.3
// CHECK:STDOUT:   %impl_witness_assoc_constant: <error> = impl_witness_assoc_constant <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.9f2]
// CHECK:STDOUT:   %V: @V.%Self.binding.as_type (%Self.binding.as_type.bcb) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%V [concrete = constants.%assoc0.1aa]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .V = @V.%assoc0
// CHECK:STDOUT:   witness = (%V)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @V(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.9f2)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bcb)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type [symbolic = %require_complete (constants.%require_complete.e1b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const V:! @V.%Self.binding.as_type (%Self.binding.as_type.bcb);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.ImplicitAs.impl: %.loc10_7.2 as %ImplicitAs.type {
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert.decl: %empty_tuple.type.as.ImplicitAs.impl.Convert.type = fn_decl @empty_tuple.type.as.ImplicitAs.impl.Convert [concrete = constants.%empty_tuple.type.as.ImplicitAs.impl.Convert] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc11_21.1: type = splice_block %.loc11_21.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc11_21.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc11_21.3: type = converted %.loc11_21.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %empty_tuple.type = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Convert = %empty_tuple.type.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref as %.loc18_13 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%I.impl_witness
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
// CHECK:STDOUT: fn @empty_tuple.type.as.ImplicitAs.impl.Convert(%self.param: %empty_tuple.type) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_41.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc11_41.2: init %C = class_init (), %return [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc11_42: init %C = converted %.loc11_41.1, %.loc11_41.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   return %.loc11_42 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%Self.9f2) {
// CHECK:STDOUT:   %Self => constants.%Self.9f2
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.bcb
// CHECK:STDOUT:   %require_complete => constants.%require_complete.e1b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%.Self) {
// CHECK:STDOUT:   %Self => constants.%.Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%.Self.binding.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d2d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_monomorphization_failure.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.3b7: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.f7b: type = facet_type <@I, @I(%N)> [symbolic]
// CHECK:STDOUT:   %Self.d78: %I.type.f7b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.e44: type = symbolic_binding_type Self, 1, %Self.d78 [symbolic]
// CHECK:STDOUT:   %array_type: type = array_type %N, %Self.binding.as_type.e44 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type [symbolic]
// CHECK:STDOUT:   %I.assoc_type.44f: type = assoc_entity_type @I, @I(%N) [symbolic]
// CHECK:STDOUT:   %assoc0.8bd: %I.assoc_type.44f = assoc_entity element0, @I.%V [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.Op.type: type = fn_type @Negate.Op [concrete]
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness imports.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value Core.IntLiteral, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %.6a1: type = fn_type_with_self_type %Negate.Op.type, %Negate.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.type: type = fn_type @Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op: %Core.IntLiteral.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.bound: <bound method> = bound_method %int_1, %Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [concrete]
// CHECK:STDOUT:   %I.type.f11: type = facet_type <@I, @I(%int_-1)> [concrete]
// CHECK:STDOUT:   %.Self.687: %I.type.f11 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.623: %I.type.f11 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.247: type = assoc_entity_type @I, @I(%int_-1) [concrete]
// CHECK:STDOUT:   %assoc0.63a: %I.assoc_type.247 = assoc_entity element0, @I.%V [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.687 [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.687, @I, @I(%int_-1) [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Negate = %Core.Negate
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Negate: type = import_ref Core//prelude/parts/int_literal, Negate, loaded [concrete = constants.%Negate.type]
// CHECK:STDOUT:   %Core.import_ref.abd = import_ref Core//prelude/parts/int_literal, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b9e: %Core.IntLiteral.as.Negate.impl.Op.type = import_ref Core//prelude/parts/int_literal, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.abd, %Core.import_ref.b9e), @Core.IntLiteral.as.Negate.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_33.1: type = splice_block %.loc4_33.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_33.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_33.3: type = converted %IntLiteral.call, %.loc4_33.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_13.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_13.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.<error>.impl [concrete] {} {
// CHECK:STDOUT:     %.loc15_7.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc15_7.2: type = converted %.loc15_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:     %impl.elem1: %.6a1 = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_1, %impl.elem1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.Negate.impl.Op.call: init Core.IntLiteral = call %bound_method(%int_1) [concrete = constants.%int_-1]
// CHECK:STDOUT:     %.loc15_16.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.Negate.impl.Op.call [concrete = constants.%int_-1]
// CHECK:STDOUT:     %.loc15_16.2: Core.IntLiteral = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc15_16.1 [concrete = constants.%int_-1]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%int_-1)> [concrete = constants.%I.type.f11]
// CHECK:STDOUT:     %.Self: %I.type.f11 = symbolic_binding .Self [symbolic_self = constants.%.Self.687]
// CHECK:STDOUT:     %.Self.ref: %I.type.f11 = name_ref .Self, %.Self [symbolic_self = constants.%.Self.687]
// CHECK:STDOUT:     %.loc15_24.1: %I.assoc_type.247 = specific_constant @V.%assoc0, @I(constants.%int_-1) [concrete = constants.%assoc0.63a]
// CHECK:STDOUT:     %V.ref: %I.assoc_type.247 = name_ref V, %.loc15_24.1 [concrete = constants.%assoc0.63a]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc15_24.2: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: <error> = impl_witness_access constants.%I.lookup_impl_witness, element0 [concrete = <error>]
// CHECK:STDOUT:     %.loc15_30: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_18: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type.f11
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%N.loc4_13.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc4_13.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_13.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%N.loc4_13.1)> [symbolic = %I.type (constants.%I.type.f7b)]
// CHECK:STDOUT:   %Self.loc4_36.2: @I.%I.type (%I.type.f7b) = symbolic_binding Self, 1 [symbolic = %Self.loc4_36.2 (constants.%Self.d78)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%N.loc4_13.1) [symbolic = %I.assoc_type (constants.%I.assoc_type.44f)]
// CHECK:STDOUT:   %assoc0: @I.%I.assoc_type (%I.assoc_type.44f) = assoc_entity element0, %V [symbolic = %assoc0 (constants.%assoc0.8bd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_36.1: @I.%I.type (%I.type.f7b) = symbolic_binding Self, 1 [symbolic = %Self.loc4_36.2 (constants.%Self.d78)]
// CHECK:STDOUT:     %V: @V.%array_type (%array_type) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:       %assoc0: @I.%I.assoc_type (%I.assoc_type.44f) = assoc_entity element0, @I.%V [symbolic = @I.%assoc0 (constants.%assoc0.8bd)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_36.1
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .V = @V.%assoc0
// CHECK:STDOUT:     witness = (%V)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @V(@I.%N.loc4_13.2: Core.IntLiteral, @I.%Self.loc4_36.1: @I.%I.type (%I.type.f7b)) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%N)> [symbolic = %I.type (constants.%I.type.f7b)]
// CHECK:STDOUT:   %Self: @V.%I.type (%I.type.f7b) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.d78)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e44)]
// CHECK:STDOUT:   %array_type: type = array_type %N, %Self.binding.as_type [symbolic = %array_type (constants.%array_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const V:! @V.%array_type (%array_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_struct_type.as.<error>.impl: %.loc15_7.2 as %.loc15_18 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%N) {
// CHECK:STDOUT:   %N.loc4_13.1 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%N, constants.%Self.d78) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %I.type => constants.%I.type.f7b
// CHECK:STDOUT:   %Self => constants.%Self.d78
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.e44
// CHECK:STDOUT:   %array_type => constants.%array_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%int_-1) {
// CHECK:STDOUT:   %N.loc4_13.1 => constants.%int_-1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.f11
// CHECK:STDOUT:   %Self.loc4_36.2 => constants.%Self.623
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.247
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.63a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%int_-1, constants.%.Self.687) {
// CHECK:STDOUT:   %N => constants.%int_-1
// CHECK:STDOUT:   %I.type => constants.%I.type.f11
// CHECK:STDOUT:   %Self => constants.%.Self.687
// CHECK:STDOUT:   %Self.binding.as_type => constants.%.Self.binding.as_type
// CHECK:STDOUT:   %array_type => <error>
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_constrained_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.9f2: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.bcb: type = symbolic_binding_type Self, 0, %Self.9f2 [symbolic]
// CHECK:STDOUT:   %require_complete.e1b: <witness> = require_complete_type %Self.binding.as_type.bcb [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.1aa: %I.assoc_type = assoc_entity element0, @I.%V [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.3b7: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.420: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.420 [symbolic_self]
// CHECK:STDOUT:   %ImplicitAs.type.164: type = facet_type <@ImplicitAs, @ImplicitAs(%.Self.binding.as_type)> [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.420, @I [symbolic_self]
// CHECK:STDOUT:   %require_complete.d2d: <witness> = require_complete_type %.Self.binding.as_type [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %.Self.binding.as_type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_struct and TODO> [concrete]
// CHECK:STDOUT:   %T: %I_where.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.a47: type = pattern_type %I_where.type [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.a47 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_12.1: type = splice_block %.loc8_12.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self.2: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.420]
// CHECK:STDOUT:       %.loc8_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:       %.Self.ref.loc8_43: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.420]
// CHECK:STDOUT:       %.Self.as_type.loc8_48: type = facet_access_type %.Self.ref.loc8_43 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc8_48: type = converted %.Self.ref.loc8_43, %.Self.as_type.loc8_48 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%.Self.binding.as_type)> [symbolic_self = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:       %.loc8_19.2: type = converted %.loc8_19.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.Self.ref.loc8_54: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.420]
// CHECK:STDOUT:       %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.1aa]
// CHECK:STDOUT:       %.Self.as_type.loc8_54: type = facet_access_type %.Self.ref.loc8_54 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc8_54: type = converted %.Self.ref.loc8_54, %.Self.as_type.loc8_54 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %impl.elem0: %.Self.binding.as_type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc8_60.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc8_60.2: %empty_struct_type = converted %.loc8_60.1, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc8_12.2: type = where_expr %.Self.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:         requirement_impls %.loc8_19.2, %ImplicitAs.type
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %.loc8_60.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_6.2: %I_where.type = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallF.decl: %CallF.type = fn_decl @CallF [concrete = constants.%CallF] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.9f2]
// CHECK:STDOUT:   %V: @V.%Self.binding.as_type (%Self.binding.as_type.bcb) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%V [concrete = constants.%assoc0.1aa]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .V = @V.%assoc0
// CHECK:STDOUT:   witness = (%V)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @V(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.9f2)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bcb)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.binding.as_type [symbolic = %require_complete (constants.%require_complete.e1b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const V:! @V.%Self.binding.as_type (%Self.binding.as_type.bcb);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_6.2: %I_where.type) {
// CHECK:STDOUT:   %T.loc8_6.1: %I_where.type = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc19: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%Self.9f2) {
// CHECK:STDOUT:   %Self => constants.%Self.9f2
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.bcb
// CHECK:STDOUT:   %require_complete => constants.%require_complete.e1b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%.Self.420) {
// CHECK:STDOUT:   %Self => constants.%.Self.420
// CHECK:STDOUT:   %Self.binding.as_type => constants.%.Self.binding.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d2d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc8_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
