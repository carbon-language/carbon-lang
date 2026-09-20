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
  let V: Self;
}

impl {} as I where .V = {} {}

impl i32 as I where .V = 0 {}

// --- fail_assoc_const_mismatch.carbon

library "[[@TEST_NAME]]";

interface I {
  let V: Self;
}

// CHECK:STDERR: fail_assoc_const_mismatch.carbon:[[@LINE+3]]:12-25: error: cannot implicitly convert expression of type `Core.IntLiteral` to `{}` [ConversionFailure]
// CHECK:STDERR: fail_assoc_const_mismatch.carbon:[[@LINE+2]]:12-25: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs({})` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:
impl {} as I where .V = 0 {}

// --- fail_assoc_const_not_constant_after_conversion.carbon

library "[[@TEST_NAME]]";

interface I {
  let V: Self;
}

class C {}

impl () as Core.ImplicitAs(C) {
  fn Convert(unused self: ()) -> C { return {}; }
}

// CHECK:STDERR: fail_assoc_const_not_constant_after_conversion.carbon:[[@LINE+2]]:11-25: error: associated constant V given value `()` that is not constant after conversion to `C` [AssociatedConstantNotConstantAfterConversion]
// CHECK:STDERR:
impl C as I where .V = () {}

// --- fail_monomorphization_failure.carbon

library "[[@TEST_NAME]]";

interface I(N: Core.IntLiteral) {
  let V: array(Self, N);
}

// CHECK:STDERR: fail_monomorphization_failure.carbon:[[@LINE+3]]:24-25: error: member access into facet of incomplete type `I(-1)` [IncompleteTypeInMemberAccessOfFacet]
// CHECK:STDERR: fail_monomorphization_failure.carbon:[[@LINE-4]]:16-19: note: array bound of -1 is negative [ArrayBoundNegative]
// CHECK:STDERR:
impl {} as I(-1) where .V = () {}

// --- fail_todo_constrained_fn.carbon

library "[[@TEST_NAME]]";

interface I {
  let V: Self;
}

fn F(generic T: I where {} impls Core.ImplicitAs(.Self) and .V = {});

fn CallF() {
  // TODO: This call should eventually work.
  // CHECK:STDERR: fail_todo_constrained_fn.carbon:[[@LINE+3]]:3-7: error: cannot convert type `{}` into type implementing `I where {} impls Core.ImplicitAs(.Self) and .(I.V) = {}` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_todo_constrained_fn.carbon:[[@LINE-5]]:14-67: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F({});
}


// CHECK:STDOUT: --- assoc_const_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.c84: type = facet_access_type %Self.37e [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.660: %I.assoc_type = assoc_entity element0, @I.WithSelf.%V [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %.f01: <witness> = impl_self_witness %.Self.frozen, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4da: %.Self.frozen.as_type = impl_witness_access %.f01, element0 [symbolic_self]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.00f: <witness> = impl_self_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ca3: %.Self.as_type = impl_witness_access %.00f, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type.b84: type = facet_type <@I where %impl.elem0.ca3 = %empty_struct> [concrete]
// CHECK:STDOUT:   %.aaf: <witness> = impl_self_witness %empty_struct_type, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.57e: <witness> = impl_witness @empty_struct_type.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_struct.type.facet: %type = facet_value %empty_struct_type, () [concrete]
// CHECK:STDOUT:   %I.facet.f72: %I.type = facet_value %empty_struct_type, (%I.impl_witness.57e) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %I_where.type.8e4: type = facet_type <@I where %impl.elem0.ca3 = %int_0.5c6> [concrete]
// CHECK:STDOUT:   %.2a7: <witness> = impl_self_witness %i32, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.8f5: <witness> = impl_witness @i32.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Int.type.facet: %type = facet_value %i32, () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %I.facet.190: %I.type = facet_value %i32, (%I.impl_witness.8f5) [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %.loc8_7.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.660]
// CHECK:STDOUT:     %impl.elem0.loc8_20.1: %.Self.frozen.as_type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.4da]
// CHECK:STDOUT:     %.loc8_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_26.2: %empty_struct_type = converted %.loc8_26.1, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %rewrite.loc8_23.1 = requirement_rewrite %impl.elem0.loc8_20.1, %.loc8_26.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc8_20.2: %.Self.as_type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.ca3]
// CHECK:STDOUT:     %rewrite.loc8_23.2 = requirement_rewrite %impl.elem0.loc8_20.2, %.loc8_26.2 [concrete]
// CHECK:STDOUT:     %.loc8_14: type = where_expr [concrete = constants.%I_where.type.b84] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc8_23.2 = requirement_rewrite %impl.elem0.loc8_20.2, %.loc8_26.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = impl_self_witness @empty_struct_type.as.I.impl.%.loc8_7.2, @I [concrete = constants.%.aaf]
// CHECK:STDOUT:   impl_decl @i32.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc10_21: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.660]
// CHECK:STDOUT:     %impl.elem0.loc10_21.1: %.Self.frozen.as_type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.4da]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %rewrite.loc10_24.1 = requirement_rewrite %impl.elem0.loc10_21.1, %int_0 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc10_21.2: %.Self.as_type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.ca3]
// CHECK:STDOUT:     %rewrite.loc10_24.2 = requirement_rewrite %impl.elem0.loc10_21.2, %int_0 [concrete]
// CHECK:STDOUT:     %.loc10_15.1: type = where_expr [concrete = constants.%I_where.type.8e4] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc10_24.2 = requirement_rewrite %impl.elem0.loc10_21.2, %int_0 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: <witness> = impl_self_witness @i32.as.I.impl.%i32, @I [concrete = constants.%.2a7]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %V: @I.WithSelf.%Self.as_type (%Self.as_type.c84) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%V [concrete = constants.%assoc0.660]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .V = @V.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%V)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_struct_type.as.I.impl: %.loc8_7.2 as %I.ref {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @empty_struct_type.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.57e]
// CHECK:STDOUT:   %impl_witness_assoc_constant: %empty_struct_type = impl_witness_assoc_constant constants.%empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.as.I.impl: %i32 as %I.ref {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @i32.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.8f5]
// CHECK:STDOUT:   %impl.elem0.loc10_15: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc10_15.1: <bound method> = bound_method constants.%int_0.5c6, %impl.elem0.loc10_15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc10_15, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_15.2: <bound method> = bound_method constants.%int_0.5c6, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc10_15.2(constants.%int_0.5c6) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc10_15.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc10_15.3: %i32 = converted constants.%int_0.5c6, %.loc10_15.2 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %impl_witness_assoc_constant: %i32 = impl_witness_assoc_constant constants.%int_0.3c0 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.c84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.frozen) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%.Self.frozen
// CHECK:STDOUT:   %Self.as_type => constants.%.Self.frozen.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%empty_struct.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%empty_struct.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%empty_struct_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.f72) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.f72
// CHECK:STDOUT:   %Self.as_type => constants.%empty_struct_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Int.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Int.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.190) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.190
// CHECK:STDOUT:   %Self.as_type => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_assoc_const_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.c84: type = facet_access_type %Self.37e [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.660: %I.assoc_type = assoc_entity element0, @I.WithSelf.%V [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %.f01: <witness> = impl_self_witness %.Self.frozen, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4da: %.Self.frozen.as_type = impl_witness_access %.f01, element0 [symbolic_self]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.00f: <witness> = impl_self_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ca3: %.Self.as_type = impl_witness_access %.00f, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0.ca3 = %int_0> [concrete]
// CHECK:STDOUT:   %.aaf: <witness> = impl_self_witness %empty_struct_type, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @empty_struct_type.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_struct.type.facet: %type = facet_value %empty_struct_type, () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %empty_struct_type, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %.loc11_7.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc11_7.2: type = converted %.loc11_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc11_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.660]
// CHECK:STDOUT:     %impl.elem0.loc11_20.1: %.Self.frozen.as_type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.4da]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %rewrite.loc11_23.1 = requirement_rewrite %impl.elem0.loc11_20.1, %int_0 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc11_20.2: %.Self.as_type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.ca3]
// CHECK:STDOUT:     %rewrite.loc11_23.2 = requirement_rewrite %impl.elem0.loc11_20.2, %int_0 [concrete]
// CHECK:STDOUT:     %.loc11_14.1: type = where_expr [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc11_23.2 = requirement_rewrite %impl.elem0.loc11_20.2, %int_0 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <witness> = impl_self_witness @empty_struct_type.as.I.impl.%.loc11_7.2, @I [concrete = constants.%.aaf]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %V: @I.WithSelf.%Self.as_type (%Self.as_type.c84) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%V [concrete = constants.%assoc0.660]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .V = @V.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%V)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_struct_type.as.I.impl: %.loc11_7.2 as %I.ref {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @empty_struct_type.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %.loc11_14.2: %empty_struct_type = converted constants.%int_0, <error> [concrete = <error>]
// CHECK:STDOUT:   %impl_witness_assoc_constant: <error> = impl_witness_assoc_constant <error> [concrete = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.c84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.frozen) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%.Self.frozen
// CHECK:STDOUT:   %Self.as_type => constants.%.Self.frozen.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%empty_struct.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%empty_struct.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%empty_struct_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type => constants.%empty_struct_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_assoc_const_not_constant_after_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.c84: type = facet_access_type %Self.37e [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.660: %I.assoc_type = assoc_entity element0, @I.WithSelf.%V [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.dee: type = facet_type <@ImplicitAs, @ImplicitAs(%C)> [concrete]
// CHECK:STDOUT:   %.5c5: <witness> = impl_self_witness %empty_tuple.type, @ImplicitAs, @ImplicitAs(%C) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness @empty_tuple.type.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %self.param_patt.154: %pattern_type.cb1 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.462: %pattern_type.cb1 = wrapper_binding_pattern self, %self.param_patt.154 [concrete]
// CHECK:STDOUT:   %.a44: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %return.param_patt.89a: %pattern_type.98b = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e4a: %pattern_type.98b = return_slot_pattern %return.param_patt.89a, %C [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert.type: type = fn_type @empty_tuple.type.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert: %empty_tuple.type.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.dee = facet_value %empty_tuple.type, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.1f2: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%C, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %.f01: <witness> = impl_self_witness %.Self.frozen, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4da: %.Self.frozen.as_type = impl_witness_access %.f01, element0 [symbolic_self]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.00f: <witness> = impl_self_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ca3: %.Self.as_type = impl_witness_access %.00f, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0.ca3 = %empty_tuple> [concrete]
// CHECK:STDOUT:   %.b7f: <witness> = impl_self_witness %C, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C.type.facet: %type = facet_value %C, () [concrete]
// CHECK:STDOUT:   %.c8c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.1f2, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %empty_tuple, %empty_tuple.type.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %.loc10_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc10_7.2: type = converted %.loc10_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%C)> [concrete = constants.%ImplicitAs.type.dee]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: <witness> = impl_self_witness @empty_tuple.type.as.ImplicitAs.impl.%.loc10_7.2, @ImplicitAs, @ImplicitAs(constants.%C) [concrete = constants.%.5c5]
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc16_19: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.660]
// CHECK:STDOUT:     %impl.elem0.loc16_19.1: %.Self.frozen.as_type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.4da]
// CHECK:STDOUT:     %.loc16_25.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc16_25.2: %empty_tuple.type = converted %.loc16_25.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %rewrite.loc16_22.1 = requirement_rewrite %impl.elem0.loc16_19.1, %.loc16_25.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc16_19.2: %.Self.as_type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.ca3]
// CHECK:STDOUT:     %rewrite.loc16_22.2 = requirement_rewrite %impl.elem0.loc16_19.2, %.loc16_25.2 [concrete]
// CHECK:STDOUT:     %.loc16_13.1: type = where_expr [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc16_22.2 = requirement_rewrite %impl.elem0.loc16_19.2, %.loc16_25.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16: <witness> = impl_self_witness @C.as.I.impl.%C.ref, @I [concrete = constants.%.b7f]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %V: @I.WithSelf.%Self.as_type (%Self.as_type.c84) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%V [concrete = constants.%assoc0.660]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .V = @V.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%V)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.ImplicitAs.impl: %.loc10_7.2 as %ImplicitAs.type {
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert.decl: %empty_tuple.type.as.ImplicitAs.impl.Convert.type = fn_decl @empty_tuple.type.as.ImplicitAs.impl.Convert [concrete = constants.%empty_tuple.type.as.ImplicitAs.impl.Convert] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern [concrete = constants.%self.param_patt.154]
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.462]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98b = out_param_pattern [concrete = constants.%return.param_patt.89a]
// CHECK:STDOUT:     %return.patt: %pattern_type.98b = return_slot_pattern %return.param_patt, %C.ref [concrete = constants.%return.patt.e4a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc11_34: Core.Form = init_form %C.ref [concrete = constants.%.a44]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc11_28.1: type = splice_block %.loc11_28.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc11_28.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc11_28.3: type = converted %.loc11_28.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %empty_tuple.type = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%empty_tuple.type.as.ImplicitAs.impl.Convert.decl), @empty_tuple.type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Convert = %empty_tuple.type.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref as %I.ref {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %impl.elem0.loc16_13: %.c8c = impl_witness_access constants.%ImplicitAs.impl_witness, element0 [concrete = constants.%empty_tuple.type.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method constants.%empty_tuple, %impl.elem0.loc16_13 [concrete = constants.%empty_tuple.type.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %.loc16_13.2: ref %C = temporary_storage
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert.call: init %C to %.loc16_13.2 = call %bound_method(constants.%empty_tuple)
// CHECK:STDOUT:   %.loc16_13.3: init %C = converted constants.%empty_tuple, %empty_tuple.type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc16_13.4: ref %C = temporary %.loc16_13.2, %.loc16_13.3
// CHECK:STDOUT:   %.loc16_13.5: %C = acquire_value %.loc16_13.4
// CHECK:STDOUT:   %impl_witness_assoc_constant: <error> = impl_witness_assoc_constant <error> [concrete = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %I.ref
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
// CHECK:STDOUT: fn @empty_tuple.type.as.ImplicitAs.impl.Convert(%self.param: %empty_tuple.type) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_46.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_46.2: init %C to %return.param = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc11_47: init %C = converted %.loc11_46.1, %.loc11_46.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   return %.loc11_47 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.c84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.frozen) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%.Self.frozen
// CHECK:STDOUT:   %Self.as_type => constants.%.Self.frozen.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%C.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_monomorphization_failure.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %I.type.335: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.ac2: type = facet_type <@I, @I(%N)> [symbolic]
// CHECK:STDOUT:   %Self.4d2: %I.type.ac2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.4c1: type = facet_access_type %Self.4d2 [symbolic]
// CHECK:STDOUT:   %array_type: type = array_type %N, %Self.as_type.4c1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.16e: type = assoc_entity_type @I, @I(%N) [symbolic]
// CHECK:STDOUT:   %assoc0.377: %I.assoc_type.16e = assoc_entity element0, @I.WithSelf.%V [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness imports.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value Core.IntLiteral, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.e85: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.758: type = fn_type_with_self_type %Negate.WithSelf.Op.type.e85, %Negate.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.type: type = fn_type @Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op: %Core.IntLiteral.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.bound: <bound method> = bound_method %int_1, %Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [concrete]
// CHECK:STDOUT:   %I.type.ce3: type = facet_type <@I, @I(%int_-1)> [concrete]
// CHECK:STDOUT:   %.Self.frozen.978: %I.type.ce3 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.4a1: %I.type.ce3 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.6c2: type = assoc_entity_type @I, @I(%int_-1) [concrete]
// CHECK:STDOUT:   %assoc0.2f4: %I.assoc_type.6c2 = assoc_entity element0, @I.WithSelf.%V [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.978 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Negate = %Core.Negate
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.Negate: type = import_ref Core//prelude/parts/int_literal, Negate, loaded [concrete = constants.%Negate.type]
// CHECK:STDOUT:   %Core.import_ref.abd = import_ref Core//prelude/parts/int_literal, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.9c3: %Core.IntLiteral.as.Negate.impl.Op.type = import_ref Core//prelude/parts/int_literal, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.abd, %Core.import_ref.9c3), @Core.IntLiteral.as.Negate.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: %I.type.335 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %N.patt.loc4_14.1: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_14.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %IntLiteral.ref [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_14.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_14.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.<error>.impl [concrete] {} {
// CHECK:STDOUT:     %.loc11_7.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc11_7.2: type = converted %.loc11_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref: %I.type.335 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:     %impl.elem1: %.758 = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_1, %impl.elem1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.Negate.impl.Op.call: init Core.IntLiteral = call %bound_method(%int_1) [concrete = constants.%int_-1]
// CHECK:STDOUT:     %.loc11_16.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.Negate.impl.Op.call [concrete = constants.%int_-1]
// CHECK:STDOUT:     %.loc11_16.2: Core.IntLiteral = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc11_16.1 [concrete = constants.%int_-1]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%int_-1)> [concrete = constants.%I.type.ce3]
// CHECK:STDOUT:     %.Self.frozen: %I.type.ce3 = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.978]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %I.type [concrete]
// CHECK:STDOUT:     %.Self.ref: %I.type.ce3 = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen.978]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc11_24: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %V.ref: <error> = name_ref V, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc11_30: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %rewrite = requirement_rewrite %V.ref, <error> [concrete]
// CHECK:STDOUT:     %.loc11_18: type = where_expr [concrete = <error>] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.type [concrete]
// CHECK:STDOUT:       %rewrite = requirement_rewrite %V.ref, <error> [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%N.loc4_14.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.patt.loc4_14.2: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_14.2 (constants.%N.patt)]
// CHECK:STDOUT:   %N.loc4_14.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_14.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%N.loc4_14.1)> [symbolic = %I.type (constants.%I.type.ac2)]
// CHECK:STDOUT:   %Self.loc4_33.2: @I.%I.type (%I.type.ac2) = symbolic_binding Self, 1 [symbolic = %Self.loc4_33.2 (constants.%Self.4d2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_33.1: @I.%I.type (%I.type.ac2) = symbolic_binding Self, 1 [symbolic = %Self.loc4_33.2 (constants.%Self.4d2)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %V: @I.WithSelf.%array_type (%array_type) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:       %assoc0: @I.WithSelf.%I.assoc_type (%I.assoc_type.16e) = assoc_entity element0, @I.WithSelf.%V [symbolic = @I.WithSelf.%assoc0 (constants.%assoc0.377)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_33.1
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .V = @V.%assoc0
// CHECK:STDOUT:     witness = (@I.WithSelf.%V)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_struct_type.as.<error>.impl: %.loc11_7.2 as <error> {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend <error>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%N) {
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%N.patt
// CHECK:STDOUT:   %N.loc4_14.1 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%N, constants.%Self.4d2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%int_-1) {
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%N.patt
// CHECK:STDOUT:   %N.loc4_14.1 => constants.%int_-1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.ce3
// CHECK:STDOUT:   %Self.loc4_33.2 => constants.%Self.4a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%int_-1, constants.%Self.4d2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N => constants.%int_-1
// CHECK:STDOUT:   %I.type => constants.%I.type.ce3
// CHECK:STDOUT:   %Self => constants.%Self.4d2
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.4c1
// CHECK:STDOUT:   %array_type => <error>
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.6c2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_constrained_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.c84: type = facet_access_type %Self.37e [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.660: %I.assoc_type = assoc_entity element0, @I.WithSelf.%V [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.3a0: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.3a0 [symbolic_self]
// CHECK:STDOUT:   %ImplicitAs.type.acb: type = facet_type <@ImplicitAs, @ImplicitAs(%.Self.frozen.as_type)> [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.ced: <witness> = lookup_impl_witness %.Self.frozen.3a0, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.cba: %.Self.frozen.as_type = impl_witness_access %I.lookup_impl_witness.ced, element0 [symbolic_self]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %ImplicitAs.type.418: type = facet_type <@ImplicitAs, @ImplicitAs(%.Self.as_type)> [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.78b: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e15: %.Self.as_type = impl_witness_access %I.lookup_impl_witness.78b, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type.14de72.1: type = facet_type <@I where %empty_struct_type impls @ImplicitAs, @ImplicitAs(%.Self.as_type) and %impl.elem0.e15 = %empty_struct> [concrete]
// CHECK:STDOUT:   %pattern_type.692: type = pattern_type %I_where.type.14de72.1 [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.692 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %I_where.type.14de72.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallF.type: type = fn_type @CallF [concrete]
// CHECK:STDOUT:   %CallF: %CallF.type = struct_value () [concrete]
// CHECK:STDOUT:   %I_where.type.14de72.2: type = facet_type <@I where %empty_struct_type impls @ImplicitAs, @ImplicitAs(%empty_struct_type) and %impl.elem0.e15 = %empty_struct> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %T.patt.loc8_15.1: %pattern_type.692 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_19.1: type = splice_block %.loc8_19.2 [concrete = constants.%I_where.type.14de72.1] {
// CHECK:STDOUT:       %.Self.frozen.loc8_15: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self.frozen.loc8_19: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %.loc8_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:       %.Self.ref.loc8_50: %I.type = name_ref .Self, %.Self.frozen.loc8_19 [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %.Self.as_type.loc8_55: type = facet_access_type %.Self.ref.loc8_50 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc8_55: type = converted %.Self.ref.loc8_50, %.Self.as_type.loc8_55 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %ImplicitAs.type.loc8_55.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%.Self.frozen.as_type)> [symbolic_self = constants.%ImplicitAs.type.acb]
// CHECK:STDOUT:       %.loc8_26.2: type = converted %.loc8_26.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %impls.loc8_28.1 = requirement_impls %.loc8_26.2, %ImplicitAs.type.loc8_55.1 [concrete]
// CHECK:STDOUT:       %.Self.ref.loc8_61: %I.type = name_ref .Self, %.Self.frozen.loc8_19 [symbolic_self = constants.%.Self.frozen.3a0]
// CHECK:STDOUT:       %.Self.as_type.loc8_61: type = facet_access_type %.Self.ref.loc8_61 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc8_61: type = converted %.Self.ref.loc8_61, %.Self.as_type.loc8_61 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.660]
// CHECK:STDOUT:       %impl.elem0.loc8_61.1: %.Self.frozen.as_type = impl_witness_access constants.%I.lookup_impl_witness.ced, element0 [symbolic_self = constants.%impl.elem0.cba]
// CHECK:STDOUT:       %.loc8_67.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc8_67.2: %empty_struct_type = converted %.loc8_67.1, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %rewrite.loc8_64.1 = requirement_rewrite %impl.elem0.loc8_61.1, %.loc8_67.2 [concrete]
// CHECK:STDOUT:       %ImplicitAs.type.loc8_55.2: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%.Self.as_type)> [symbolic_self = constants.%ImplicitAs.type.418]
// CHECK:STDOUT:       %impls.loc8_28.2 = requirement_impls %.loc8_26.2, %ImplicitAs.type.loc8_55.2 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc8_61.2: %.Self.as_type = impl_witness_access constants.%I.lookup_impl_witness.78b, element0 [symbolic_self = constants.%impl.elem0.e15]
// CHECK:STDOUT:       %rewrite.loc8_64.2 = requirement_rewrite %impl.elem0.loc8_61.2, %.loc8_67.2 [concrete]
// CHECK:STDOUT:       %.loc8_19.2: type = where_expr [concrete = constants.%I_where.type.14de72.1] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:         %impls.loc8_28.2 = requirement_impls %.loc8_26.2, %ImplicitAs.type.loc8_55.2 [concrete]
// CHECK:STDOUT:         %rewrite.loc8_64.2 = requirement_rewrite %impl.elem0.loc8_61.2, %.loc8_67.2 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_15.2: %I_where.type.14de72.1 = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallF.decl: %CallF.type = fn_decl @CallF [concrete = constants.%CallF] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %V: @I.WithSelf.%Self.as_type (%Self.as_type.c84) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%V [concrete = constants.%assoc0.660]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .V = @V.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%V)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_15.2: %I_where.type.14de72.1) {
// CHECK:STDOUT:   %T.patt.loc8_15.2: %pattern_type.692 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_15.1: %I_where.type.14de72.1 = symbolic_binding T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc15: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where constants.%empty_struct_type impls @ImplicitAs, @ImplicitAs(constants.%empty_struct_type) and constants.%impl.elem0.e15 = constants.%empty_struct> [concrete = constants.%I_where.type.14de72.2]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.c84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.frozen.3a0) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%.Self.frozen.3a0
// CHECK:STDOUT:   %Self.as_type => constants.%.Self.frozen.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
