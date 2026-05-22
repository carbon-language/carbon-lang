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
// CHECK:STDERR: fail_assoc_const_mismatch.carbon:[[@LINE+4]]:12: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs({})` [MissingImplInMemberAccessInContext]
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
  fn Convert[unused self: ()]() -> C { return {}; }
}

// CHECK:STDERR: fail_assoc_const_not_constant_after_conversion.carbon:[[@LINE+4]]:11: error: associated constant V given value `()` that is not constant after conversion to `C` [AssociatedConstantNotConstantAfterConversion]
// CHECK:STDERR: impl C as I where .V = () {}
// CHECK:STDERR:           ^~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C as I where .V = () {}

// --- fail_monomorphization_failure.carbon

library "[[@TEST_NAME]]";

interface I(N:! Core.IntLiteral()) {
  let V:! array(Self, N);
}

// CHECK:STDERR: fail_monomorphization_failure.carbon:[[@LINE+7]]:24: error: member access into facet of incomplete type `I(-1)` [IncompleteTypeInMemberAccessOfFacet]
// CHECK:STDERR: impl {} as I(-1) where .V = () {}
// CHECK:STDERR:                        ^~
// CHECK:STDERR: fail_monomorphization_failure.carbon:[[@LINE-6]]:17: note: array bound of -1 is negative [ArrayBoundNegative]
// CHECK:STDERR:   let V:! array(Self, N);
// CHECK:STDERR:                 ^~~~
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
  // CHECK:STDERR: fail_todo_constrained_fn.carbon:[[@LINE+7]]:3: error: cannot convert type `{}` into type implementing `I where {} impls Core.ImplicitAs(.Self) and .(I.V) = {}` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   F({});
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_todo_constrained_fn.carbon:[[@LINE-7]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F(T:! I where {} impls Core.ImplicitAs(.Self) and .V = {});
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F({});
}

// CHECK:STDOUT: --- assoc_const_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.d5e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.704: type = facet_access_type %Self.d5e [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.279: %I.assoc_type = assoc_entity element0, @I.WithSelf.%V [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %.Self.as_type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type.13c: type = facet_type <@I where %impl.elem0 = %empty_struct> [concrete]
// CHECK:STDOUT:   %I.impl_witness.7f4: <witness> = impl_witness @empty_struct_type.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_struct.type.facet: %type = facet_value %empty_struct_type, () [concrete]
// CHECK:STDOUT:   %I.facet.43a: %I.type = facet_value %empty_struct_type, (%I.impl_witness.7f4) [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %I_where.type.4c3: type = facet_type <@I where %impl.elem0 = %int_0.5c6> [concrete]
// CHECK:STDOUT:   %I.impl_witness.312: <witness> = impl_witness @i32.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Int.type.facet: %type = facet_value %i32, () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.155: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %I.facet.ad2: %I.type = facet_value %i32, (%I.impl_witness.312) [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.279]
// CHECK:STDOUT:     %impl.elem0: %.Self.as_type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc8_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_26.2: %empty_struct_type = converted %.loc8_26.1, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_14: type = where_expr [concrete = constants.%I_where.type.13c] {
// CHECK:STDOUT:       requirement_base_facet_type %I.ref
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc8_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @i32.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc10_21: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.279]
// CHECK:STDOUT:     %impl.elem0.loc10_21: %.Self.as_type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %.loc10_15.1: type = where_expr [concrete = constants.%I_where.type.4c3] {
// CHECK:STDOUT:       requirement_base_facet_type %I.ref
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10_21, %int_0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d5e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %V: @I.WithSelf.%Self.as_type (%Self.as_type.704) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%V [concrete = constants.%assoc0.279]
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
// CHECK:STDOUT: impl @empty_struct_type.as.I.impl: %.loc8_7.2 as %.loc8_14 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @empty_struct_type.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.7f4]
// CHECK:STDOUT:   %impl_witness_assoc_constant: %empty_struct_type = impl_witness_assoc_constant constants.%empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %.loc8_14
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.as.I.impl: %i32 as %.loc10_15.1 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @i32.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.312]
// CHECK:STDOUT:   %impl.elem0.loc10_15: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc10_15.1: <bound method> = bound_method constants.%int_0.5c6, %impl.elem0.loc10_15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc10_15, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_15.2: <bound method> = bound_method constants.%int_0.5c6, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc10_15.2(constants.%int_0.5c6) [concrete = constants.%int_0.155]
// CHECK:STDOUT:   %.loc10_15.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.155]
// CHECK:STDOUT:   %.loc10_15.3: %i32 = converted constants.%int_0.5c6, %.loc10_15.2 [concrete = constants.%int_0.155]
// CHECK:STDOUT:   %impl_witness_assoc_constant: %i32 = impl_witness_assoc_constant constants.%int_0.155 [concrete = constants.%int_0.155]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %.loc10_15.1
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.d5e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.d5e
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.704
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%.Self
// CHECK:STDOUT:   %Self.as_type => constants.%.Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%empty_struct.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%empty_struct.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%empty_struct_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.43a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.43a
// CHECK:STDOUT:   %Self.as_type => constants.%empty_struct_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Int.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Int.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.ad2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.ad2
// CHECK:STDOUT:   %Self.as_type => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_assoc_const_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.a79: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.94d: type = facet_access_type %Self.a79 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.a69: %I.assoc_type = assoc_entity element0, @I.WithSelf.%V [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %.Self.as_type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %int_0> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @empty_struct_type.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_struct.type.facet: %type = facet_value %empty_struct_type, () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %empty_struct_type, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %.loc15_7.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc15_7.2: type = converted %.loc15_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc15_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.a69]
// CHECK:STDOUT:     %impl.elem0: %.Self.as_type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %.loc15_14.1: type = where_expr [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type %I.ref
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %int_0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.a79]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %V: @I.WithSelf.%Self.as_type (%Self.as_type.94d) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%V [concrete = constants.%assoc0.a69]
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
// CHECK:STDOUT: impl @empty_struct_type.as.I.impl: %.loc15_7.2 as %.loc15_14.1 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @empty_struct_type.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %.loc15_14.2: %empty_struct_type = converted constants.%int_0, <error> [concrete = <error>]
// CHECK:STDOUT:   %impl_witness_assoc_constant: <error> = impl_witness_assoc_constant <error> [concrete = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %.loc15_14.1
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.a79) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.a79
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.94d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%.Self
// CHECK:STDOUT:   %Self.as_type => constants.%.Self.as_type
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
// CHECK:STDOUT:   %Self.214: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.38e: type = facet_access_type %Self.214 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.8e6: %I.assoc_type = assoc_entity element0, @I.WithSelf.%V [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.401: type = facet_type <@ImplicitAs, @ImplicitAs(%C)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness @empty_tuple.type.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %.f7b: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.bde: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert.type: type = fn_type @empty_tuple.type.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert: %empty_tuple.type.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.401 = facet_value %empty_tuple.type, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.076: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%C, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %.Self.as_type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_tuple> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C.type.facet: %type = facet_value %C, () [concrete]
// CHECK:STDOUT:   %.2ef: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.076, %ImplicitAs.facet [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.649 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%C)> [concrete = constants.%ImplicitAs.type.401]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc18_19: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.8e6]
// CHECK:STDOUT:     %impl.elem0.loc18_19: %.Self.as_type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc18_25.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc18_25.2: %empty_tuple.type = converted %.loc18_25.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc18_13.1: type = where_expr [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type %I.ref
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc18_19, %.loc18_25.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.214]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %V: @I.WithSelf.%Self.as_type (%Self.as_type.38e) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%V [concrete = constants.%assoc0.8e6]
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.bde = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.bde = return_slot_pattern %return.param_patt, %C.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc11_36: Core.Form = init_form %C.ref [concrete = constants.%.f7b]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc11_28.1: type = splice_block %.loc11_28.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc11_28.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc11_28.3: type = converted %.loc11_28.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %empty_tuple.type = value_binding self, %self.param
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
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref as %.loc18_13.1 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %impl.elem0.loc18_13: %.2ef = impl_witness_access constants.%ImplicitAs.impl_witness, element0 [concrete = constants.%empty_tuple.type.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method constants.%empty_tuple, %impl.elem0.loc18_13 [concrete = constants.%empty_tuple.type.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %.loc18_13.2: ref %C = temporary_storage
// CHECK:STDOUT:   %empty_tuple.type.as.ImplicitAs.impl.Convert.call: init %C to %.loc18_13.2 = call %bound_method(constants.%empty_tuple)
// CHECK:STDOUT:   %.loc18_13.3: init %C = converted constants.%empty_tuple, %empty_tuple.type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc18_13.4: ref %C = temporary %.loc18_13.2, %.loc18_13.3
// CHECK:STDOUT:   %.loc18_13.5: %C = acquire_value %.loc18_13.4
// CHECK:STDOUT:   %impl_witness_assoc_constant: <error> = impl_witness_assoc_constant <error> [concrete = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %.loc18_13.1
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
// CHECK:STDOUT:   %.loc11_48.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_48.2: init %C to %return.param = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc11_49: init %C = converted %.loc11_48.1, %.loc11_48.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   return %.loc11_49 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.214) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.214
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.38e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%.Self
// CHECK:STDOUT:   %Self.as_type => constants.%.Self.as_type
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
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %I.type.1e5: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.1e5 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.432: type = facet_type <@I, @I(%N)> [symbolic]
// CHECK:STDOUT:   %Self.560: %I.type.432 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.48f: type = facet_access_type %Self.560 [symbolic]
// CHECK:STDOUT:   %array_type: type = array_type %N, %Self.as_type.48f [symbolic]
// CHECK:STDOUT:   %I.assoc_type.0c7: type = assoc_entity_type @I, @I(%N) [symbolic]
// CHECK:STDOUT:   %assoc0.f5b: %I.assoc_type.0c7 = assoc_entity element0, @I.WithSelf.%V [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness imports.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value Core.IntLiteral, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.40e: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.279: type = fn_type_with_self_type %Negate.WithSelf.Op.type.40e, %Negate.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.type: type = fn_type @Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op: %Core.IntLiteral.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.bound: <bound method> = bound_method %int_1, %Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [concrete]
// CHECK:STDOUT:   %I.type.33b: type = facet_type <@I, @I(%int_-1)> [concrete]
// CHECK:STDOUT:   %.Self.67a: %I.type.33b = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.07a: %I.type.33b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.4af: type = assoc_entity_type @I, @I(%int_-1) [concrete]
// CHECK:STDOUT:   %assoc0.ab1: %I.assoc_type.4af = assoc_entity element0, @I.WithSelf.%V [concrete]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.67a [symbolic_self]
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
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Negate: type = import_ref Core//prelude/parts/int_literal, Negate, loaded [concrete = constants.%Negate.type]
// CHECK:STDOUT:   %Core.import_ref.abd = import_ref Core//prelude/parts/int_literal, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.968: %Core.IntLiteral.as.Negate.impl.Op.type = import_ref Core//prelude/parts/int_literal, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.abd, %Core.import_ref.968), @Core.IntLiteral.as.Negate.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: %I.type.1e5 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_33.1: type = splice_block %.loc4_33.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_33.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_33.3: type = converted %IntLiteral.call, %.loc4_33.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_14.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_14.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.<error>.impl [concrete] {} {
// CHECK:STDOUT:     %.loc15_7.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc15_7.2: type = converted %.loc15_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.ref: %I.type.1e5 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:     %impl.elem1: %.279 = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_1, %impl.elem1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.Negate.impl.Op.call: init Core.IntLiteral = call %bound_method(%int_1) [concrete = constants.%int_-1]
// CHECK:STDOUT:     %.loc15_16.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.Negate.impl.Op.call [concrete = constants.%int_-1]
// CHECK:STDOUT:     %.loc15_16.2: Core.IntLiteral = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc15_16.1 [concrete = constants.%int_-1]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%int_-1)> [concrete = constants.%I.type.33b]
// CHECK:STDOUT:     %.Self: %I.type.33b = symbolic_binding .Self [symbolic_self = constants.%.Self.67a]
// CHECK:STDOUT:     %.Self.ref: %I.type.33b = name_ref .Self, %.Self [symbolic_self = constants.%.Self.67a]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc15_24: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %V.ref: <error> = name_ref V, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc15_30: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc15_18: type = where_expr [concrete = <error>] {
// CHECK:STDOUT:       requirement_base_facet_type %I.type
// CHECK:STDOUT:       requirement_rewrite %V.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%N.loc4_14.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc4_14.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_14.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%N.loc4_14.1)> [symbolic = %I.type (constants.%I.type.432)]
// CHECK:STDOUT:   %Self.loc4_36.2: @I.%I.type (%I.type.432) = symbolic_binding Self, 1 [symbolic = %Self.loc4_36.2 (constants.%Self.560)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_36.1: @I.%I.type (%I.type.432) = symbolic_binding Self, 1 [symbolic = %Self.loc4_36.2 (constants.%Self.560)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %V: @I.WithSelf.%array_type (%array_type) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:       %assoc0: @I.WithSelf.%I.assoc_type (%I.assoc_type.0c7) = assoc_entity element0, @I.WithSelf.%V [symbolic = @I.WithSelf.%assoc0 (constants.%assoc0.f5b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_36.1
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .V = @V.%assoc0
// CHECK:STDOUT:     witness = (@I.WithSelf.%V)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_struct_type.as.<error>.impl: %.loc15_7.2 as <error> {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend <error>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%N) {
// CHECK:STDOUT:   %N.loc4_14.1 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%N, constants.%Self.560) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%int_-1) {
// CHECK:STDOUT:   %N.loc4_14.1 => constants.%int_-1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.33b
// CHECK:STDOUT:   %Self.loc4_36.2 => constants.%Self.07a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%int_-1, constants.%Self.560) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N => constants.%int_-1
// CHECK:STDOUT:   %I.type => constants.%I.type.33b
// CHECK:STDOUT:   %Self => constants.%Self.560
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.48f
// CHECK:STDOUT:   %array_type => <error>
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.4af
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ab1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_constrained_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.a1e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.521: type = facet_access_type %Self.a1e [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.a9a: %I.assoc_type = assoc_entity element0, @I.WithSelf.%V [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.908: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.908 [symbolic_self]
// CHECK:STDOUT:   %ImplicitAs.type.7ec: type = facet_type <@ImplicitAs, @ImplicitAs(%.Self.as_type)> [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.5fc: <witness> = lookup_impl_witness %.Self.908, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %.Self.as_type = impl_witness_access %I.lookup_impl_witness.5fc, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %empty_struct_type impls @ImplicitAs, @ImplicitAs(%.Self.as_type) and %impl.elem0 = %empty_struct> [symbolic_self]
// CHECK:STDOUT:   %pattern_type.010: type = pattern_type %I_where.type [symbolic_self]
// CHECK:STDOUT:   %T: %I_where.type = symbolic_binding T, 0 [symbolic]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.010 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_12.1: type = splice_block %.loc8_12.2 [symbolic_self = constants.%I_where.type] {
// CHECK:STDOUT:       %.Self.loc8_7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %.Self.loc8_12: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.908]
// CHECK:STDOUT:       %.loc8_19.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %ImplicitAs.ref: %ImplicitAs.type.649 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:       %.Self.ref.loc8_43: %I.type = name_ref .Self, %.Self.loc8_12 [symbolic_self = constants.%.Self.908]
// CHECK:STDOUT:       %.Self.as_type.loc8_48: type = facet_access_type %.Self.ref.loc8_43 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc8_48: type = converted %.Self.ref.loc8_43, %.Self.as_type.loc8_48 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%.Self.as_type)> [symbolic_self = constants.%ImplicitAs.type.7ec]
// CHECK:STDOUT:       %.loc8_19.2: type = converted %.loc8_19.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.Self.ref.loc8_54: %I.type = name_ref .Self, %.Self.loc8_12 [symbolic_self = constants.%.Self.908]
// CHECK:STDOUT:       %.Self.as_type.loc8_54: type = facet_access_type %.Self.ref.loc8_54 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc8_54: type = converted %.Self.ref.loc8_54, %.Self.as_type.loc8_54 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %V.ref: %I.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0.a9a]
// CHECK:STDOUT:       %impl.elem0: %.Self.as_type = impl_witness_access constants.%I.lookup_impl_witness.5fc, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc8_60.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc8_60.2: %empty_struct_type = converted %.loc8_60.1, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc8_12.2: type = where_expr [symbolic_self = constants.%I_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type %I.ref
// CHECK:STDOUT:         requirement_impls %.loc8_19.2, %ImplicitAs.type
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %.loc8_60.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_7.2: %I_where.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallF.decl: %CallF.type = fn_decl @CallF [concrete = constants.%CallF] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.a1e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %V: @I.WithSelf.%Self.as_type (%Self.as_type.521) = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%V [concrete = constants.%assoc0.a9a]
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
// CHECK:STDOUT: generic fn @F(%T.loc8_7.2: %I_where.type) {
// CHECK:STDOUT:   %T.loc8_7.1: %I_where.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc19: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.a1e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.a1e
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.521
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.908) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%.Self.908
// CHECK:STDOUT:   %Self.as_type => constants.%.Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
