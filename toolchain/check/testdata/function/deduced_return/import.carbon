// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/deduced_return/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/deduced_return/import.carbon

// --- lib.carbon
library "[[@TEST_NAME]]";

fn F() -> auto { return 0 as i32; }

class C {
  fn G() -> auto { return 1 as i32; }
}

fn Id[T: Core.Copy](x: T) -> auto { return x; }
fn Pair[T: Core.Copy](x: T) -> auto { return (x, x); }

// --- lib.impl.carbon
impl library "[[@TEST_NAME]]";

fn H() -> i32 { return F(); }
fn J() -> (i32, i32) { return Pair(Id(0 as i32)); }
fn K[U: Core.Copy & Core.Destroy](u: U) -> (U, U) { return Pair(Id(u)); }

// --- use.carbon
library "[[@TEST_NAME]]";

import library "lib";

//@dump-sem-ir-begin
fn I() -> i32 { return F(); }
fn J() -> i32 { return C.G(); }
//@dump-sem-ir-end

// --- use_generic.carbon
library "[[@TEST_NAME]]";

import library "lib";

//@dump-sem-ir-begin
fn J() -> (i32, i32) { return Pair(Id(0 as i32)); }
fn K[U: Core.Copy & Core.Destroy](u: U) -> (U, U) { return Pair(Id(u)); }
//@dump-sem-ir-end

fn L() -> (bool, bool) { return K(true); }

// CHECK:STDOUT: --- use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   %I.type: type = fn_type @I [concrete]
// CHECK:STDOUT:   %I: %I.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.type: type = fn_type @J [concrete]
// CHECK:STDOUT:   %J: %J.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.G.type: type = fn_type @C.G [concrete]
// CHECK:STDOUT:   %C.G: %C.G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.F: %F.type = import_ref Main//lib, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//lib, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.341: %C.G.type = import_ref Main//lib, loc6_18, loaded [concrete = constants.%C.G]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %I.decl: %I.type = fn_decl @I [concrete = constants.%I] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6: Core.Form = init_form %i32 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: %J.type = fn_decl @J [concrete = constants.%J] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7: Core.Form = init_form %i32 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.F [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref()
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %G.ref: %C.G.type = name_ref G, imports.%Main.import_ref.341 [concrete = constants.%C.G]
// CHECK:STDOUT:   %C.G.call: init %i32 = call %G.ref()
// CHECK:STDOUT:   return %C.G.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.be9: %tuple.type.693 = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.35f: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %.30d: Core.Form = init_form %tuple.type.35f [concrete]
// CHECK:STDOUT:   %pattern_type.354: type = pattern_type %tuple.type.35f [concrete]
// CHECK:STDOUT:   %return.param_patt.46d: %pattern_type.354 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.d16: %pattern_type.354 = return_slot_pattern %return.param_patt.46d, %tuple.type.35f [concrete]
// CHECK:STDOUT:   %J.type: type = fn_type @J [concrete]
// CHECK:STDOUT:   %J: %J.type = struct_value () [concrete]
// CHECK:STDOUT:   %Pair.type: type = fn_type @Pair [concrete]
// CHECK:STDOUT:   %Pair: %Pair.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Id.type: type = fn_type @Id [concrete]
// CHECK:STDOUT:   %Id: %Id.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.b93: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.396: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.84a: %Core.IntLiteral.as.As.impl.Convert.type.396 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.e09: <witness> = impl_witness imports.%As.impl_witness_table.1c2, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.2b0: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.94f: %Core.IntLiteral.as.As.impl.Convert.type.2b0 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.b93 = facet_value Core.IntLiteral, (%As.impl_witness.e09) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.2be: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.979: type = fn_type_with_self_type %As.WithSelf.Convert.type.2be, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.As.impl.Convert.94f [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.94f, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.753: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.facet.95d: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Id.specific_fn.04c: <specific function> = specific_function %Id, @Id(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %Pair.specific_fn.24c: <specific function> = specific_function %Pair, @Pair(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.7f7: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.7f7 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.1a6: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.b4e: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.1a6, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %Copy.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Copy & @Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.143: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %U.patt.256: %pattern_type.143 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.dc4: %facet_type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %U.as_type.5d6: type = facet_access_type %U.dc4 [symbolic]
// CHECK:STDOUT:   %pattern_type.0e9: type = pattern_type %U.as_type.5d6 [symbolic]
// CHECK:STDOUT:   %u.param_patt.f9a: %pattern_type.0e9 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %u.patt.d30: %pattern_type.0e9 = wrapper_binding_pattern u, %u.param_patt.f9a [symbolic]
// CHECK:STDOUT:   %tuple.type.56f: type = tuple_type (%facet_type, %facet_type) [concrete]
// CHECK:STDOUT:   %tuple.f81: %tuple.type.56f = tuple_value (%U.dc4, %U.dc4) [symbolic]
// CHECK:STDOUT:   %tuple.type.7d0: type = tuple_type (%U.as_type.5d6, %U.as_type.5d6) [symbolic]
// CHECK:STDOUT:   %.7db: Core.Form = init_form %tuple.type.7d0 [symbolic]
// CHECK:STDOUT:   %pattern_type.c29: type = pattern_type %tuple.type.7d0 [symbolic]
// CHECK:STDOUT:   %return.param_patt.df9: %pattern_type.c29 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.b82: %pattern_type.c29 = return_slot_pattern %return.param_patt.df9, %tuple.type.7d0 [symbolic]
// CHECK:STDOUT:   %K.type: type = fn_type @K [concrete]
// CHECK:STDOUT:   %K: %K.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.091: <witness> = require_complete_type %U.as_type.5d6 [symbolic]
// CHECK:STDOUT:   %require_complete.8f6: <witness> = require_complete_type %tuple.type.7d0 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.fd8: <witness> = lookup_impl_witness %U.dc4, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.40c: %Copy.type = facet_value %U.as_type.5d6, (%Copy.lookup_impl_witness.fd8) [symbolic]
// CHECK:STDOUT:   %Id.specific_fn.e53: <specific function> = specific_function %Id, @Id(%Copy.facet.40c) [symbolic]
// CHECK:STDOUT:   %Pair.specific_fn.540: <specific function> = specific_function %Pair, @Pair(%Copy.facet.40c) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.491: <witness> = lookup_impl_witness %U.dc4, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.589: %Destroy.type = facet_value %U.as_type.5d6, (%Destroy.lookup_impl_witness.491) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.613: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.589) [symbolic]
// CHECK:STDOUT:   %.a48: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.613, %Destroy.facet.589 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.a48 = impl_witness_access %Destroy.lookup_impl_witness.491, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.eba: <specific function> = specific_impl_function %impl.elem2, @Destroy.WithSelf.SelfDestruct.1(%Destroy.facet.589) [symbolic]
// CHECK:STDOUT:   %tuple.type.784: type = tuple_type (bool, bool) [concrete]
// CHECK:STDOUT:   %.7b8: Core.Form = init_form %tuple.type.784 [concrete]
// CHECK:STDOUT:   %pattern_type.860: type = pattern_type %tuple.type.784 [concrete]
// CHECK:STDOUT:   %return.param_patt.3b7: %pattern_type.860 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.fc1: %pattern_type.860 = return_slot_pattern %return.param_patt.3b7, %tuple.type.784 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.5e1: <witness> = impl_witness imports.%Copy.impl_witness_table.bda [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f: <witness> = custom_witness (%Destroy.WithSelf.Op.403, %Destroy.WithSelf.SubobjectDestroy.d01, %Destroy.WithSelf.SelfDestruct.db3), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value bool, (%Copy.impl_witness.5e1, %custom_witness.f8f) [concrete]
// CHECK:STDOUT:   %u.param_patt.56f: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:   %u.patt.0d7: %pattern_type.831 = wrapper_binding_pattern u, %u.param_patt.56f [concrete]
// CHECK:STDOUT:   %tuple.6be: %tuple.type.56f = tuple_value (%facet_value, %facet_value) [concrete]
// CHECK:STDOUT:   %complete_type.127: <witness> = complete_type_witness bool [concrete]
// CHECK:STDOUT:   %complete_type.e0b: <witness> = complete_type_witness %tuple.type.784 [concrete]
// CHECK:STDOUT:   %Copy.facet.7af: %Copy.type = facet_value bool, (%Copy.impl_witness.5e1) [concrete]
// CHECK:STDOUT:   %Id.specific_fn.06a: <specific function> = specific_function %Id, @Id(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %Pair.specific_fn.62b: <specific function> = specific_function %Pair, @Pair(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %Destroy.facet.75f: %Destroy.type = facet_value bool, (%custom_witness.f8f) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.2a2: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.75f) [concrete]
// CHECK:STDOUT:   %.fb9: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.2a2, %Destroy.facet.75f [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Id: %Id.type = import_ref Main//lib, Id, loaded [concrete = constants.%Id]
// CHECK:STDOUT:   %Main.Pair: %Pair.type = import_ref Main//lib, Pair, loaded [concrete = constants.%Pair]
// CHECK:STDOUT:   %Core.048aa3.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.As: %As.type.116 = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.9e0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.396) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.84a)]
// CHECK:STDOUT:   %As.impl_witness_table.1c2 = impl_witness_table (%Core.import_ref.9e0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.a06 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.9c3475.1: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.9c3475.1), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT:   %Core.Bool: type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = bool]
// CHECK:STDOUT:   %Core.import_ref.9c3475.2: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.bda = impl_witness_table (%Core.import_ref.9c3475.2), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %J.decl: %J.type = fn_decl @J [concrete = constants.%J] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.354 = out_param_pattern [concrete = constants.%return.param_patt.46d]
// CHECK:STDOUT:     %return.patt: %pattern_type.354 = return_slot_pattern %return.param_patt, %.loc6_20.3 [concrete = constants.%return.patt.d16]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc6_12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc6_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_20.2: %tuple.type.693 = tuple_literal (%i32.loc6_12, %i32.loc6_17) [concrete = constants.%tuple.be9]
// CHECK:STDOUT:     %.loc6_20.3: type = converted %.loc6_20.2, constants.%tuple.type.35f [concrete = constants.%tuple.type.35f]
// CHECK:STDOUT:     %.loc6_20.4: Core.Form = init_form %.loc6_20.3 [concrete = constants.%.30d]
// CHECK:STDOUT:     %return.param: ref %tuple.type.35f = out_param call_param0
// CHECK:STDOUT:     %return: ref %tuple.type.35f = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %K.decl: %K.type = fn_decl @K [concrete = constants.%K] {
// CHECK:STDOUT:     %U.patt.loc7_7.1: %pattern_type.143 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc7_7.2 (constants.%U.patt.256)]
// CHECK:STDOUT:     %u.param_patt.loc7_36.1: @K.%pattern_type.loc7_36 (%pattern_type.0e9) = value_param_pattern [symbolic = %u.param_patt.loc7_36.2 (constants.%u.param_patt.f9a)]
// CHECK:STDOUT:     %u.patt.loc7_36.1: @K.%pattern_type.loc7_36 (%pattern_type.0e9) = wrapper_binding_pattern u, %u.param_patt.loc7_36.1 [symbolic = %u.patt.loc7_36.2 (constants.%u.patt.d30)]
// CHECK:STDOUT:     %return.param_patt.loc7_49.1: @K.%pattern_type.loc7_49 (%pattern_type.c29) = out_param_pattern [symbolic = %return.param_patt.loc7_49.2 (constants.%return.param_patt.df9)]
// CHECK:STDOUT:     %return.patt.loc7_41.1: @K.%pattern_type.loc7_49 (%pattern_type.c29) = return_slot_pattern %return.param_patt.loc7_49.1, %.loc7_49.6 [symbolic = %return.patt.loc7_41.2 (constants.%return.patt.b82)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc7_45: %facet_type = name_ref U, %U.loc7_7.2 [symbolic = %U.loc7_7.1 (constants.%U.dc4)]
// CHECK:STDOUT:     %U.ref.loc7_48: %facet_type = name_ref U, %U.loc7_7.2 [symbolic = %U.loc7_7.1 (constants.%U.dc4)]
// CHECK:STDOUT:     %.loc7_49.3: %tuple.type.56f = tuple_literal (%U.ref.loc7_45, %U.ref.loc7_48) [symbolic = %tuple (constants.%tuple.f81)]
// CHECK:STDOUT:     %U.as_type.loc7_49.1: type = facet_access_type constants.%U.dc4 [symbolic = %U.as_type.loc7_38.1 (constants.%U.as_type.5d6)]
// CHECK:STDOUT:     %.loc7_49.4: type = converted constants.%U.dc4, %U.as_type.loc7_49.1 [symbolic = %U.as_type.loc7_38.1 (constants.%U.as_type.5d6)]
// CHECK:STDOUT:     %U.as_type.loc7_49.2: type = facet_access_type constants.%U.dc4 [symbolic = %U.as_type.loc7_38.1 (constants.%U.as_type.5d6)]
// CHECK:STDOUT:     %.loc7_49.5: type = converted constants.%U.dc4, %U.as_type.loc7_49.2 [symbolic = %U.as_type.loc7_38.1 (constants.%U.as_type.5d6)]
// CHECK:STDOUT:     %.loc7_49.6: type = converted %.loc7_49.3, constants.%tuple.type.7d0 [symbolic = %tuple.type (constants.%tuple.type.7d0)]
// CHECK:STDOUT:     %.loc7_49.7: Core.Form = init_form %.loc7_49.6 [symbolic = %.loc7_49.2 (constants.%.7db)]
// CHECK:STDOUT:     %.loc7_19.1: type = splice_block %.loc7_19.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref.loc7_9: <namespace> = name_ref Core, imports.%Core.048aa3.1 [concrete = imports.%Core.048aa3.1]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:       %Core.ref.loc7_21: <namespace> = name_ref Core, imports.%Core.048aa3.1 [concrete = imports.%Core.048aa3.1]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:       %impl.elem0: %.b4e = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc7_19: <bound method> = bound_method %Copy.ref, %impl.elem0 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method.loc7_19(%Copy.ref, %Destroy.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc7_19.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc7_19.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc7_19.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_7.2: %facet_type = symbolic_binding U, 0 [symbolic = %U.loc7_7.1 (constants.%U.dc4)]
// CHECK:STDOUT:     %u.param: @K.%U.as_type.loc7_38.1 (%U.as_type.5d6) = value_param call_param0
// CHECK:STDOUT:     %.loc7_38.1: type = splice_block %.loc7_38.2 [symbolic = %U.as_type.loc7_38.1 (constants.%U.as_type.5d6)] {
// CHECK:STDOUT:       %U.ref.loc7_38: %facet_type = name_ref U, %U.loc7_7.2 [symbolic = %U.loc7_7.1 (constants.%U.dc4)]
// CHECK:STDOUT:       %U.as_type.loc7_38.2: type = facet_access_type %U.ref.loc7_38 [symbolic = %U.as_type.loc7_38.1 (constants.%U.as_type.5d6)]
// CHECK:STDOUT:       %.loc7_38.2: type = converted %U.ref.loc7_38, %U.as_type.loc7_38.2 [symbolic = %U.as_type.loc7_38.1 (constants.%U.as_type.5d6)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @K.%U.as_type.loc7_38.1 (%U.as_type.5d6) = wrapper_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @K.%tuple.type (%tuple.type.7d0) = out_param call_param1
// CHECK:STDOUT:     %return: ref @K.%tuple.type (%tuple.type.7d0) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J() -> out %return.param: %tuple.type.35f {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Pair.ref: %Pair.type = name_ref Pair, imports.%Main.Pair [concrete = constants.%Pair]
// CHECK:STDOUT:   %Id.ref: %Id.type = name_ref Id, imports.%Main.Id [concrete = constants.%Id]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %i32.loc6_44: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.979 = impl_witness_access constants.%As.impl_witness.e09, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.94f]
// CHECK:STDOUT:   %bound_method.loc6_41.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_41.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc6_41.2(%int_0) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc6_41.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc6_41.2: %i32 = converted %int_0, %.loc6_41.1 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %Copy.facet.loc6_47.1: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.8af) [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %.loc6_47.1: %Copy.type = converted constants.%i32, %Copy.facet.loc6_47.1 [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %Copy.facet.loc6_47.2: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.8af) [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %.loc6_47.2: %Copy.type = converted constants.%i32, %Copy.facet.loc6_47.2 [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %Id.specific_fn: <specific function> = specific_function %Id.ref, @Id(constants.%Copy.facet.95d) [concrete = constants.%Id.specific_fn.04c]
// CHECK:STDOUT:   %Id.call: init %i32 = call %Id.specific_fn(%.loc6_41.2)
// CHECK:STDOUT:   %Copy.facet.loc6_48.1: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.8af) [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %.loc6_48.1: %Copy.type = converted constants.%i32, %Copy.facet.loc6_48.1 [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %Copy.facet.loc6_48.2: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.8af) [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %.loc6_48.2: %Copy.type = converted constants.%i32, %Copy.facet.loc6_48.2 [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %Pair.specific_fn: <specific function> = specific_function %Pair.ref, @Pair(constants.%Copy.facet.95d) [concrete = constants.%Pair.specific_fn.24c]
// CHECK:STDOUT:   %.loc6_20.1: ref %tuple.type.35f = splice_block %return.param {}
// CHECK:STDOUT:   %.loc6_47.3: %i32 = value_of_initializer %Id.call
// CHECK:STDOUT:   %.loc6_47.4: %i32 = converted %Id.call, %.loc6_47.3
// CHECK:STDOUT:   %Pair.call: init %tuple.type.35f to %.loc6_20.1 = call %Pair.specific_fn(%.loc6_47.4)
// CHECK:STDOUT:   return %Pair.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @K(%U.loc7_7.2: %facet_type) {
// CHECK:STDOUT:   %U.patt.loc7_7.2: %pattern_type.143 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc7_7.2 (constants.%U.patt.256)]
// CHECK:STDOUT:   %U.loc7_7.1: %facet_type = symbolic_binding U, 0 [symbolic = %U.loc7_7.1 (constants.%U.dc4)]
// CHECK:STDOUT:   %U.as_type.loc7_38.1: type = facet_access_type %U.loc7_7.1 [symbolic = %U.as_type.loc7_38.1 (constants.%U.as_type.5d6)]
// CHECK:STDOUT:   %pattern_type.loc7_36: type = pattern_type %U.as_type.loc7_38.1 [symbolic = %pattern_type.loc7_36 (constants.%pattern_type.0e9)]
// CHECK:STDOUT:   %u.param_patt.loc7_36.2: @K.%pattern_type.loc7_36 (%pattern_type.0e9) = value_param_pattern [symbolic = %u.param_patt.loc7_36.2 (constants.%u.param_patt.f9a)]
// CHECK:STDOUT:   %u.patt.loc7_36.2: @K.%pattern_type.loc7_36 (%pattern_type.0e9) = wrapper_binding_pattern u, %u.param_patt.loc7_36.2 [symbolic = %u.patt.loc7_36.2 (constants.%u.patt.d30)]
// CHECK:STDOUT:   %tuple: %tuple.type.56f = tuple_value (%U.loc7_7.1, %U.loc7_7.1) [symbolic = %tuple (constants.%tuple.f81)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%U.as_type.loc7_38.1, %U.as_type.loc7_38.1) [symbolic = %tuple.type (constants.%tuple.type.7d0)]
// CHECK:STDOUT:   %.loc7_49.2: Core.Form = init_form %tuple.type [symbolic = %.loc7_49.2 (constants.%.7db)]
// CHECK:STDOUT:   %pattern_type.loc7_49: type = pattern_type %tuple.type [symbolic = %pattern_type.loc7_49 (constants.%pattern_type.c29)]
// CHECK:STDOUT:   %return.param_patt.loc7_49.2: @K.%pattern_type.loc7_49 (%pattern_type.c29) = out_param_pattern [symbolic = %return.param_patt.loc7_49.2 (constants.%return.param_patt.df9)]
// CHECK:STDOUT:   %return.patt.loc7_41.2: @K.%pattern_type.loc7_49 (%pattern_type.c29) = return_slot_pattern %return.param_patt.loc7_49.2, %tuple.type [symbolic = %return.patt.loc7_41.2 (constants.%return.patt.b82)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_36: <witness> = require_complete_type %U.as_type.loc7_38.1 [symbolic = %require_complete.loc7_36 (constants.%require_complete.091)]
// CHECK:STDOUT:   %require_complete.loc7_41: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc7_41 (constants.%require_complete.8f6)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc7_7.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.fd8)]
// CHECK:STDOUT:   %Copy.facet.loc7_69.3: %Copy.type = facet_value %U.as_type.loc7_38.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc7_69.3 (constants.%Copy.facet.40c)]
// CHECK:STDOUT:   %Id.specific_fn.loc7_65.2: <specific function> = specific_function constants.%Id, @Id(%Copy.facet.loc7_69.3) [symbolic = %Id.specific_fn.loc7_65.2 (constants.%Id.specific_fn.e53)]
// CHECK:STDOUT:   %Pair.specific_fn.loc7_60.2: <specific function> = specific_function constants.%Pair, @Pair(%Copy.facet.loc7_69.3) [symbolic = %Pair.specific_fn.loc7_60.2 (constants.%Pair.specific_fn.540)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc7_7.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.491)]
// CHECK:STDOUT:   %Destroy.facet.loc7_69.3: %Destroy.type = facet_value %U.as_type.loc7_38.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc7_69.3 (constants.%Destroy.facet.589)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.loc7_69.3) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.613)]
// CHECK:STDOUT:   %.loc7_69.8: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %Destroy.facet.loc7_69.3 [symbolic = %.loc7_69.8 (constants.%.a48)]
// CHECK:STDOUT:   %impl.elem2.loc7_69.2: @K.%.loc7_69.8 (%.a48) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc7_69.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_69.2: <specific function> = specific_impl_function %impl.elem2.loc7_69.2, @Destroy.WithSelf.SelfDestruct.1(%Destroy.facet.loc7_69.3) [symbolic = %specific_impl_fn.loc7_69.2 (constants.%specific_impl_fn.eba)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @K.%U.as_type.loc7_38.1 (%U.as_type.5d6)) -> out %return.param: @K.%tuple.type (%tuple.type.7d0) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Pair.ref: %Pair.type = name_ref Pair, imports.%Main.Pair [concrete = constants.%Pair]
// CHECK:STDOUT:     %Id.ref: %Id.type = name_ref Id, imports.%Main.Id [concrete = constants.%Id]
// CHECK:STDOUT:     %u.ref: @K.%U.as_type.loc7_38.1 (%U.as_type.5d6) = name_ref u, %u
// CHECK:STDOUT:     %Copy.facet.loc7_69.1: %Copy.type = facet_value constants.%U.as_type.5d6, (constants.%Copy.lookup_impl_witness.fd8) [symbolic = %Copy.facet.loc7_69.3 (constants.%Copy.facet.40c)]
// CHECK:STDOUT:     %.loc7_69.1: %Copy.type = converted constants.%U.as_type.5d6, %Copy.facet.loc7_69.1 [symbolic = %Copy.facet.loc7_69.3 (constants.%Copy.facet.40c)]
// CHECK:STDOUT:     %Copy.facet.loc7_69.2: %Copy.type = facet_value constants.%U.as_type.5d6, (constants.%Copy.lookup_impl_witness.fd8) [symbolic = %Copy.facet.loc7_69.3 (constants.%Copy.facet.40c)]
// CHECK:STDOUT:     %.loc7_69.2: %Copy.type = converted constants.%U.as_type.5d6, %Copy.facet.loc7_69.2 [symbolic = %Copy.facet.loc7_69.3 (constants.%Copy.facet.40c)]
// CHECK:STDOUT:     %Id.specific_fn.loc7_65.1: <specific function> = specific_function %Id.ref, @Id(constants.%Copy.facet.40c) [symbolic = %Id.specific_fn.loc7_65.2 (constants.%Id.specific_fn.e53)]
// CHECK:STDOUT:     %.loc7_69.3: ref @K.%U.as_type.loc7_38.1 (%U.as_type.5d6) = temporary_storage
// CHECK:STDOUT:     %Id.call: init @K.%U.as_type.loc7_38.1 (%U.as_type.5d6) to %.loc7_69.3 = call %Id.specific_fn.loc7_65.1(%u.ref)
// CHECK:STDOUT:     %Copy.facet.loc7_70.1: %Copy.type = facet_value constants.%U.as_type.5d6, (constants.%Copy.lookup_impl_witness.fd8) [symbolic = %Copy.facet.loc7_69.3 (constants.%Copy.facet.40c)]
// CHECK:STDOUT:     %.loc7_70.1: %Copy.type = converted constants.%U.as_type.5d6, %Copy.facet.loc7_70.1 [symbolic = %Copy.facet.loc7_69.3 (constants.%Copy.facet.40c)]
// CHECK:STDOUT:     %Copy.facet.loc7_70.2: %Copy.type = facet_value constants.%U.as_type.5d6, (constants.%Copy.lookup_impl_witness.fd8) [symbolic = %Copy.facet.loc7_69.3 (constants.%Copy.facet.40c)]
// CHECK:STDOUT:     %.loc7_70.2: %Copy.type = converted constants.%U.as_type.5d6, %Copy.facet.loc7_70.2 [symbolic = %Copy.facet.loc7_69.3 (constants.%Copy.facet.40c)]
// CHECK:STDOUT:     %Pair.specific_fn.loc7_60.1: <specific function> = specific_function %Pair.ref, @Pair(constants.%Copy.facet.40c) [symbolic = %Pair.specific_fn.loc7_60.2 (constants.%Pair.specific_fn.540)]
// CHECK:STDOUT:     %.loc7_49.1: ref @K.%tuple.type (%tuple.type.7d0) = splice_block %return.param {}
// CHECK:STDOUT:     %.loc7_69.4: ref @K.%U.as_type.loc7_38.1 (%U.as_type.5d6) = temporary %.loc7_69.3, %Id.call
// CHECK:STDOUT:     %.loc7_69.5: @K.%U.as_type.loc7_38.1 (%U.as_type.5d6) = acquire_value %.loc7_69.4
// CHECK:STDOUT:     %Pair.call: init @K.%tuple.type (%tuple.type.7d0) to %.loc7_49.1 = call %Pair.specific_fn.loc7_60.1(%.loc7_69.5)
// CHECK:STDOUT:     %impl.elem2.loc7_69.1: @K.%.loc7_69.8 (%.a48) = impl_witness_access constants.%Destroy.lookup_impl_witness.491, element2 [symbolic = %impl.elem2.loc7_69.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc7_69.1: <bound method> = bound_method %.loc7_69.4, %impl.elem2.loc7_69.1
// CHECK:STDOUT:     %Destroy.facet.loc7_69.1: %Destroy.type = facet_value constants.%U.as_type.5d6, (constants.%Destroy.lookup_impl_witness.491) [symbolic = %Destroy.facet.loc7_69.3 (constants.%Destroy.facet.589)]
// CHECK:STDOUT:     %.loc7_69.6: %Destroy.type = converted constants.%U.as_type.5d6, %Destroy.facet.loc7_69.1 [symbolic = %Destroy.facet.loc7_69.3 (constants.%Destroy.facet.589)]
// CHECK:STDOUT:     %Destroy.facet.loc7_69.2: %Destroy.type = facet_value constants.%U.as_type.5d6, (constants.%Destroy.lookup_impl_witness.491) [symbolic = %Destroy.facet.loc7_69.3 (constants.%Destroy.facet.589)]
// CHECK:STDOUT:     %.loc7_69.7: %Destroy.type = converted constants.%U.as_type.5d6, %Destroy.facet.loc7_69.2 [symbolic = %Destroy.facet.loc7_69.3 (constants.%Destroy.facet.589)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_69.1: <specific function> = specific_impl_function %impl.elem2.loc7_69.1, @Destroy.WithSelf.SelfDestruct.1(constants.%Destroy.facet.589) [symbolic = %specific_impl_fn.loc7_69.2 (constants.%specific_impl_fn.eba)]
// CHECK:STDOUT:     %bound_method.loc7_69.2: <bound method> = bound_method %.loc7_69.4, %specific_impl_fn.loc7_69.1
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %bound_method.loc7_69.2(%.loc7_69.4)
// CHECK:STDOUT:     return %Pair.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K(constants.%U.dc4) {
// CHECK:STDOUT:   %U.patt.loc7_7.2 => constants.%U.patt.256
// CHECK:STDOUT:   %U.loc7_7.1 => constants.%U.dc4
// CHECK:STDOUT:   %U.as_type.loc7_38.1 => constants.%U.as_type.5d6
// CHECK:STDOUT:   %pattern_type.loc7_36 => constants.%pattern_type.0e9
// CHECK:STDOUT:   %u.param_patt.loc7_36.2 => constants.%u.param_patt.f9a
// CHECK:STDOUT:   %u.patt.loc7_36.2 => constants.%u.patt.d30
// CHECK:STDOUT:   %tuple => constants.%tuple.f81
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.7d0
// CHECK:STDOUT:   %.loc7_49.2 => constants.%.7db
// CHECK:STDOUT:   %pattern_type.loc7_49 => constants.%pattern_type.c29
// CHECK:STDOUT:   %return.param_patt.loc7_49.2 => constants.%return.param_patt.df9
// CHECK:STDOUT:   %return.patt.loc7_41.2 => constants.%return.patt.b82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K(constants.%facet_value) {
// CHECK:STDOUT:   %U.patt.loc7_7.2 => constants.%U.patt.256
// CHECK:STDOUT:   %U.loc7_7.1 => constants.%facet_value
// CHECK:STDOUT:   %U.as_type.loc7_38.1 => bool
// CHECK:STDOUT:   %pattern_type.loc7_36 => constants.%pattern_type.831
// CHECK:STDOUT:   %u.param_patt.loc7_36.2 => constants.%u.param_patt.56f
// CHECK:STDOUT:   %u.patt.loc7_36.2 => constants.%u.patt.0d7
// CHECK:STDOUT:   %tuple => constants.%tuple.6be
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.784
// CHECK:STDOUT:   %.loc7_49.2 => constants.%.7b8
// CHECK:STDOUT:   %pattern_type.loc7_49 => constants.%pattern_type.860
// CHECK:STDOUT:   %return.param_patt.loc7_49.2 => constants.%return.param_patt.3b7
// CHECK:STDOUT:   %return.patt.loc7_41.2 => constants.%return.patt.fc1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_36 => constants.%complete_type.127
// CHECK:STDOUT:   %require_complete.loc7_41 => constants.%complete_type.e0b
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.5e1
// CHECK:STDOUT:   %Copy.facet.loc7_69.3 => constants.%Copy.facet.7af
// CHECK:STDOUT:   %Id.specific_fn.loc7_65.2 => constants.%Id.specific_fn.06a
// CHECK:STDOUT:   %Pair.specific_fn.loc7_60.2 => constants.%Pair.specific_fn.62b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f
// CHECK:STDOUT:   %Destroy.facet.loc7_69.3 => constants.%Destroy.facet.75f
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.2a2
// CHECK:STDOUT:   %.loc7_69.8 => constants.%.fb9
// CHECK:STDOUT:   %impl.elem2.loc7_69.2 => constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT:   %specific_impl_fn.loc7_69.2 => constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT: }
// CHECK:STDOUT:
