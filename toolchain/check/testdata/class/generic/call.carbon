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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/call.carbon

// --- call.carbon

library "[[@TEST_NAME]]";

class Class(T:! type, N:! i32) {}

var a: Class(i32*, 5);

// Requires an implicit conversion to type `type`.
var b: Class((), 0);

// --- fail_too_few.carbon

library "[[@TEST_NAME]]";

class Class(T:! type, N:! i32) {}

// CHECK:STDERR: fail_too_few.carbon:[[@LINE+7]]:8: error: 1 argument passed to generic class expecting 2 arguments [CallArgCountMismatch]
// CHECK:STDERR: var a: Class(i32*);
// CHECK:STDERR:        ^~~~~~~~~~~
// CHECK:STDERR: fail_too_few.carbon:[[@LINE-5]]:1: note: calling generic class declared here [InCallToEntity]
// CHECK:STDERR: class Class(T:! type, N:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: Class(i32*);

// --- fail_too_many.carbon

library "[[@TEST_NAME]]";

class Class(T:! type, N:! i32) {}

// CHECK:STDERR: fail_too_many.carbon:[[@LINE+7]]:8: error: 3 arguments passed to generic class expecting 2 arguments [CallArgCountMismatch]
// CHECK:STDERR: var a: Class(i32*, 1, 2);
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_too_many.carbon:[[@LINE-5]]:1: note: calling generic class declared here [InCallToEntity]
// CHECK:STDERR: class Class(T:! type, N:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: Class(i32*, 1, 2);

// --- fail_no_conversion.carbon

library "[[@TEST_NAME]]";

class Class(T:! type, N:! i32) {}

// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+10]]:8: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: var a: Class(5, i32*);
// CHECK:STDERR:        ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+7]]:8: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: var a: Class(5, i32*);
// CHECK:STDERR:        ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE-8]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: class Class(T:! type, N:! i32) {}
// CHECK:STDERR:             ^~~~~~~~
// CHECK:STDERR:
var a: Class(5, i32*);

// --- call_in_nested_return_type.carbon

class Outer(T:! type) {
  class Inner(U:! type) {
    fn A() -> Outer(T) {
      return {};
    }
    fn B() -> Outer(U) {
      return {};
    }
    fn C() -> Inner(T) {
      return {};
    }
    fn D() -> Inner(U) {
      return {};
    }
  }
}

// CHECK:STDOUT: --- call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.ce9: %i32 = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.2bd: type = class_type @Class, @Class(%T.67d, %N.ce9) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr.d08: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f6c: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.c9d: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.eef: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Class.4f7: type = class_type @Class, @Class(%ptr.d08, %int_5.eef) [concrete]
// CHECK:STDOUT:   %pattern_type.081: type = pattern_type %Class.4f7 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.150: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.53a: %T.as.DefaultOrUnformed.impl.Op.type.150 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.d52: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.374, @T.as.DefaultOrUnformed.impl(%Class.4f7) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.cd1: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%Class.4f7) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.bc2: %T.as.DefaultOrUnformed.impl.Op.type.cd1 = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.912: %DefaultOrUnformed.type = facet_value %Class.4f7, (%DefaultOrUnformed.impl_witness.d52) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.5c2: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.bc2, @T.as.DefaultOrUnformed.impl.Op(%Class.4f7) [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7b4: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.6e0: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Class.958: type = class_type @Class, @Class(%empty_tuple.type, %int_0.3c0) [concrete]
// CHECK:STDOUT:   %pattern_type.a95: type = pattern_type %Class.958 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.57a: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.374, @T.as.DefaultOrUnformed.impl(%Class.958) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.a07: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%Class.958) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.7eb: %T.as.DefaultOrUnformed.impl.Op.type.a07 = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.1b4: %DefaultOrUnformed.type = facet_value %Class.958, (%DefaultOrUnformed.impl_witness.57a) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.894: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.7eb, @T.as.DefaultOrUnformed.impl.Op(%Class.958) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.cc8: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.150) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.53a)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.374 = impl_witness_table (%Core.import_ref.cc8), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %N.patt: %pattern_type.6b6 = symbolic_binding_pattern N, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc4_14: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc4_27: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.loc4_24: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_24.2: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_24.1 (constants.%N.ce9)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.081 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.081 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Class.4f7 = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc6_21.1: type = splice_block %Class.loc6 [concrete = constants.%Class.4f7] {
// CHECK:STDOUT:     %Class.ref.loc6: %Class.type = name_ref Class, %Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32 [concrete = constants.%ptr.d08]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:     %impl.elem0.loc6: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc6_21.1: <bound method> = bound_method %int_5, %impl.elem0.loc6 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f6c]
// CHECK:STDOUT:     %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_21.2: <bound method> = bound_method %int_5, %specific_fn.loc6 [concrete = constants.%bound_method.c9d]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %i32 = call %bound_method.loc6_21.2(%int_5) [concrete = constants.%int_5.eef]
// CHECK:STDOUT:     %.loc6_21.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%int_5.eef]
// CHECK:STDOUT:     %.loc6_21.3: %i32 = converted %int_5, %.loc6_21.2 [concrete = constants.%int_5.eef]
// CHECK:STDOUT:     %Class.loc6: type = class_type @Class, @Class(constants.%ptr.d08, constants.%int_5.eef) [concrete = constants.%Class.4f7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Class.4f7 = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.a95 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.a95 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Class.958 = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc9_19.1: type = splice_block %Class.loc9 [concrete = constants.%Class.958] {
// CHECK:STDOUT:     %Class.ref.loc9: %Class.type = name_ref Class, %Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %.loc9_15: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %.loc9_19.2: type = converted %.loc9_15, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %impl.elem0.loc9: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc9_19.1: <bound method> = bound_method %int_0, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7b4]
// CHECK:STDOUT:     %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_19.2: <bound method> = bound_method %int_0, %specific_fn.loc9 [concrete = constants.%bound_method.6e0]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_19.2(%int_0) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:     %.loc9_19.3: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:     %.loc9_19.4: %i32 = converted %int_0, %.loc9_19.3 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:     %Class.loc9: type = class_type @Class, @Class(constants.%empty_tuple.type, constants.%int_0.3c0) [concrete = constants.%Class.958]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Class.958 = ref_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_14.2: type, %N.loc4_24.2: %i32) {
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T.67d)]
// CHECK:STDOUT:   %N.loc4_24.1: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_24.1 (constants.%N.ce9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2bd
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc6: %DefaultOrUnformed.type = facet_value constants.%Class.4f7, (constants.%DefaultOrUnformed.impl_witness.d52) [concrete = constants.%DefaultOrUnformed.facet.912]
// CHECK:STDOUT:   %.loc6_22.1: %DefaultOrUnformed.type = converted constants.%Class.4f7, %DefaultOrUnformed.facet.loc6 [concrete = constants.%DefaultOrUnformed.facet.912]
// CHECK:STDOUT:   %as_type.loc6: type = facet_access_type %.loc6_22.1 [concrete = constants.%Class.4f7]
// CHECK:STDOUT:   %.loc6_22.2: type = converted %.loc6_22.1, %as_type.loc6 [concrete = constants.%Class.4f7]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.1: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.bc2, @T.as.DefaultOrUnformed.impl.Op(constants.%Class.4f7) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn.5c2]
// CHECK:STDOUT:   %.loc6_1: ref %Class.4f7 = splice_block file.%a.var [concrete = file.%a.var] {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc6: init %Class.4f7 to %.loc6_1 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.1()
// CHECK:STDOUT:   assign file.%a.var, %T.as.DefaultOrUnformed.impl.Op.call.loc6
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc9: %DefaultOrUnformed.type = facet_value constants.%Class.958, (constants.%DefaultOrUnformed.impl_witness.57a) [concrete = constants.%DefaultOrUnformed.facet.1b4]
// CHECK:STDOUT:   %.loc9_20.1: %DefaultOrUnformed.type = converted constants.%Class.958, %DefaultOrUnformed.facet.loc9 [concrete = constants.%DefaultOrUnformed.facet.1b4]
// CHECK:STDOUT:   %as_type.loc9: type = facet_access_type %.loc9_20.1 [concrete = constants.%Class.958]
// CHECK:STDOUT:   %.loc9_20.2: type = converted %.loc9_20.1, %as_type.loc9 [concrete = constants.%Class.958]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.2: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.7eb, @T.as.DefaultOrUnformed.impl.Op(constants.%Class.958) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn.894]
// CHECK:STDOUT:   %.loc9_1: ref %Class.958 = splice_block file.%b.var [concrete = file.%b.var] {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc9: init %Class.958 to %.loc9_1 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.2()
// CHECK:STDOUT:   assign file.%b.var, %T.as.DefaultOrUnformed.impl.Op.call.loc9
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.67d, constants.%N.ce9) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T.67d
// CHECK:STDOUT:   %N.loc4_24.1 => constants.%N.ce9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%ptr.d08, constants.%int_5.eef) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%ptr.d08
// CHECK:STDOUT:   %N.loc4_24.1 => constants.%int_5.eef
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%empty_tuple.type, constants.%int_0.3c0) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%empty_tuple.type
// CHECK:STDOUT:   %N.loc4_24.1 => constants.%int_0.3c0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_few.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.ce9: %i32 = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T, %N.ce9) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %N.patt: %pattern_type.6b6 = symbolic_binding_pattern N, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc4_14: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4_27: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.loc4_24: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_24.2: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_24.1 (constants.%N.ce9)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: <error> = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: <error> = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref <error> = var %a.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, %Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32 [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref <error> = ref_binding a, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_14.2: type, %N.loc4_24.2: %i32) {
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:   %N.loc4_24.1: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_24.1 (constants.%N.ce9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   assign file.%a.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N.ce9) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT:   %N.loc4_24.1 => constants.%N.ce9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_many.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.ce9: %i32 = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T, %N.ce9) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %N.patt: %pattern_type.6b6 = symbolic_binding_pattern N, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc4_14: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4_27: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.loc4_24: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_24.2: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_24.1 (constants.%N.ce9)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: <error> = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: <error> = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref <error> = var %a.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, %Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32 [concrete = constants.%ptr]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref <error> = ref_binding a, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_14.2: type, %N.loc4_24.2: %i32) {
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:   %N.loc4_24.1: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_24.1 (constants.%N.ce9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   assign file.%a.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N.ce9) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT:   %N.loc4_24.1 => constants.%N.ce9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.ce9: %i32 = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T, %N.ce9) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %N.patt: %pattern_type.6b6 = symbolic_binding_pattern N, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc4_14: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4_27: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.loc4_24: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_24.2: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_24.1 (constants.%N.ce9)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: <error> = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: <error> = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref <error> = var %a.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, %Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32 [concrete = constants.%ptr]
// CHECK:STDOUT:     %.loc16: type = converted %int_5, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref <error> = ref_binding a, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_14.2: type, %N.loc4_24.2: %i32) {
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:   %N.loc4_24.1: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_24.1 (constants.%N.ce9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   assign file.%a.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N.ce9) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT:   %N.loc4_24.1 => constants.%N.ce9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_in_nested_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.dee: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Inner.type.8f7: type = generic_class_type @Inner, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.generic.b3a: %Inner.type.8f7 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.34b: type = class_type @Inner, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %.156: Core.Form = init_form %Outer.dee [symbolic]
// CHECK:STDOUT:   %pattern_type.dec: type = pattern_type %Outer.dee [symbolic]
// CHECK:STDOUT:   %Inner.A.type.b38: type = fn_type @Inner.A, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %Inner.A.456: %Inner.A.type.b38 = struct_value () [symbolic]
// CHECK:STDOUT:   %Outer.20b: type = class_type @Outer, @Outer(%U) [symbolic]
// CHECK:STDOUT:   %.aaf: Core.Form = init_form %Outer.20b [symbolic]
// CHECK:STDOUT:   %pattern_type.a71: type = pattern_type %Outer.20b [symbolic]
// CHECK:STDOUT:   %Inner.B.type.374: type = fn_type @Inner.B, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %Inner.B.cc2: %Inner.B.type.374 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.61d: type = class_type @Inner, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %.2f2: Core.Form = init_form %Inner.61d [symbolic]
// CHECK:STDOUT:   %pattern_type.5b1: type = pattern_type %Inner.61d [symbolic]
// CHECK:STDOUT:   %Inner.C.type.5dd: type = fn_type @Inner.C, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %Inner.C.5e1: %Inner.C.type.5dd = struct_value () [symbolic]
// CHECK:STDOUT:   %.556: Core.Form = init_form %Inner.34b [symbolic]
// CHECK:STDOUT:   %pattern_type.f00: type = pattern_type %Inner.34b [symbolic]
// CHECK:STDOUT:   %Inner.D.type.faa: type = fn_type @Inner.D, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %Inner.D.039: %Inner.D.type.faa = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.dae: <witness> = require_complete_type %Outer.dee [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.val.1e6: %Outer.dee = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.type.1c5: type = generic_class_type @Inner, @Outer(%U) [symbolic]
// CHECK:STDOUT:   %Inner.generic.8bb: %Inner.type.1c5 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.7da: <witness> = require_complete_type %Outer.20b [symbolic]
// CHECK:STDOUT:   %Outer.val.8e5: %Outer.20b = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.A.type.6cc: type = fn_type @Inner.A, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %Inner.A.0e7: %Inner.A.type.6cc = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.B.type.e01: type = fn_type @Inner.B, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %Inner.B.020: %Inner.B.type.e01 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.C.type.382: type = fn_type @Inner.C, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %Inner.C.fa4: %Inner.C.type.382 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.D.type.f65: type = fn_type @Inner.D, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %Inner.D.9e2: %Inner.D.type.f65 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.25e: <witness> = require_complete_type %Inner.61d [symbolic]
// CHECK:STDOUT:   %Inner.val.284: %Inner.61d = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.8ae: <witness> = require_complete_type %Inner.34b [symbolic]
// CHECK:STDOUT:   %Inner.val.9fd: %Inner.34b = struct_value () [symbolic]
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
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Outer.decl: %Outer.type = class_decl @Outer [concrete = constants.%Outer.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc2_17.1: type = splice_block %.loc2_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc2_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc2_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc2_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc2_14.2: type) {
// CHECK:STDOUT:   %T.loc2_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc2_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T.loc2_14.1) [symbolic = %Inner.type (constants.%Inner.type.8f7)]
// CHECK:STDOUT:   %Inner.generic: @Outer.%Inner.type (%Inner.type.8f7) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.b3a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: @Outer.%Inner.type (%Inner.type.8f7) = class_decl @Inner [symbolic = @Outer.%Inner.generic (constants.%Inner.generic.b3a)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc3_19.1: type = splice_block %.loc3_19.2 [concrete = type] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %.loc3_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc3_16.2: type = symbolic_binding U, 1 [symbolic = %U.loc3_16.1 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.dee
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:     .Outer = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc2_14.2: type, %U.loc3_16.2: type) {
// CHECK:STDOUT:   %U.loc3_16.1: type = symbolic_binding U, 1 [symbolic = %U.loc3_16.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.A.type: type = fn_type @Inner.A, @Inner(%T, %U.loc3_16.1) [symbolic = %Inner.A.type (constants.%Inner.A.type.b38)]
// CHECK:STDOUT:   %Inner.A: @Inner.%Inner.A.type (%Inner.A.type.b38) = struct_value () [symbolic = %Inner.A (constants.%Inner.A.456)]
// CHECK:STDOUT:   %Inner.B.type: type = fn_type @Inner.B, @Inner(%T, %U.loc3_16.1) [symbolic = %Inner.B.type (constants.%Inner.B.type.374)]
// CHECK:STDOUT:   %Inner.B: @Inner.%Inner.B.type (%Inner.B.type.374) = struct_value () [symbolic = %Inner.B (constants.%Inner.B.cc2)]
// CHECK:STDOUT:   %Inner.C.type: type = fn_type @Inner.C, @Inner(%T, %U.loc3_16.1) [symbolic = %Inner.C.type (constants.%Inner.C.type.5dd)]
// CHECK:STDOUT:   %Inner.C: @Inner.%Inner.C.type (%Inner.C.type.5dd) = struct_value () [symbolic = %Inner.C (constants.%Inner.C.5e1)]
// CHECK:STDOUT:   %Inner.D.type: type = fn_type @Inner.D, @Inner(%T, %U.loc3_16.1) [symbolic = %Inner.D.type (constants.%Inner.D.type.faa)]
// CHECK:STDOUT:   %Inner.D: @Inner.%Inner.D.type (%Inner.D.type.faa) = struct_value () [symbolic = %Inner.D (constants.%Inner.D.039)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.A.decl: @Inner.%Inner.A.type (%Inner.A.type.b38) = fn_decl @Inner.A [symbolic = @Inner.%Inner.A (constants.%Inner.A.456)] {
// CHECK:STDOUT:       %return.param_patt: @Inner.A.%pattern_type (%pattern_type.dec) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Inner.A.%pattern_type (%pattern_type.dec) = return_slot_pattern %return.param_patt, %Outer.loc4_22.2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc2_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Outer.loc4_22.2: type = class_type @Outer, @Outer(constants.%T) [symbolic = %Outer.loc4_22.1 (constants.%Outer.dee)]
// CHECK:STDOUT:       %.loc4_22.2: Core.Form = init_form %Outer.loc4_22.2 [symbolic = %.loc4_22.1 (constants.%.156)]
// CHECK:STDOUT:       %return.param: ref @Inner.A.%Outer.loc4_22.1 (%Outer.dee) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Inner.A.%Outer.loc4_22.1 (%Outer.dee) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Inner.B.decl: @Inner.%Inner.B.type (%Inner.B.type.374) = fn_decl @Inner.B [symbolic = @Inner.%Inner.B (constants.%Inner.B.cc2)] {
// CHECK:STDOUT:       %return.param_patt: @Inner.B.%pattern_type (%pattern_type.a71) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Inner.B.%pattern_type (%pattern_type.a71) = return_slot_pattern %return.param_patt, %Outer.loc7_22.2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @Inner.%U.loc3_16.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %Outer.loc7_22.2: type = class_type @Outer, @Outer(constants.%U) [symbolic = %Outer.loc7_22.1 (constants.%Outer.20b)]
// CHECK:STDOUT:       %.loc7_22.2: Core.Form = init_form %Outer.loc7_22.2 [symbolic = %.loc7_22.1 (constants.%.aaf)]
// CHECK:STDOUT:       %return.param: ref @Inner.B.%Outer.loc7_22.1 (%Outer.20b) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Inner.B.%Outer.loc7_22.1 (%Outer.20b) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Inner.C.decl: @Inner.%Inner.C.type (%Inner.C.type.5dd) = fn_decl @Inner.C [symbolic = @Inner.%Inner.C (constants.%Inner.C.5e1)] {
// CHECK:STDOUT:       %return.param_patt: @Inner.C.%pattern_type (%pattern_type.5b1) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Inner.C.%pattern_type (%pattern_type.5b1) = return_slot_pattern %return.param_patt, %Inner.loc10_22.2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc10_15: @Inner.C.%Inner.type (%Inner.type.8f7) = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.generic (constants.%Inner.generic.b3a)]
// CHECK:STDOUT:       %Inner.ref: @Inner.C.%Inner.type (%Inner.type.8f7) = name_ref Inner, %.loc10_15 [symbolic = %Inner.generic (constants.%Inner.generic.b3a)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc2_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Inner.loc10_22.2: type = class_type @Inner, @Inner(constants.%T, constants.%T) [symbolic = %Inner.loc10_22.1 (constants.%Inner.61d)]
// CHECK:STDOUT:       %.loc10_22.2: Core.Form = init_form %Inner.loc10_22.2 [symbolic = %.loc10_22.1 (constants.%.2f2)]
// CHECK:STDOUT:       %return.param: ref @Inner.C.%Inner.loc10_22.1 (%Inner.61d) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Inner.C.%Inner.loc10_22.1 (%Inner.61d) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Inner.D.decl: @Inner.%Inner.D.type (%Inner.D.type.faa) = fn_decl @Inner.D [symbolic = @Inner.%Inner.D (constants.%Inner.D.039)] {
// CHECK:STDOUT:       %return.param_patt: @Inner.D.%pattern_type (%pattern_type.f00) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Inner.D.%pattern_type (%pattern_type.f00) = return_slot_pattern %return.param_patt, %Inner.loc13_22.2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc13_15: @Inner.D.%Inner.type (%Inner.type.8f7) = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.generic (constants.%Inner.generic.b3a)]
// CHECK:STDOUT:       %Inner.ref: @Inner.D.%Inner.type (%Inner.type.8f7) = name_ref Inner, %.loc13_15 [symbolic = %Inner.generic (constants.%Inner.generic.b3a)]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @Inner.%U.loc3_16.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %Inner.loc13_22.2: type = class_type @Inner, @Inner(constants.%T, constants.%U) [symbolic = %Inner.loc13_22.1 (constants.%Inner.34b)]
// CHECK:STDOUT:       %.loc13_22.2: Core.Form = init_form %Inner.loc13_22.2 [symbolic = %.loc13_22.1 (constants.%.556)]
// CHECK:STDOUT:       %return.param: ref @Inner.D.%Inner.loc13_22.1 (%Inner.34b) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Inner.D.%Inner.loc13_22.1 (%Inner.34b) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.34b
// CHECK:STDOUT:     .Outer = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .A = %Inner.A.decl
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     .B = %Inner.B.decl
// CHECK:STDOUT:     .Inner = <poisoned>
// CHECK:STDOUT:     .C = %Inner.C.decl
// CHECK:STDOUT:     .D = %Inner.D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.A(@Outer.%T.loc2_14.2: type, @Inner.%U.loc3_16.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Outer.loc4_22.1: type = class_type @Outer, @Outer(%T) [symbolic = %Outer.loc4_22.1 (constants.%Outer.dee)]
// CHECK:STDOUT:   %.loc4_22.1: Core.Form = init_form %Outer.loc4_22.1 [symbolic = %.loc4_22.1 (constants.%.156)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Outer.loc4_22.1 [symbolic = %pattern_type (constants.%pattern_type.dec)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Outer.loc4_22.1 [symbolic = %require_complete (constants.%require_complete.dae)]
// CHECK:STDOUT:   %Outer.val: @Inner.A.%Outer.loc4_22.1 (%Outer.dee) = struct_value () [symbolic = %Outer.val (constants.%Outer.val.1e6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Inner.A.%Outer.loc4_22.1 (%Outer.dee) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc5_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc5_15.2: init @Inner.A.%Outer.loc4_22.1 (%Outer.dee) to %return.param = class_init () [symbolic = %Outer.val (constants.%Outer.val.1e6)]
// CHECK:STDOUT:     %.loc5_16: init @Inner.A.%Outer.loc4_22.1 (%Outer.dee) = converted %.loc5_15.1, %.loc5_15.2 [symbolic = %Outer.val (constants.%Outer.val.1e6)]
// CHECK:STDOUT:     return %.loc5_16 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.B(@Outer.%T.loc2_14.2: type, @Inner.%U.loc3_16.2: type) {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %Outer.loc7_22.1: type = class_type @Outer, @Outer(%U) [symbolic = %Outer.loc7_22.1 (constants.%Outer.20b)]
// CHECK:STDOUT:   %.loc7_22.1: Core.Form = init_form %Outer.loc7_22.1 [symbolic = %.loc7_22.1 (constants.%.aaf)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Outer.loc7_22.1 [symbolic = %pattern_type (constants.%pattern_type.a71)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Outer.loc7_22.1 [symbolic = %require_complete (constants.%require_complete.7da)]
// CHECK:STDOUT:   %Outer.val: @Inner.B.%Outer.loc7_22.1 (%Outer.20b) = struct_value () [symbolic = %Outer.val (constants.%Outer.val.8e5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Inner.B.%Outer.loc7_22.1 (%Outer.20b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc8_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_15.2: init @Inner.B.%Outer.loc7_22.1 (%Outer.20b) to %return.param = class_init () [symbolic = %Outer.val (constants.%Outer.val.8e5)]
// CHECK:STDOUT:     %.loc8_16: init @Inner.B.%Outer.loc7_22.1 (%Outer.20b) = converted %.loc8_15.1, %.loc8_15.2 [symbolic = %Outer.val (constants.%Outer.val.8e5)]
// CHECK:STDOUT:     return %.loc8_16 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.C(@Outer.%T.loc2_14.2: type, @Inner.%U.loc3_16.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T) [symbolic = %Inner.type (constants.%Inner.type.8f7)]
// CHECK:STDOUT:   %Inner.generic: @Inner.C.%Inner.type (%Inner.type.8f7) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.b3a)]
// CHECK:STDOUT:   %Inner.loc10_22.1: type = class_type @Inner, @Inner(%T, %T) [symbolic = %Inner.loc10_22.1 (constants.%Inner.61d)]
// CHECK:STDOUT:   %.loc10_22.1: Core.Form = init_form %Inner.loc10_22.1 [symbolic = %.loc10_22.1 (constants.%.2f2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Inner.loc10_22.1 [symbolic = %pattern_type (constants.%pattern_type.5b1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Inner.loc10_22.1 [symbolic = %require_complete (constants.%require_complete.25e)]
// CHECK:STDOUT:   %Inner.val: @Inner.C.%Inner.loc10_22.1 (%Inner.61d) = struct_value () [symbolic = %Inner.val (constants.%Inner.val.284)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Inner.C.%Inner.loc10_22.1 (%Inner.61d) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc11_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc11_15.2: init @Inner.C.%Inner.loc10_22.1 (%Inner.61d) to %return.param = class_init () [symbolic = %Inner.val (constants.%Inner.val.284)]
// CHECK:STDOUT:     %.loc11_16: init @Inner.C.%Inner.loc10_22.1 (%Inner.61d) = converted %.loc11_15.1, %.loc11_15.2 [symbolic = %Inner.val (constants.%Inner.val.284)]
// CHECK:STDOUT:     return %.loc11_16 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.D(@Outer.%T.loc2_14.2: type, @Inner.%U.loc3_16.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T) [symbolic = %Inner.type (constants.%Inner.type.8f7)]
// CHECK:STDOUT:   %Inner.generic: @Inner.D.%Inner.type (%Inner.type.8f7) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.b3a)]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %Inner.loc13_22.1: type = class_type @Inner, @Inner(%T, %U) [symbolic = %Inner.loc13_22.1 (constants.%Inner.34b)]
// CHECK:STDOUT:   %.loc13_22.1: Core.Form = init_form %Inner.loc13_22.1 [symbolic = %.loc13_22.1 (constants.%.556)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Inner.loc13_22.1 [symbolic = %pattern_type (constants.%pattern_type.f00)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Inner.loc13_22.1 [symbolic = %require_complete (constants.%require_complete.8ae)]
// CHECK:STDOUT:   %Inner.val: @Inner.D.%Inner.loc13_22.1 (%Inner.34b) = struct_value () [symbolic = %Inner.val (constants.%Inner.val.9fd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Inner.D.%Inner.loc13_22.1 (%Inner.34b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc14_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc14_15.2: init @Inner.D.%Inner.loc13_22.1 (%Inner.34b) to %return.param = class_init () [symbolic = %Inner.val (constants.%Inner.val.9fd)]
// CHECK:STDOUT:     %.loc14_16: init @Inner.D.%Inner.loc13_22.1 (%Inner.34b) = converted %.loc14_15.1, %.loc14_15.2 [symbolic = %Inner.val (constants.%Inner.val.9fd)]
// CHECK:STDOUT:     return %.loc14_16 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.loc2_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.8f7
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.b3a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc3_16.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.A.type => constants.%Inner.A.type.b38
// CHECK:STDOUT:   %Inner.A => constants.%Inner.A.456
// CHECK:STDOUT:   %Inner.B.type => constants.%Inner.B.type.374
// CHECK:STDOUT:   %Inner.B => constants.%Inner.B.cc2
// CHECK:STDOUT:   %Inner.C.type => constants.%Inner.C.type.5dd
// CHECK:STDOUT:   %Inner.C => constants.%Inner.C.5e1
// CHECK:STDOUT:   %Inner.D.type => constants.%Inner.D.type.faa
// CHECK:STDOUT:   %Inner.D => constants.%Inner.D.039
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.A(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Outer.loc4_22.1 => constants.%Outer.dee
// CHECK:STDOUT:   %.loc4_22.1 => constants.%.156
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.dec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%U) {
// CHECK:STDOUT:   %T.loc2_14.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.1c5
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.8bb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.B(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %Outer.loc7_22.1 => constants.%Outer.20b
// CHECK:STDOUT:   %.loc7_22.1 => constants.%.aaf
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T, constants.%T) {
// CHECK:STDOUT:   %U.loc3_16.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.A.type => constants.%Inner.A.type.6cc
// CHECK:STDOUT:   %Inner.A => constants.%Inner.A.0e7
// CHECK:STDOUT:   %Inner.B.type => constants.%Inner.B.type.e01
// CHECK:STDOUT:   %Inner.B => constants.%Inner.B.020
// CHECK:STDOUT:   %Inner.C.type => constants.%Inner.C.type.382
// CHECK:STDOUT:   %Inner.C => constants.%Inner.C.fa4
// CHECK:STDOUT:   %Inner.D.type => constants.%Inner.D.type.f65
// CHECK:STDOUT:   %Inner.D => constants.%Inner.D.9e2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.C(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.8f7
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.b3a
// CHECK:STDOUT:   %Inner.loc10_22.1 => constants.%Inner.61d
// CHECK:STDOUT:   %.loc10_22.1 => constants.%.2f2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.5b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.D(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.8f7
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.b3a
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %Inner.loc13_22.1 => constants.%Inner.34b
// CHECK:STDOUT:   %.loc13_22.1 => constants.%.556
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f00
// CHECK:STDOUT: }
// CHECK:STDOUT:
