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

class Class(T: type, N: i32) {}

var a: Class(i32*, 5);

// Requires an implicit conversion to type `type`.
var b: Class((), 0);

// --- fail_too_few.carbon

library "[[@TEST_NAME]]";

class Class(T: type, N: i32) {}

// CHECK:STDERR: fail_too_few.carbon:[[@LINE+3]]:8-18: error: 1 argument passed to generic class expecting 2 arguments [CallArgCountMismatch]
// CHECK:STDERR: fail_too_few.carbon:[[@LINE-3]]:1-30: note: calling generic class declared here [InCallToEntity]
// CHECK:STDERR:
var a: Class(i32*);

// --- fail_too_many.carbon

library "[[@TEST_NAME]]";

class Class(T: type, N: i32) {}

// CHECK:STDERR: fail_too_many.carbon:[[@LINE+3]]:8-24: error: 3 arguments passed to generic class expecting 2 arguments [CallArgCountMismatch]
// CHECK:STDERR: fail_too_many.carbon:[[@LINE-3]]:1-30: note: calling generic class declared here [InCallToEntity]
// CHECK:STDERR:
var a: Class(i32*, 1, 2);

// --- fail_no_conversion.carbon

library "[[@TEST_NAME]]";

class Class(T: type, N: i32) {}

// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+4]]:14: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+3]]:8-21: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplForType]
// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE-4]]:13-19: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR:
var a: Class(5, i32*);

// --- call_in_nested_return_type.carbon

class Outer(T: type) {
  class Inner(U: type) {
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.b85: %pattern_type.d94 = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %N.55f: %i32 = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = parameterized_class_name_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.c39: type = class_type @Class, @Class(%T.67d, %N.55f) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f65: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.ab2: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.708: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Class.29e: type = class_type @Class, @Class(%ptr.a17, %int_5.708) [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %Class.29e [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.049 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.049 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.5f6: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.5c0: %T.as.DefaultOrUnformed.impl.Op.type.5f6 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.90f: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.856, @T.as.DefaultOrUnformed.impl(%Class.29e) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.ec3: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%Class.29e) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.7ee: %T.as.DefaultOrUnformed.impl.Op.type.ec3 = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.3e6: %DefaultOrUnformed.type = facet_value %Class.29e, (%DefaultOrUnformed.impl_witness.90f) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.7ad: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.7ee, @T.as.DefaultOrUnformed.impl.Op(%Class.29e) [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.762: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.036: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.753: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Class.766: type = class_type @Class, @Class(%empty_tuple.type, %int_0.753) [concrete]
// CHECK:STDOUT:   %pattern_type.6a3: type = pattern_type %Class.766 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.6a3 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.6a3 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.7d5: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.856, @T.as.DefaultOrUnformed.impl(%Class.766) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.991: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%Class.766) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.715: %T.as.DefaultOrUnformed.impl.Op.type.991 = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.501: %DefaultOrUnformed.type = facet_value %Class.766, (%DefaultOrUnformed.impl_witness.7d5) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.5ed: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.715, @T.as.DefaultOrUnformed.impl.Op(%Class.766) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.002: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.5f6) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.5c0)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.856 = impl_witness_table (%Core.import_ref.002), @T.as.DefaultOrUnformed.impl [concrete]
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
// CHECK:STDOUT:     %T.patt.loc4_14.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:     %N.patt.loc4_23.1: %pattern_type.d94 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.b85)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_16.1: type = splice_block %.loc4_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc4_14: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc4_25: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.frozen.loc4_23: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_23.2: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_23.1 (constants.%N.55f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Class.29e = var_storage %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc6_21.1: type = splice_block %Class.loc6 [concrete = constants.%Class.29e] {
// CHECK:STDOUT:     %Class.ref.loc6: %Class.type = name_ref Class, %Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32 [concrete = constants.%ptr.a17]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:     %impl.elem0.loc6: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:     %bound_method.loc6_21.1: <bound method> = bound_method %int_5, %impl.elem0.loc6 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f65]
// CHECK:STDOUT:     %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_21.2: <bound method> = bound_method %int_5, %specific_fn.loc6 [concrete = constants.%bound_method.ab2]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %i32 = call %bound_method.loc6_21.2(%int_5) [concrete = constants.%int_5.708]
// CHECK:STDOUT:     %.loc6_21.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%int_5.708]
// CHECK:STDOUT:     %.loc6_21.3: %i32 = converted %int_5, %.loc6_21.2 [concrete = constants.%int_5.708]
// CHECK:STDOUT:     %Class.loc6: type = class_type @Class, @Class(constants.%ptr.a17, constants.%int_5.708) [concrete = constants.%Class.29e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Class.29e = wrapper_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.049 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.049 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Class.766 = var_storage %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc9_19.1: type = splice_block %Class.loc9 [concrete = constants.%Class.766] {
// CHECK:STDOUT:     %Class.ref.loc9: %Class.type = name_ref Class, %Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %.loc9_15: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %.loc9_19.2: type = converted %.loc9_15, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %impl.elem0.loc9: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:     %bound_method.loc9_19.1: <bound method> = bound_method %int_0, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.762]
// CHECK:STDOUT:     %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_19.2: <bound method> = bound_method %int_0, %specific_fn.loc9 [concrete = constants.%bound_method.036]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_19.2(%int_0) [concrete = constants.%int_0.753]
// CHECK:STDOUT:     %.loc9_19.3: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_0.753]
// CHECK:STDOUT:     %.loc9_19.4: %i32 = converted %int_0, %.loc9_19.3 [concrete = constants.%int_0.753]
// CHECK:STDOUT:     %Class.loc9: type = class_type @Class, @Class(constants.%empty_tuple.type, constants.%int_0.753) [concrete = constants.%Class.766]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Class.766 = wrapper_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.6a3 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.6a3 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_14.2: type, %N.loc4_23.2: %i32) {
// CHECK:STDOUT:   %T.patt.loc4_14.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T.67d)]
// CHECK:STDOUT:   %N.patt.loc4_23.2: %pattern_type.d94 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.b85)]
// CHECK:STDOUT:   %N.loc4_23.1: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_23.1 (constants.%N.55f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.c39
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc6: %DefaultOrUnformed.type = facet_value constants.%Class.29e, (constants.%DefaultOrUnformed.impl_witness.90f) [concrete = constants.%DefaultOrUnformed.facet.3e6]
// CHECK:STDOUT:   %.loc6_22.1: %DefaultOrUnformed.type = converted constants.%Class.29e, %DefaultOrUnformed.facet.loc6 [concrete = constants.%DefaultOrUnformed.facet.3e6]
// CHECK:STDOUT:   %as_type.loc6: type = facet_access_type %.loc6_22.1 [concrete = constants.%Class.29e]
// CHECK:STDOUT:   %.loc6_22.2: type = converted %.loc6_22.1, %as_type.loc6 [concrete = constants.%Class.29e]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.1: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.7ee, @T.as.DefaultOrUnformed.impl.Op(constants.%Class.29e) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn.7ad]
// CHECK:STDOUT:   %.loc6_1: ref %Class.29e = splice_block file.%a.var [concrete = file.%a.var] {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc6: init %Class.29e to %.loc6_1 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.1()
// CHECK:STDOUT:   assign file.%a.var, %T.as.DefaultOrUnformed.impl.Op.call.loc6
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc9: %DefaultOrUnformed.type = facet_value constants.%Class.766, (constants.%DefaultOrUnformed.impl_witness.7d5) [concrete = constants.%DefaultOrUnformed.facet.501]
// CHECK:STDOUT:   %.loc9_20.1: %DefaultOrUnformed.type = converted constants.%Class.766, %DefaultOrUnformed.facet.loc9 [concrete = constants.%DefaultOrUnformed.facet.501]
// CHECK:STDOUT:   %as_type.loc9: type = facet_access_type %.loc9_20.1 [concrete = constants.%Class.766]
// CHECK:STDOUT:   %.loc9_20.2: type = converted %.loc9_20.1, %as_type.loc9 [concrete = constants.%Class.766]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.2: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.715, @T.as.DefaultOrUnformed.impl.Op(constants.%Class.766) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn.5ed]
// CHECK:STDOUT:   %.loc9_1: ref %Class.766 = splice_block file.%b.var [concrete = file.%b.var] {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc9: init %Class.766 to %.loc9_1 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.2()
// CHECK:STDOUT:   assign file.%b.var, %T.as.DefaultOrUnformed.impl.Op.call.loc9
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.67d, constants.%N.55f) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T.67d
// CHECK:STDOUT:   %N.patt.loc4_23.2 => constants.%N.patt.b85
// CHECK:STDOUT:   %N.loc4_23.1 => constants.%N.55f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%ptr.a17, constants.%int_5.708) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%ptr.a17
// CHECK:STDOUT:   %N.patt.loc4_23.2 => constants.%N.patt.b85
// CHECK:STDOUT:   %N.loc4_23.1 => constants.%int_5.708
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%empty_tuple.type, constants.%int_0.753) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%empty_tuple.type
// CHECK:STDOUT:   %N.patt.loc4_23.2 => constants.%N.patt.b85
// CHECK:STDOUT:   %N.loc4_23.1 => constants.%int_0.753
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_few.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.b85: %pattern_type.d94 = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %N.55f: %i32 = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = parameterized_class_name_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T, %N.55f) [symbolic]
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
// CHECK:STDOUT:     %T.patt.loc4_14.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc4_23.1: %pattern_type.d94 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.b85)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_16.1: type = splice_block %.loc4_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc4_14: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4_25: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.frozen.loc4_23: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_23.2: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_23.1 (constants.%N.55f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref <error> = var_storage %a.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, %Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32 [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: <error> = wrapper_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: <error> = ref_binding_pattern a [concrete = <error>]
// CHECK:STDOUT:     %a.var_patt: <error> = var_pattern %a.patt [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_14.2: type, %N.loc4_23.2: %i32) {
// CHECK:STDOUT:   %T.patt.loc4_14.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:   %N.patt.loc4_23.2: %pattern_type.d94 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.b85)]
// CHECK:STDOUT:   %N.loc4_23.1: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_23.1 (constants.%N.55f)]
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
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N.55f) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT:   %N.patt.loc4_23.2 => constants.%N.patt.b85
// CHECK:STDOUT:   %N.loc4_23.1 => constants.%N.55f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_many.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.b85: %pattern_type.d94 = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %N.55f: %i32 = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = parameterized_class_name_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T, %N.55f) [symbolic]
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
// CHECK:STDOUT:     %T.patt.loc4_14.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc4_23.1: %pattern_type.d94 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.b85)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_16.1: type = splice_block %.loc4_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc4_14: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4_25: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.frozen.loc4_23: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_23.2: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_23.1 (constants.%N.55f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref <error> = var_storage %a.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, %Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32 [concrete = constants.%ptr]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: <error> = wrapper_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: <error> = ref_binding_pattern a [concrete = <error>]
// CHECK:STDOUT:     %a.var_patt: <error> = var_pattern %a.patt [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_14.2: type, %N.loc4_23.2: %i32) {
// CHECK:STDOUT:   %T.patt.loc4_14.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:   %N.patt.loc4_23.2: %pattern_type.d94 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.b85)]
// CHECK:STDOUT:   %N.loc4_23.1: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_23.1 (constants.%N.55f)]
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
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N.55f) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT:   %N.patt.loc4_23.2 => constants.%N.patt.b85
// CHECK:STDOUT:   %N.loc4_23.1 => constants.%N.55f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.b85: %pattern_type.d94 = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %N.55f: %i32 = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = parameterized_class_name_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T, %N.55f) [symbolic]
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
// CHECK:STDOUT:     %T.patt.loc4_14.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc4_23.1: %pattern_type.d94 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.b85)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_16.1: type = splice_block %.loc4_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc4_14: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4_25: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.frozen.loc4_23: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_23.2: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_23.1 (constants.%N.55f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref <error> = var_storage %a.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, %Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32 [concrete = constants.%ptr]
// CHECK:STDOUT:     %.loc10: type = converted %int_5, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: <error> = wrapper_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: <error> = ref_binding_pattern a [concrete = <error>]
// CHECK:STDOUT:     %a.var_patt: <error> = var_pattern %a.patt [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_14.2: type, %N.loc4_23.2: %i32) {
// CHECK:STDOUT:   %T.patt.loc4_14.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:   %N.patt.loc4_23.2: %pattern_type.d94 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.b85)]
// CHECK:STDOUT:   %N.loc4_23.1: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_23.1 (constants.%N.55f)]
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
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N.55f) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT:   %N.patt.loc4_23.2 => constants.%N.patt.b85
// CHECK:STDOUT:   %N.loc4_23.1 => constants.%N.55f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_in_nested_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = parameterized_class_name_type @Outer [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.651: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %U.patt: %pattern_type.9a5 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Inner.type.19b: type = parameterized_class_name_type @Inner, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.generic.282: %Inner.type.19b = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.b8a: type = class_type @Inner, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %.a0c: Core.Form = init_form %Outer.651 [symbolic]
// CHECK:STDOUT:   %pattern_type.fc0: type = pattern_type %Outer.651 [symbolic]
// CHECK:STDOUT:   %return.param_patt.e08: %pattern_type.fc0 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.3d6: %pattern_type.fc0 = return_slot_pattern %return.param_patt.e08, %Outer.651 [symbolic]
// CHECK:STDOUT:   %Inner.A.type.af3: type = fn_type @Inner.A, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %Inner.A.ff2: %Inner.A.type.af3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Outer.004: type = class_type @Outer, @Outer(%U) [symbolic]
// CHECK:STDOUT:   %.38b: Core.Form = init_form %Outer.004 [symbolic]
// CHECK:STDOUT:   %pattern_type.673: type = pattern_type %Outer.004 [symbolic]
// CHECK:STDOUT:   %return.param_patt.4af: %pattern_type.673 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.2eb: %pattern_type.673 = return_slot_pattern %return.param_patt.4af, %Outer.004 [symbolic]
// CHECK:STDOUT:   %Inner.B.type.ffd: type = fn_type @Inner.B, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %Inner.B.84d: %Inner.B.type.ffd = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.002: type = class_type @Inner, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %.f2f: Core.Form = init_form %Inner.002 [symbolic]
// CHECK:STDOUT:   %pattern_type.0c1: type = pattern_type %Inner.002 [symbolic]
// CHECK:STDOUT:   %return.param_patt.6fe: %pattern_type.0c1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.61f: %pattern_type.0c1 = return_slot_pattern %return.param_patt.6fe, %Inner.002 [symbolic]
// CHECK:STDOUT:   %Inner.C.type.e8a: type = fn_type @Inner.C, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %Inner.C.660: %Inner.C.type.e8a = struct_value () [symbolic]
// CHECK:STDOUT:   %.f1f: Core.Form = init_form %Inner.b8a [symbolic]
// CHECK:STDOUT:   %pattern_type.b11: type = pattern_type %Inner.b8a [symbolic]
// CHECK:STDOUT:   %return.param_patt.49a: %pattern_type.b11 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.936: %pattern_type.b11 = return_slot_pattern %return.param_patt.49a, %Inner.b8a [symbolic]
// CHECK:STDOUT:   %Inner.D.type.177: type = fn_type @Inner.D, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %Inner.D.4bf: %Inner.D.type.177 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.868: <witness> = require_complete_type %Outer.651 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.val.1fd: %Outer.651 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.type.760: type = parameterized_class_name_type @Inner, @Outer(%U) [symbolic]
// CHECK:STDOUT:   %Inner.generic.2d3: %Inner.type.760 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.48e: <witness> = require_complete_type %Outer.004 [symbolic]
// CHECK:STDOUT:   %Outer.val.64e: %Outer.004 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.A.type.b2a: type = fn_type @Inner.A, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %Inner.A.13a: %Inner.A.type.b2a = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.B.type.e39: type = fn_type @Inner.B, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %Inner.B.4e1: %Inner.B.type.e39 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.C.type.398: type = fn_type @Inner.C, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %Inner.C.09d: %Inner.C.type.398 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.D.type.4d2: type = fn_type @Inner.D, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %Inner.D.e8e: %Inner.D.type.4d2 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.071: <witness> = require_complete_type %Inner.002 [symbolic]
// CHECK:STDOUT:   %Inner.val.ef2: %Inner.002 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.cf8: <witness> = require_complete_type %Inner.b8a [symbolic]
// CHECK:STDOUT:   %Inner.val.13f: %Inner.b8a = struct_value () [symbolic]
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
// CHECK:STDOUT:     %T.patt.loc2_14.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc2_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc2_16.1: type = splice_block %.loc2_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc2_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc2_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc2_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc2_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc2_14.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc2_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc2_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc2_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = parameterized_class_name_type @Inner, @Outer(%T.loc2_14.1) [symbolic = %Inner.type (constants.%Inner.type.19b)]
// CHECK:STDOUT:   %Inner.generic: @Outer.%Inner.type (%Inner.type.19b) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.282)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: @Outer.%Inner.type (%Inner.type.19b) = class_decl @Inner [symbolic = @Outer.%Inner.generic (constants.%Inner.generic.282)] {
// CHECK:STDOUT:       %U.patt.loc3_16.1: %pattern_type.9a5 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc3_16.2 (constants.%U.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc3_18.1: type = splice_block %.loc3_18.2 [concrete = type] {
// CHECK:STDOUT:         %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:         %.loc3_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc3_16.2: type = symbolic_binding U, 1 [symbolic = %U.loc3_16.1 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.651
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:     .Outer = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc2_14.2: type, %U.loc3_16.2: type) {
// CHECK:STDOUT:   %U.patt.loc3_16.2: %pattern_type.9a5 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc3_16.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc3_16.1: type = symbolic_binding U, 1 [symbolic = %U.loc3_16.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.A.type: type = fn_type @Inner.A, @Inner(%T, %U.loc3_16.1) [symbolic = %Inner.A.type (constants.%Inner.A.type.af3)]
// CHECK:STDOUT:   %Inner.A: @Inner.%Inner.A.type (%Inner.A.type.af3) = struct_value () [symbolic = %Inner.A (constants.%Inner.A.ff2)]
// CHECK:STDOUT:   %Inner.B.type: type = fn_type @Inner.B, @Inner(%T, %U.loc3_16.1) [symbolic = %Inner.B.type (constants.%Inner.B.type.ffd)]
// CHECK:STDOUT:   %Inner.B: @Inner.%Inner.B.type (%Inner.B.type.ffd) = struct_value () [symbolic = %Inner.B (constants.%Inner.B.84d)]
// CHECK:STDOUT:   %Inner.C.type: type = fn_type @Inner.C, @Inner(%T, %U.loc3_16.1) [symbolic = %Inner.C.type (constants.%Inner.C.type.e8a)]
// CHECK:STDOUT:   %Inner.C: @Inner.%Inner.C.type (%Inner.C.type.e8a) = struct_value () [symbolic = %Inner.C (constants.%Inner.C.660)]
// CHECK:STDOUT:   %Inner.D.type: type = fn_type @Inner.D, @Inner(%T, %U.loc3_16.1) [symbolic = %Inner.D.type (constants.%Inner.D.type.177)]
// CHECK:STDOUT:   %Inner.D: @Inner.%Inner.D.type (%Inner.D.type.177) = struct_value () [symbolic = %Inner.D (constants.%Inner.D.4bf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.A.decl: @Inner.%Inner.A.type (%Inner.A.type.af3) = fn_decl @Inner.A [symbolic = @Inner.%Inner.A (constants.%Inner.A.ff2)] {
// CHECK:STDOUT:       %return.param_patt.loc4_22.1: @Inner.A.%pattern_type (%pattern_type.fc0) = out_param_pattern [symbolic = %return.param_patt.loc4_22.2 (constants.%return.param_patt.e08)]
// CHECK:STDOUT:       %return.patt.loc4_12.1: @Inner.A.%pattern_type (%pattern_type.fc0) = return_slot_pattern %return.param_patt.loc4_22.1, %Outer.loc4_22.2 [symbolic = %return.patt.loc4_12.2 (constants.%return.patt.3d6)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:       %.loc4_21: type = specific_constant @Outer.%T.loc2_14.2, @Outer(constants.%T) [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %.loc4_21 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Outer.loc4_22.2: type = class_type @Outer, @Outer(constants.%T) [symbolic = %Outer.loc4_22.1 (constants.%Outer.651)]
// CHECK:STDOUT:       %.loc4_22.2: Core.Form = init_form %Outer.loc4_22.2 [symbolic = %.loc4_22.1 (constants.%.a0c)]
// CHECK:STDOUT:       %return.param: ref @Inner.A.%Outer.loc4_22.1 (%Outer.651) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Inner.A.%Outer.loc4_22.1 (%Outer.651) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Inner.B.decl: @Inner.%Inner.B.type (%Inner.B.type.ffd) = fn_decl @Inner.B [symbolic = @Inner.%Inner.B (constants.%Inner.B.84d)] {
// CHECK:STDOUT:       %return.param_patt.loc7_22.1: @Inner.B.%pattern_type (%pattern_type.673) = out_param_pattern [symbolic = %return.param_patt.loc7_22.2 (constants.%return.param_patt.4af)]
// CHECK:STDOUT:       %return.patt.loc7_12.1: @Inner.B.%pattern_type (%pattern_type.673) = return_slot_pattern %return.param_patt.loc7_22.1, %Outer.loc7_22.2 [symbolic = %return.patt.loc7_12.2 (constants.%return.patt.2eb)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:       %.loc7_21: type = specific_constant @Inner.%U.loc3_16.2, @Inner(constants.%T, constants.%U) [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %U.ref: type = name_ref U, %.loc7_21 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %Outer.loc7_22.2: type = class_type @Outer, @Outer(constants.%U) [symbolic = %Outer.loc7_22.1 (constants.%Outer.004)]
// CHECK:STDOUT:       %.loc7_22.2: Core.Form = init_form %Outer.loc7_22.2 [symbolic = %.loc7_22.1 (constants.%.38b)]
// CHECK:STDOUT:       %return.param: ref @Inner.B.%Outer.loc7_22.1 (%Outer.004) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Inner.B.%Outer.loc7_22.1 (%Outer.004) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Inner.C.decl: @Inner.%Inner.C.type (%Inner.C.type.e8a) = fn_decl @Inner.C [symbolic = @Inner.%Inner.C (constants.%Inner.C.660)] {
// CHECK:STDOUT:       %return.param_patt.loc10_22.1: @Inner.C.%pattern_type (%pattern_type.0c1) = out_param_pattern [symbolic = %return.param_patt.loc10_22.2 (constants.%return.param_patt.6fe)]
// CHECK:STDOUT:       %return.patt.loc10_12.1: @Inner.C.%pattern_type (%pattern_type.0c1) = return_slot_pattern %return.param_patt.loc10_22.1, %Inner.loc10_22.2 [symbolic = %return.patt.loc10_12.2 (constants.%return.patt.61f)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc10_15: @Inner.C.%Inner.type (%Inner.type.19b) = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.generic (constants.%Inner.generic.282)]
// CHECK:STDOUT:       %Inner.ref: @Inner.C.%Inner.type (%Inner.type.19b) = name_ref Inner, %.loc10_15 [symbolic = %Inner.generic (constants.%Inner.generic.282)]
// CHECK:STDOUT:       %.loc10_21: type = specific_constant @Outer.%T.loc2_14.2, @Outer(constants.%T) [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %.loc10_21 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Inner.loc10_22.2: type = class_type @Inner, @Inner(constants.%T, constants.%T) [symbolic = %Inner.loc10_22.1 (constants.%Inner.002)]
// CHECK:STDOUT:       %.loc10_22.2: Core.Form = init_form %Inner.loc10_22.2 [symbolic = %.loc10_22.1 (constants.%.f2f)]
// CHECK:STDOUT:       %return.param: ref @Inner.C.%Inner.loc10_22.1 (%Inner.002) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Inner.C.%Inner.loc10_22.1 (%Inner.002) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Inner.D.decl: @Inner.%Inner.D.type (%Inner.D.type.177) = fn_decl @Inner.D [symbolic = @Inner.%Inner.D (constants.%Inner.D.4bf)] {
// CHECK:STDOUT:       %return.param_patt.loc13_22.1: @Inner.D.%pattern_type (%pattern_type.b11) = out_param_pattern [symbolic = %return.param_patt.loc13_22.2 (constants.%return.param_patt.49a)]
// CHECK:STDOUT:       %return.patt.loc13_12.1: @Inner.D.%pattern_type (%pattern_type.b11) = return_slot_pattern %return.param_patt.loc13_22.1, %Inner.loc13_22.2 [symbolic = %return.patt.loc13_12.2 (constants.%return.patt.936)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc13_15: @Inner.D.%Inner.type (%Inner.type.19b) = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.generic (constants.%Inner.generic.282)]
// CHECK:STDOUT:       %Inner.ref: @Inner.D.%Inner.type (%Inner.type.19b) = name_ref Inner, %.loc13_15 [symbolic = %Inner.generic (constants.%Inner.generic.282)]
// CHECK:STDOUT:       %.loc13_21: type = specific_constant @Inner.%U.loc3_16.2, @Inner(constants.%T, constants.%U) [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %U.ref: type = name_ref U, %.loc13_21 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %Inner.loc13_22.2: type = class_type @Inner, @Inner(constants.%T, constants.%U) [symbolic = %Inner.loc13_22.1 (constants.%Inner.b8a)]
// CHECK:STDOUT:       %.loc13_22.2: Core.Form = init_form %Inner.loc13_22.2 [symbolic = %.loc13_22.1 (constants.%.f1f)]
// CHECK:STDOUT:       %return.param: ref @Inner.D.%Inner.loc13_22.1 (%Inner.b8a) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Inner.D.%Inner.loc13_22.1 (%Inner.b8a) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.b8a
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
// CHECK:STDOUT:   %Outer.loc4_22.1: type = class_type @Outer, @Outer(%T) [symbolic = %Outer.loc4_22.1 (constants.%Outer.651)]
// CHECK:STDOUT:   %.loc4_22.1: Core.Form = init_form %Outer.loc4_22.1 [symbolic = %.loc4_22.1 (constants.%.a0c)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Outer.loc4_22.1 [symbolic = %pattern_type (constants.%pattern_type.fc0)]
// CHECK:STDOUT:   %return.param_patt.loc4_22.2: @Inner.A.%pattern_type (%pattern_type.fc0) = out_param_pattern [symbolic = %return.param_patt.loc4_22.2 (constants.%return.param_patt.e08)]
// CHECK:STDOUT:   %return.patt.loc4_12.2: @Inner.A.%pattern_type (%pattern_type.fc0) = return_slot_pattern %return.param_patt.loc4_22.2, %Outer.loc4_22.1 [symbolic = %return.patt.loc4_12.2 (constants.%return.patt.3d6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Outer.loc4_22.1 [symbolic = %require_complete (constants.%require_complete.868)]
// CHECK:STDOUT:   %Outer.val: @Inner.A.%Outer.loc4_22.1 (%Outer.651) = struct_value () [symbolic = %Outer.val (constants.%Outer.val.1fd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Inner.A.%Outer.loc4_22.1 (%Outer.651) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc5_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc5_15.2: init @Inner.A.%Outer.loc4_22.1 (%Outer.651) to %return.param = class_init () [symbolic = %Outer.val (constants.%Outer.val.1fd)]
// CHECK:STDOUT:     %.loc5_16: init @Inner.A.%Outer.loc4_22.1 (%Outer.651) = converted %.loc5_15.1, %.loc5_15.2 [symbolic = %Outer.val (constants.%Outer.val.1fd)]
// CHECK:STDOUT:     return %.loc5_16 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.B(@Outer.%T.loc2_14.2: type, @Inner.%U.loc3_16.2: type) {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %Outer.loc7_22.1: type = class_type @Outer, @Outer(%U) [symbolic = %Outer.loc7_22.1 (constants.%Outer.004)]
// CHECK:STDOUT:   %.loc7_22.1: Core.Form = init_form %Outer.loc7_22.1 [symbolic = %.loc7_22.1 (constants.%.38b)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Outer.loc7_22.1 [symbolic = %pattern_type (constants.%pattern_type.673)]
// CHECK:STDOUT:   %return.param_patt.loc7_22.2: @Inner.B.%pattern_type (%pattern_type.673) = out_param_pattern [symbolic = %return.param_patt.loc7_22.2 (constants.%return.param_patt.4af)]
// CHECK:STDOUT:   %return.patt.loc7_12.2: @Inner.B.%pattern_type (%pattern_type.673) = return_slot_pattern %return.param_patt.loc7_22.2, %Outer.loc7_22.1 [symbolic = %return.patt.loc7_12.2 (constants.%return.patt.2eb)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Outer.loc7_22.1 [symbolic = %require_complete (constants.%require_complete.48e)]
// CHECK:STDOUT:   %Outer.val: @Inner.B.%Outer.loc7_22.1 (%Outer.004) = struct_value () [symbolic = %Outer.val (constants.%Outer.val.64e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Inner.B.%Outer.loc7_22.1 (%Outer.004) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc8_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_15.2: init @Inner.B.%Outer.loc7_22.1 (%Outer.004) to %return.param = class_init () [symbolic = %Outer.val (constants.%Outer.val.64e)]
// CHECK:STDOUT:     %.loc8_16: init @Inner.B.%Outer.loc7_22.1 (%Outer.004) = converted %.loc8_15.1, %.loc8_15.2 [symbolic = %Outer.val (constants.%Outer.val.64e)]
// CHECK:STDOUT:     return %.loc8_16 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.C(@Outer.%T.loc2_14.2: type, @Inner.%U.loc3_16.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = parameterized_class_name_type @Inner, @Outer(%T) [symbolic = %Inner.type (constants.%Inner.type.19b)]
// CHECK:STDOUT:   %Inner.generic: @Inner.C.%Inner.type (%Inner.type.19b) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.282)]
// CHECK:STDOUT:   %Inner.loc10_22.1: type = class_type @Inner, @Inner(%T, %T) [symbolic = %Inner.loc10_22.1 (constants.%Inner.002)]
// CHECK:STDOUT:   %.loc10_22.1: Core.Form = init_form %Inner.loc10_22.1 [symbolic = %.loc10_22.1 (constants.%.f2f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Inner.loc10_22.1 [symbolic = %pattern_type (constants.%pattern_type.0c1)]
// CHECK:STDOUT:   %return.param_patt.loc10_22.2: @Inner.C.%pattern_type (%pattern_type.0c1) = out_param_pattern [symbolic = %return.param_patt.loc10_22.2 (constants.%return.param_patt.6fe)]
// CHECK:STDOUT:   %return.patt.loc10_12.2: @Inner.C.%pattern_type (%pattern_type.0c1) = return_slot_pattern %return.param_patt.loc10_22.2, %Inner.loc10_22.1 [symbolic = %return.patt.loc10_12.2 (constants.%return.patt.61f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Inner.loc10_22.1 [symbolic = %require_complete (constants.%require_complete.071)]
// CHECK:STDOUT:   %Inner.val: @Inner.C.%Inner.loc10_22.1 (%Inner.002) = struct_value () [symbolic = %Inner.val (constants.%Inner.val.ef2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Inner.C.%Inner.loc10_22.1 (%Inner.002) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc11_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc11_15.2: init @Inner.C.%Inner.loc10_22.1 (%Inner.002) to %return.param = class_init () [symbolic = %Inner.val (constants.%Inner.val.ef2)]
// CHECK:STDOUT:     %.loc11_16: init @Inner.C.%Inner.loc10_22.1 (%Inner.002) = converted %.loc11_15.1, %.loc11_15.2 [symbolic = %Inner.val (constants.%Inner.val.ef2)]
// CHECK:STDOUT:     return %.loc11_16 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.D(@Outer.%T.loc2_14.2: type, @Inner.%U.loc3_16.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = parameterized_class_name_type @Inner, @Outer(%T) [symbolic = %Inner.type (constants.%Inner.type.19b)]
// CHECK:STDOUT:   %Inner.generic: @Inner.D.%Inner.type (%Inner.type.19b) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.282)]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %Inner.loc13_22.1: type = class_type @Inner, @Inner(%T, %U) [symbolic = %Inner.loc13_22.1 (constants.%Inner.b8a)]
// CHECK:STDOUT:   %.loc13_22.1: Core.Form = init_form %Inner.loc13_22.1 [symbolic = %.loc13_22.1 (constants.%.f1f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Inner.loc13_22.1 [symbolic = %pattern_type (constants.%pattern_type.b11)]
// CHECK:STDOUT:   %return.param_patt.loc13_22.2: @Inner.D.%pattern_type (%pattern_type.b11) = out_param_pattern [symbolic = %return.param_patt.loc13_22.2 (constants.%return.param_patt.49a)]
// CHECK:STDOUT:   %return.patt.loc13_12.2: @Inner.D.%pattern_type (%pattern_type.b11) = return_slot_pattern %return.param_patt.loc13_22.2, %Inner.loc13_22.1 [symbolic = %return.patt.loc13_12.2 (constants.%return.patt.936)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Inner.loc13_22.1 [symbolic = %require_complete (constants.%require_complete.cf8)]
// CHECK:STDOUT:   %Inner.val: @Inner.D.%Inner.loc13_22.1 (%Inner.b8a) = struct_value () [symbolic = %Inner.val (constants.%Inner.val.13f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Inner.D.%Inner.loc13_22.1 (%Inner.b8a) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc14_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc14_15.2: init @Inner.D.%Inner.loc13_22.1 (%Inner.b8a) to %return.param = class_init () [symbolic = %Inner.val (constants.%Inner.val.13f)]
// CHECK:STDOUT:     %.loc14_16: init @Inner.D.%Inner.loc13_22.1 (%Inner.b8a) = converted %.loc14_15.1, %.loc14_15.2 [symbolic = %Inner.val (constants.%Inner.val.13f)]
// CHECK:STDOUT:     return %.loc14_16 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc2_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc2_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.19b
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.282
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.patt.loc3_16.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc3_16.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.A.type => constants.%Inner.A.type.af3
// CHECK:STDOUT:   %Inner.A => constants.%Inner.A.ff2
// CHECK:STDOUT:   %Inner.B.type => constants.%Inner.B.type.ffd
// CHECK:STDOUT:   %Inner.B => constants.%Inner.B.84d
// CHECK:STDOUT:   %Inner.C.type => constants.%Inner.C.type.e8a
// CHECK:STDOUT:   %Inner.C => constants.%Inner.C.660
// CHECK:STDOUT:   %Inner.D.type => constants.%Inner.D.type.177
// CHECK:STDOUT:   %Inner.D => constants.%Inner.D.4bf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.A(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Outer.loc4_22.1 => constants.%Outer.651
// CHECK:STDOUT:   %.loc4_22.1 => constants.%.a0c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fc0
// CHECK:STDOUT:   %return.param_patt.loc4_22.2 => constants.%return.param_patt.e08
// CHECK:STDOUT:   %return.patt.loc4_12.2 => constants.%return.patt.3d6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%U) {
// CHECK:STDOUT:   %T.patt.loc2_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc2_14.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.760
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.2d3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.B(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %Outer.loc7_22.1 => constants.%Outer.004
// CHECK:STDOUT:   %.loc7_22.1 => constants.%.38b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.673
// CHECK:STDOUT:   %return.param_patt.loc7_22.2 => constants.%return.param_patt.4af
// CHECK:STDOUT:   %return.patt.loc7_12.2 => constants.%return.patt.2eb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T, constants.%T) {
// CHECK:STDOUT:   %U.patt.loc3_16.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc3_16.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.A.type => constants.%Inner.A.type.b2a
// CHECK:STDOUT:   %Inner.A => constants.%Inner.A.13a
// CHECK:STDOUT:   %Inner.B.type => constants.%Inner.B.type.e39
// CHECK:STDOUT:   %Inner.B => constants.%Inner.B.4e1
// CHECK:STDOUT:   %Inner.C.type => constants.%Inner.C.type.398
// CHECK:STDOUT:   %Inner.C => constants.%Inner.C.09d
// CHECK:STDOUT:   %Inner.D.type => constants.%Inner.D.type.4d2
// CHECK:STDOUT:   %Inner.D => constants.%Inner.D.e8e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.C(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.19b
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.282
// CHECK:STDOUT:   %Inner.loc10_22.1 => constants.%Inner.002
// CHECK:STDOUT:   %.loc10_22.1 => constants.%.f2f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0c1
// CHECK:STDOUT:   %return.param_patt.loc10_22.2 => constants.%return.param_patt.6fe
// CHECK:STDOUT:   %return.patt.loc10_12.2 => constants.%return.patt.61f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.D(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.19b
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.282
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %Inner.loc13_22.1 => constants.%Inner.b8a
// CHECK:STDOUT:   %.loc13_22.1 => constants.%.f1f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b11
// CHECK:STDOUT:   %return.param_patt.loc13_22.2 => constants.%return.param_patt.49a
// CHECK:STDOUT:   %return.patt.loc13_12.2 => constants.%return.patt.936
// CHECK:STDOUT: }
// CHECK:STDOUT:
