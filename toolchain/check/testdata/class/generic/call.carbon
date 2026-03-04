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
// CHECK:STDERR:             ^
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
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.6f3: %i32 = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.e86: type = class_type @Class, @Class(%T, %N.6f3) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.e9d: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Class.ff1: type = class_type @Class, @Class(%ptr.235, %int_5.0f6) [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %Class.ff1 [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.d2e: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Class.7c6: type = class_type @Class, @Class(%empty_tuple.type, %int_0.6a9) [concrete]
// CHECK:STDOUT:   %pattern_type.a3c: type = pattern_type %Class.7c6 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %N.patt: %pattern_type.7ce = symbolic_binding_pattern N, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4_27: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_23.2: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_23.1 (constants.%N.6f3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d58 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.d58 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Class.ff1 = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc6_21.1: type = splice_block %Class.loc6 [concrete = constants.%Class.ff1] {
// CHECK:STDOUT:     %Class.ref.loc6: %Class.type = name_ref Class, %Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32 [concrete = constants.%ptr.235]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:     %impl.elem0.loc6: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc6_21.1: <bound method> = bound_method %int_5, %impl.elem0.loc6 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005]
// CHECK:STDOUT:     %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_21.2: <bound method> = bound_method %int_5, %specific_fn.loc6 [concrete = constants.%bound_method.e9d]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %i32 = call %bound_method.loc6_21.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %.loc6_21.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %.loc6_21.3: %i32 = converted %int_5, %.loc6_21.2 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %Class.loc6: type = class_type @Class, @Class(constants.%ptr.235, constants.%int_5.0f6) [concrete = constants.%Class.ff1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Class.ff1 = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.a3c = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.a3c = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Class.7c6 = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc9_19.1: type = splice_block %Class.loc9 [concrete = constants.%Class.7c6] {
// CHECK:STDOUT:     %Class.ref.loc9: %Class.type = name_ref Class, %Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %.loc9_15: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %.loc9_19.2: type = converted %.loc9_15, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %impl.elem0.loc9: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc9_19.1: <bound method> = bound_method %int_0, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:     %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_19.2: <bound method> = bound_method %int_0, %specific_fn.loc9 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_19.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc9_19.3: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc9_19.4: %i32 = converted %int_0, %.loc9_19.3 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %Class.loc9: type = class_type @Class, @Class(constants.%empty_tuple.type, constants.%int_0.6a9) [concrete = constants.%Class.7c6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Class.7c6 = ref_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_13.2: type, %N.loc4_23.2: %i32) {
// CHECK:STDOUT:   %T.loc4_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:   %N.loc4_23.1: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_23.1 (constants.%N.6f3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.e86
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N.6f3) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT:   %N.loc4_23.1 => constants.%N.6f3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%ptr.235, constants.%int_5.0f6) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%ptr.235
// CHECK:STDOUT:   %N.loc4_23.1 => constants.%int_5.0f6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%empty_tuple.type, constants.%int_0.6a9) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%empty_tuple.type
// CHECK:STDOUT:   %N.loc4_23.1 => constants.%int_0.6a9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_few.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.6f3: %i32 = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T, %N.6f3) [symbolic]
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
// CHECK:STDOUT:     %N.patt: %pattern_type.7ce = symbolic_binding_pattern N, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4_27: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_23.2: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_23.1 (constants.%N.6f3)]
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
// CHECK:STDOUT: generic class @Class(%T.loc4_13.2: type, %N.loc4_23.2: %i32) {
// CHECK:STDOUT:   %T.loc4_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:   %N.loc4_23.1: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_23.1 (constants.%N.6f3)]
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
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N.6f3) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT:   %N.loc4_23.1 => constants.%N.6f3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_many.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.6f3: %i32 = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T, %N.6f3) [symbolic]
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
// CHECK:STDOUT:     %N.patt: %pattern_type.7ce = symbolic_binding_pattern N, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4_27: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_23.2: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_23.1 (constants.%N.6f3)]
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
// CHECK:STDOUT: generic class @Class(%T.loc4_13.2: type, %N.loc4_23.2: %i32) {
// CHECK:STDOUT:   %T.loc4_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:   %N.loc4_23.1: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_23.1 (constants.%N.6f3)]
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
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N.6f3) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT:   %N.loc4_23.1 => constants.%N.6f3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.6f3: %i32 = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T, %N.6f3) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
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
// CHECK:STDOUT:     %N.patt: %pattern_type.7ce = symbolic_binding_pattern N, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4_27: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_23.2: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_23.1 (constants.%N.6f3)]
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
// CHECK:STDOUT: generic class @Class(%T.loc4_13.2: type, %N.loc4_23.2: %i32) {
// CHECK:STDOUT:   %T.loc4_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:   %N.loc4_23.1: %i32 = symbolic_binding N, 1 [symbolic = %N.loc4_23.1 (constants.%N.6f3)]
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
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N.6f3) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT:   %N.loc4_23.1 => constants.%N.6f3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_in_nested_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.387: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Inner.type.e0d: type = generic_class_type @Inner, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.generic.ada: %Inner.type.e0d = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.e21: type = class_type @Inner, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %.e35: Core.Form = init_form %Outer.387, call_param0 [symbolic]
// CHECK:STDOUT:   %pattern_type.130: type = pattern_type %Outer.387 [symbolic]
// CHECK:STDOUT:   %Inner.A.type.c2f: type = fn_type @Inner.A, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %Inner.A.07b: %Inner.A.type.c2f = struct_value () [symbolic]
// CHECK:STDOUT:   %Outer.d2f: type = class_type @Outer, @Outer(%U) [symbolic]
// CHECK:STDOUT:   %.66e: Core.Form = init_form %Outer.d2f, call_param0 [symbolic]
// CHECK:STDOUT:   %pattern_type.760: type = pattern_type %Outer.d2f [symbolic]
// CHECK:STDOUT:   %Inner.B.type.f42: type = fn_type @Inner.B, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %Inner.B.b14: %Inner.B.type.f42 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.ddc: type = class_type @Inner, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %.024: Core.Form = init_form %Inner.ddc, call_param0 [symbolic]
// CHECK:STDOUT:   %pattern_type.81e: type = pattern_type %Inner.ddc [symbolic]
// CHECK:STDOUT:   %Inner.C.type.912: type = fn_type @Inner.C, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %Inner.C.125: %Inner.C.type.912 = struct_value () [symbolic]
// CHECK:STDOUT:   %.8b8: Core.Form = init_form %Inner.e21, call_param0 [symbolic]
// CHECK:STDOUT:   %pattern_type.0d1: type = pattern_type %Inner.e21 [symbolic]
// CHECK:STDOUT:   %Inner.D.type.cd0: type = fn_type @Inner.D, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %Inner.D.147: %Inner.D.type.cd0 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.448: <witness> = require_complete_type %Outer.387 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.val.ca1: %Outer.387 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.type.26a: type = generic_class_type @Inner, @Outer(%U) [symbolic]
// CHECK:STDOUT:   %Inner.generic.ddc: %Inner.type.26a = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.ade: <witness> = require_complete_type %Outer.d2f [symbolic]
// CHECK:STDOUT:   %Outer.val.756: %Outer.d2f = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.A.type.e74: type = fn_type @Inner.A, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %Inner.A.d94: %Inner.A.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.B.type.539: type = fn_type @Inner.B, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %Inner.B.455: %Inner.B.type.539 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.C.type.4b3: type = fn_type @Inner.C, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %Inner.C.f04: %Inner.C.type.4b3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.D.type.d17: type = fn_type @Inner.D, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %Inner.D.c1c: %Inner.D.type.d17 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.de9: <witness> = require_complete_type %Inner.ddc [symbolic]
// CHECK:STDOUT:   %Inner.val.240: %Inner.ddc = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.41c: <witness> = require_complete_type %Inner.e21 [symbolic]
// CHECK:STDOUT:   %Inner.val.43a: %Inner.e21 = struct_value () [symbolic]
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
// CHECK:STDOUT:     %T.loc2_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc2_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc2_13.2: type) {
// CHECK:STDOUT:   %T.loc2_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc2_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T.loc2_13.1) [symbolic = %Inner.type (constants.%Inner.type.e0d)]
// CHECK:STDOUT:   %Inner.generic: @Outer.%Inner.type (%Inner.type.e0d) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.ada)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: @Outer.%Inner.type (%Inner.type.e0d) = class_decl @Inner [symbolic = @Outer.%Inner.generic (constants.%Inner.generic.ada)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc3_19.1: type = splice_block %.loc3_19.2 [concrete = type] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %.loc3_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc3_15.2: type = symbolic_binding U, 1 [symbolic = %U.loc3_15.1 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.387
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:     .Outer = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc2_13.2: type, %U.loc3_15.2: type) {
// CHECK:STDOUT:   %U.loc3_15.1: type = symbolic_binding U, 1 [symbolic = %U.loc3_15.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.A.type: type = fn_type @Inner.A, @Inner(%T, %U.loc3_15.1) [symbolic = %Inner.A.type (constants.%Inner.A.type.c2f)]
// CHECK:STDOUT:   %Inner.A: @Inner.%Inner.A.type (%Inner.A.type.c2f) = struct_value () [symbolic = %Inner.A (constants.%Inner.A.07b)]
// CHECK:STDOUT:   %Inner.B.type: type = fn_type @Inner.B, @Inner(%T, %U.loc3_15.1) [symbolic = %Inner.B.type (constants.%Inner.B.type.f42)]
// CHECK:STDOUT:   %Inner.B: @Inner.%Inner.B.type (%Inner.B.type.f42) = struct_value () [symbolic = %Inner.B (constants.%Inner.B.b14)]
// CHECK:STDOUT:   %Inner.C.type: type = fn_type @Inner.C, @Inner(%T, %U.loc3_15.1) [symbolic = %Inner.C.type (constants.%Inner.C.type.912)]
// CHECK:STDOUT:   %Inner.C: @Inner.%Inner.C.type (%Inner.C.type.912) = struct_value () [symbolic = %Inner.C (constants.%Inner.C.125)]
// CHECK:STDOUT:   %Inner.D.type: type = fn_type @Inner.D, @Inner(%T, %U.loc3_15.1) [symbolic = %Inner.D.type (constants.%Inner.D.type.cd0)]
// CHECK:STDOUT:   %Inner.D: @Inner.%Inner.D.type (%Inner.D.type.cd0) = struct_value () [symbolic = %Inner.D (constants.%Inner.D.147)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.A.decl: @Inner.%Inner.A.type (%Inner.A.type.c2f) = fn_decl @Inner.A [symbolic = @Inner.%Inner.A (constants.%Inner.A.07b)] {
// CHECK:STDOUT:       %return.patt: @Inner.A.%pattern_type (%pattern_type.130) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Inner.A.%pattern_type (%pattern_type.130) = out_param_pattern %return.patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc2_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Outer.loc4_22.2: type = class_type @Outer, @Outer(constants.%T) [symbolic = %Outer.loc4_22.1 (constants.%Outer.387)]
// CHECK:STDOUT:       %.loc4_22.2: Core.Form = init_form %Outer.loc4_22.2, call_param0 [symbolic = %.loc4_22.1 (constants.%.e35)]
// CHECK:STDOUT:       %return.param: ref @Inner.A.%Outer.loc4_22.1 (%Outer.387) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Inner.A.%Outer.loc4_22.1 (%Outer.387) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Inner.B.decl: @Inner.%Inner.B.type (%Inner.B.type.f42) = fn_decl @Inner.B [symbolic = @Inner.%Inner.B (constants.%Inner.B.b14)] {
// CHECK:STDOUT:       %return.patt: @Inner.B.%pattern_type (%pattern_type.760) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Inner.B.%pattern_type (%pattern_type.760) = out_param_pattern %return.patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @Inner.%U.loc3_15.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %Outer.loc7_22.2: type = class_type @Outer, @Outer(constants.%U) [symbolic = %Outer.loc7_22.1 (constants.%Outer.d2f)]
// CHECK:STDOUT:       %.loc7_22.2: Core.Form = init_form %Outer.loc7_22.2, call_param0 [symbolic = %.loc7_22.1 (constants.%.66e)]
// CHECK:STDOUT:       %return.param: ref @Inner.B.%Outer.loc7_22.1 (%Outer.d2f) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Inner.B.%Outer.loc7_22.1 (%Outer.d2f) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Inner.C.decl: @Inner.%Inner.C.type (%Inner.C.type.912) = fn_decl @Inner.C [symbolic = @Inner.%Inner.C (constants.%Inner.C.125)] {
// CHECK:STDOUT:       %return.patt: @Inner.C.%pattern_type (%pattern_type.81e) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Inner.C.%pattern_type (%pattern_type.81e) = out_param_pattern %return.patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc10_15: @Inner.C.%Inner.type (%Inner.type.e0d) = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.generic (constants.%Inner.generic.ada)]
// CHECK:STDOUT:       %Inner.ref: @Inner.C.%Inner.type (%Inner.type.e0d) = name_ref Inner, %.loc10_15 [symbolic = %Inner.generic (constants.%Inner.generic.ada)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc2_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Inner.loc10_22.2: type = class_type @Inner, @Inner(constants.%T, constants.%T) [symbolic = %Inner.loc10_22.1 (constants.%Inner.ddc)]
// CHECK:STDOUT:       %.loc10_22.2: Core.Form = init_form %Inner.loc10_22.2, call_param0 [symbolic = %.loc10_22.1 (constants.%.024)]
// CHECK:STDOUT:       %return.param: ref @Inner.C.%Inner.loc10_22.1 (%Inner.ddc) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Inner.C.%Inner.loc10_22.1 (%Inner.ddc) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Inner.D.decl: @Inner.%Inner.D.type (%Inner.D.type.cd0) = fn_decl @Inner.D [symbolic = @Inner.%Inner.D (constants.%Inner.D.147)] {
// CHECK:STDOUT:       %return.patt: @Inner.D.%pattern_type (%pattern_type.0d1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Inner.D.%pattern_type (%pattern_type.0d1) = out_param_pattern %return.patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc13_15: @Inner.D.%Inner.type (%Inner.type.e0d) = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.generic (constants.%Inner.generic.ada)]
// CHECK:STDOUT:       %Inner.ref: @Inner.D.%Inner.type (%Inner.type.e0d) = name_ref Inner, %.loc13_15 [symbolic = %Inner.generic (constants.%Inner.generic.ada)]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @Inner.%U.loc3_15.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %Inner.loc13_22.2: type = class_type @Inner, @Inner(constants.%T, constants.%U) [symbolic = %Inner.loc13_22.1 (constants.%Inner.e21)]
// CHECK:STDOUT:       %.loc13_22.2: Core.Form = init_form %Inner.loc13_22.2, call_param0 [symbolic = %.loc13_22.1 (constants.%.8b8)]
// CHECK:STDOUT:       %return.param: ref @Inner.D.%Inner.loc13_22.1 (%Inner.e21) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Inner.D.%Inner.loc13_22.1 (%Inner.e21) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.e21
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
// CHECK:STDOUT: generic fn @Inner.A(@Outer.%T.loc2_13.2: type, @Inner.%U.loc3_15.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Outer.loc4_22.1: type = class_type @Outer, @Outer(%T) [symbolic = %Outer.loc4_22.1 (constants.%Outer.387)]
// CHECK:STDOUT:   %.loc4_22.1: Core.Form = init_form %Outer.loc4_22.1, call_param0 [symbolic = %.loc4_22.1 (constants.%.e35)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Outer.loc4_22.1 [symbolic = %pattern_type (constants.%pattern_type.130)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Outer.loc4_22.1 [symbolic = %require_complete (constants.%require_complete.448)]
// CHECK:STDOUT:   %Outer.val: @Inner.A.%Outer.loc4_22.1 (%Outer.387) = struct_value () [symbolic = %Outer.val (constants.%Outer.val.ca1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Inner.A.%Outer.loc4_22.1 (%Outer.387) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc5_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc5_15.2: init @Inner.A.%Outer.loc4_22.1 (%Outer.387) to %return.param = class_init () [symbolic = %Outer.val (constants.%Outer.val.ca1)]
// CHECK:STDOUT:     %.loc5_16: init @Inner.A.%Outer.loc4_22.1 (%Outer.387) = converted %.loc5_15.1, %.loc5_15.2 [symbolic = %Outer.val (constants.%Outer.val.ca1)]
// CHECK:STDOUT:     return %.loc5_16 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.B(@Outer.%T.loc2_13.2: type, @Inner.%U.loc3_15.2: type) {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %Outer.loc7_22.1: type = class_type @Outer, @Outer(%U) [symbolic = %Outer.loc7_22.1 (constants.%Outer.d2f)]
// CHECK:STDOUT:   %.loc7_22.1: Core.Form = init_form %Outer.loc7_22.1, call_param0 [symbolic = %.loc7_22.1 (constants.%.66e)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Outer.loc7_22.1 [symbolic = %pattern_type (constants.%pattern_type.760)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Outer.loc7_22.1 [symbolic = %require_complete (constants.%require_complete.ade)]
// CHECK:STDOUT:   %Outer.val: @Inner.B.%Outer.loc7_22.1 (%Outer.d2f) = struct_value () [symbolic = %Outer.val (constants.%Outer.val.756)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Inner.B.%Outer.loc7_22.1 (%Outer.d2f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc8_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_15.2: init @Inner.B.%Outer.loc7_22.1 (%Outer.d2f) to %return.param = class_init () [symbolic = %Outer.val (constants.%Outer.val.756)]
// CHECK:STDOUT:     %.loc8_16: init @Inner.B.%Outer.loc7_22.1 (%Outer.d2f) = converted %.loc8_15.1, %.loc8_15.2 [symbolic = %Outer.val (constants.%Outer.val.756)]
// CHECK:STDOUT:     return %.loc8_16 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.C(@Outer.%T.loc2_13.2: type, @Inner.%U.loc3_15.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T) [symbolic = %Inner.type (constants.%Inner.type.e0d)]
// CHECK:STDOUT:   %Inner.generic: @Inner.C.%Inner.type (%Inner.type.e0d) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.ada)]
// CHECK:STDOUT:   %Inner.loc10_22.1: type = class_type @Inner, @Inner(%T, %T) [symbolic = %Inner.loc10_22.1 (constants.%Inner.ddc)]
// CHECK:STDOUT:   %.loc10_22.1: Core.Form = init_form %Inner.loc10_22.1, call_param0 [symbolic = %.loc10_22.1 (constants.%.024)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Inner.loc10_22.1 [symbolic = %pattern_type (constants.%pattern_type.81e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Inner.loc10_22.1 [symbolic = %require_complete (constants.%require_complete.de9)]
// CHECK:STDOUT:   %Inner.val: @Inner.C.%Inner.loc10_22.1 (%Inner.ddc) = struct_value () [symbolic = %Inner.val (constants.%Inner.val.240)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Inner.C.%Inner.loc10_22.1 (%Inner.ddc) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc11_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc11_15.2: init @Inner.C.%Inner.loc10_22.1 (%Inner.ddc) to %return.param = class_init () [symbolic = %Inner.val (constants.%Inner.val.240)]
// CHECK:STDOUT:     %.loc11_16: init @Inner.C.%Inner.loc10_22.1 (%Inner.ddc) = converted %.loc11_15.1, %.loc11_15.2 [symbolic = %Inner.val (constants.%Inner.val.240)]
// CHECK:STDOUT:     return %.loc11_16 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.D(@Outer.%T.loc2_13.2: type, @Inner.%U.loc3_15.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T) [symbolic = %Inner.type (constants.%Inner.type.e0d)]
// CHECK:STDOUT:   %Inner.generic: @Inner.D.%Inner.type (%Inner.type.e0d) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.ada)]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %Inner.loc13_22.1: type = class_type @Inner, @Inner(%T, %U) [symbolic = %Inner.loc13_22.1 (constants.%Inner.e21)]
// CHECK:STDOUT:   %.loc13_22.1: Core.Form = init_form %Inner.loc13_22.1, call_param0 [symbolic = %.loc13_22.1 (constants.%.8b8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Inner.loc13_22.1 [symbolic = %pattern_type (constants.%pattern_type.0d1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Inner.loc13_22.1 [symbolic = %require_complete (constants.%require_complete.41c)]
// CHECK:STDOUT:   %Inner.val: @Inner.D.%Inner.loc13_22.1 (%Inner.e21) = struct_value () [symbolic = %Inner.val (constants.%Inner.val.43a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Inner.D.%Inner.loc13_22.1 (%Inner.e21) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc14_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc14_15.2: init @Inner.D.%Inner.loc13_22.1 (%Inner.e21) to %return.param = class_init () [symbolic = %Inner.val (constants.%Inner.val.43a)]
// CHECK:STDOUT:     %.loc14_16: init @Inner.D.%Inner.loc13_22.1 (%Inner.e21) = converted %.loc14_15.1, %.loc14_15.2 [symbolic = %Inner.val (constants.%Inner.val.43a)]
// CHECK:STDOUT:     return %.loc14_16 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.loc2_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.e0d
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.ada
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc3_15.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.A.type => constants.%Inner.A.type.c2f
// CHECK:STDOUT:   %Inner.A => constants.%Inner.A.07b
// CHECK:STDOUT:   %Inner.B.type => constants.%Inner.B.type.f42
// CHECK:STDOUT:   %Inner.B => constants.%Inner.B.b14
// CHECK:STDOUT:   %Inner.C.type => constants.%Inner.C.type.912
// CHECK:STDOUT:   %Inner.C => constants.%Inner.C.125
// CHECK:STDOUT:   %Inner.D.type => constants.%Inner.D.type.cd0
// CHECK:STDOUT:   %Inner.D => constants.%Inner.D.147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.A(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Outer.loc4_22.1 => constants.%Outer.387
// CHECK:STDOUT:   %.loc4_22.1 => constants.%.e35
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%U) {
// CHECK:STDOUT:   %T.loc2_13.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.26a
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.ddc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.B(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %Outer.loc7_22.1 => constants.%Outer.d2f
// CHECK:STDOUT:   %.loc7_22.1 => constants.%.66e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.760
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T, constants.%T) {
// CHECK:STDOUT:   %U.loc3_15.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.A.type => constants.%Inner.A.type.e74
// CHECK:STDOUT:   %Inner.A => constants.%Inner.A.d94
// CHECK:STDOUT:   %Inner.B.type => constants.%Inner.B.type.539
// CHECK:STDOUT:   %Inner.B => constants.%Inner.B.455
// CHECK:STDOUT:   %Inner.C.type => constants.%Inner.C.type.4b3
// CHECK:STDOUT:   %Inner.C => constants.%Inner.C.f04
// CHECK:STDOUT:   %Inner.D.type => constants.%Inner.D.type.d17
// CHECK:STDOUT:   %Inner.D => constants.%Inner.D.c1c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.C(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.e0d
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.ada
// CHECK:STDOUT:   %Inner.loc10_22.1 => constants.%Inner.ddc
// CHECK:STDOUT:   %.loc10_22.1 => constants.%.024
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.81e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.D(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.e0d
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.ada
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %Inner.loc13_22.1 => constants.%Inner.e21
// CHECK:STDOUT:   %.loc13_22.1 => constants.%.8b8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0d1
// CHECK:STDOUT: }
// CHECK:STDOUT:
