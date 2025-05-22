// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/call.carbon

// --- call.carbon

library "[[@TEST_NAME]]";

class Class(T:! type, N:! i32) {}

//@dump-sem-ir-begin
var a: Class(i32*, 5);

// Requires an implicit conversion to type `type`.
var b: Class((), 0);
//@dump-sem-ir-end

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
// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+7]]:8: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.4e6: <bound method> = bound_method %int_5.64b, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.a25: <bound method> = bound_method %int_5.64b, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Class.f29: type = class_type @Class, @Class(%ptr.235, %int_5.0f6) [concrete]
// CHECK:STDOUT:   %pattern_type.d51: type = pattern_type %Class.f29 [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Convert.bound.d04: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Class.dd4: type = class_type @Class, @Class(%empty_tuple.type, %int_0.6a9) [concrete]
// CHECK:STDOUT:   %pattern_type.c73: type = pattern_type %Class.dd4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d51 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.d51 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Class.f29 = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc7_21.1: type = splice_block %Class.loc7 [concrete = constants.%Class.f29] {
// CHECK:STDOUT:     %Class.ref.loc7: %Class.type = name_ref Class, %Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32 [concrete = constants.%ptr.235]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:     %impl.elem0.loc7: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc7_21.1: <bound method> = bound_method %int_5, %impl.elem0.loc7 [concrete = constants.%Convert.bound.4e6]
// CHECK:STDOUT:     %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc7_21.2: <bound method> = bound_method %int_5, %specific_fn.loc7 [concrete = constants.%bound_method.a25]
// CHECK:STDOUT:     %int.convert_checked.loc7: init %i32 = call %bound_method.loc7_21.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %.loc7_21.2: %i32 = value_of_initializer %int.convert_checked.loc7 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %.loc7_21.3: %i32 = converted %int_5, %.loc7_21.2 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %Class.loc7: type = class_type @Class, @Class(constants.%ptr.235, constants.%int_5.0f6) [concrete = constants.%Class.f29]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Class.f29 = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.c73 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.c73 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Class.dd4 = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc10_19.1: type = splice_block %Class.loc10 [concrete = constants.%Class.dd4] {
// CHECK:STDOUT:     %Class.ref.loc10: %Class.type = name_ref Class, %Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %.loc10_15: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %.loc10_19.2: type = converted %.loc10_15, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %impl.elem0.loc10: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc10_19.1: <bound method> = bound_method %int_0, %impl.elem0.loc10 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:     %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_19.2: <bound method> = bound_method %int_0, %specific_fn.loc10 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:     %int.convert_checked.loc10: init %i32 = call %bound_method.loc10_19.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc10_19.3: %i32 = value_of_initializer %int.convert_checked.loc10 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc10_19.4: %i32 = converted %int_0, %.loc10_19.3 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %Class.loc10: type = class_type @Class, @Class(constants.%empty_tuple.type, constants.%int_0.6a9) [concrete = constants.%Class.dd4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %Class.dd4 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
