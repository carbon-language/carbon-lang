// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/prefer_unqualified_lookup.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/prefer_unqualified_lookup.carbon

// --- prefer_unqualified_lookup.carbon

library "[[@TEST_NAME]]";

// F in the lexical scope.
class Class(F:! type) {
  class Inner {
    // F in a non-lexical scope.
    fn F() -> i32 { return 0; }
    fn G() -> i32;
  }
}

fn Class(F:! type).Inner.G() -> i32 { return F(); }

// CHECK:STDOUT: --- prefer_unqualified_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.8b3: type = bind_symbolic_name F, 0 [symbolic]
// CHECK:STDOUT:   %F.patt: type = symbolic_binding_pattern F, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%F.8b3) [symbolic]
// CHECK:STDOUT:   %Inner.b32: type = class_type @Inner [concrete]
// CHECK:STDOUT:   %Inner.6e0: type = class_type @Inner, @Inner(%F.8b3) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Inner(%F.8b3) [symbolic]
// CHECK:STDOUT:   %F.3fa: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @Inner(%F.8b3) [symbolic]
// CHECK:STDOUT:   %G: %G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
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
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.3fa, @F(%F.8b3) [symbolic]
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
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %F.patt.loc5_13.1: type = symbolic_binding_pattern F, 0 [symbolic = %F.patt.loc5_13.2 (constants.%F.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %F.loc5_13.1: type = bind_symbolic_name F, 0 [symbolic = %F.loc5_13.2 (constants.%F.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: @Inner.%G.type (%G.type) = fn_decl @G [symbolic = constants.%G] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %F.loc13_10: type = bind_symbolic_name F, 0 [symbolic = @Class.%F.loc5_13.2 (constants.%F.8b3)]
// CHECK:STDOUT:     %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param.loc13: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return.loc13: ref %i32 = return_slot %return.param.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%F.loc5_13.1: type) {
// CHECK:STDOUT:   %F.loc5_13.2: type = bind_symbolic_name F, 0 [symbolic = %F.loc5_13.2 (constants.%F.8b3)]
// CHECK:STDOUT:   %F.patt.loc5_13.2: type = symbolic_binding_pattern F, 0 [symbolic = %F.patt.loc5_13.2 (constants.%F.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = class_decl @Inner [concrete = constants.%Inner.b32] {} {}
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Class.%F.loc5_13.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.loc8_19.1: type = bind_symbolic_name F, 0 [symbolic = %F.loc8_19.1 (constants.%F.8b3)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Inner(%F.loc8_19.1) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F.loc8_19.2: @Inner.%F.type (%F.type) = struct_value () [symbolic = %F.loc8_19.2 (constants.%F.3fa)]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @Inner(%F.loc8_19.1) [symbolic = %G.type (constants.%G.type)]
// CHECK:STDOUT:   %G: @Inner.%G.type (%G.type) = struct_value () [symbolic = %G (constants.%G)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @Inner.%F.type (%F.type) = fn_decl @F [symbolic = @Inner.%F.loc8_19.2 (constants.%F.3fa)] {
// CHECK:STDOUT:       %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:       %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %G.decl: @Inner.%G.type (%G.type) = fn_decl @G [symbolic = @Inner.%G (constants.%G)] {
// CHECK:STDOUT:       %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %int_32.loc9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %return.param.loc9: ref %i32 = out_param call_param0
// CHECK:STDOUT:       %return.loc9: ref %i32 = return_slot %return.param.loc9
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.6e0
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Class.%F.loc5_13.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc8_29.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc8_29.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %bound_method.loc8_29.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc8_29.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc8_29.2: %i32 = converted %int_0, %.loc8_29.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     return %.loc8_29.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@Class.%F.loc5_13.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.loc13_46.1: type = bind_symbolic_name F, 0 [symbolic = %F.loc13_46.1 (constants.%F.8b3)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Inner(%F.loc13_46.1) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F.loc13_46.2: @G.%F.type (%F.type) = struct_value () [symbolic = %F.loc13_46.2 (constants.%F.3fa)]
// CHECK:STDOUT:   %F.specific_fn.loc13_46.2: <specific function> = specific_function %F.loc13_46.2, @F(%F.loc13_46.1) [symbolic = %F.specific_fn.loc13_46.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc13_46: @G.%F.type (%F.type) = specific_constant @Inner.%F.decl, @Inner(constants.%F.8b3) [symbolic = %F.loc13_46.2 (constants.%F.3fa)]
// CHECK:STDOUT:     %F.ref: @G.%F.type (%F.type) = name_ref F, %.loc13_46 [symbolic = %F.loc13_46.2 (constants.%F.3fa)]
// CHECK:STDOUT:     %F.specific_fn.loc13_46.1: <specific function> = specific_function %F.ref, @F(constants.%F.8b3) [symbolic = %F.specific_fn.loc13_46.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %F.call: init %i32 = call %F.specific_fn.loc13_46.1()
// CHECK:STDOUT:     %.loc13_49.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc13_49.2: %i32 = converted %F.call, %.loc13_49.1
// CHECK:STDOUT:     return %.loc13_49.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%F.8b3) {
// CHECK:STDOUT:   %F.loc5_13.2 => constants.%F.8b3
// CHECK:STDOUT:   %F.patt.loc5_13.2 => constants.%F.8b3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%F.8b3) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.loc8_19.1 => constants.%F.8b3
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F.loc8_19.2 => constants.%F.3fa
// CHECK:STDOUT:   %G.type => constants.%G.type
// CHECK:STDOUT:   %G => constants.%G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%F.8b3) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%F.8b3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(%F.loc8_19.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(@G.%F.loc13_46.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(@G.%F.loc13_46.1) {}
// CHECK:STDOUT:
