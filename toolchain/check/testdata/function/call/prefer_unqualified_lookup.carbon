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
// CHECK:STDOUT:   %F.1: type = bind_symbolic_name F, 0 [symbolic]
// CHECK:STDOUT:   %F.patt: type = symbolic_binding_pattern F, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%F.1) [symbolic]
// CHECK:STDOUT:   %Inner.1: type = class_type @Inner [template]
// CHECK:STDOUT:   %Inner.2: type = class_type @Inner, @Inner(%F.1) [symbolic]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Inner(%F.1) [symbolic]
// CHECK:STDOUT:   %F.2: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @Inner(%F.1) [symbolic]
// CHECK:STDOUT:   %G: %G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.28: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.4, %Convert.14 [template]
// CHECK:STDOUT:   %.30: <specific function> = specific_function %.29, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.31: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %F.2, @F(%F.1) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %F.patt.loc5_13.1: type = symbolic_binding_pattern F, 0 [symbolic = %F.patt.loc5_13.2 (constants.%F.patt)]
// CHECK:STDOUT:     %F.param_patt: type = value_param_pattern %F.patt.loc5_13.1, runtime_param<invalid> [symbolic = %F.patt.loc5_13.2 (constants.%F.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %F.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %F.loc5_13.1: type = bind_symbolic_name F, 0, %F.param [symbolic = %F.loc5_13.2 (constants.%F.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [symbolic = constants.%G] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %F.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %F.loc13_10: type = bind_symbolic_name F, 0, %F.param [symbolic = constants.%F.1]
// CHECK:STDOUT:     %.loc13_33.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc13: init type = call constants.%Int(%.loc13_33.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_33.2: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_33.3: type = converted %int.make_type_signed.loc13, %.loc13_33.2 [template = constants.%i32]
// CHECK:STDOUT:     %return.param.loc13: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%F.loc5_13.1: type) {
// CHECK:STDOUT:   %F.loc5_13.2: type = bind_symbolic_name F, 0 [symbolic = %F.loc5_13.2 (constants.%F.1)]
// CHECK:STDOUT:   %F.patt.loc5_13.2: type = symbolic_binding_pattern F, 0 [symbolic = %F.patt.loc5_13.2 (constants.%F.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = class_decl @Inner [template = constants.%Inner.1] {} {}
// CHECK:STDOUT:     %.loc11: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:     complete_type_witness = %.loc11
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Class.%F.loc5_13.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.loc8_19.1: type = bind_symbolic_name F, 0 [symbolic = %F.loc8_19.1 (constants.%F.1)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Inner(%F.loc8_19.1) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F.loc8_19.2: @Inner.%F.type (%F.type) = struct_value () [symbolic = %F.loc8_19.2 (constants.%F.2)]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @Inner(%F.loc8_19.1) [symbolic = %G.type (constants.%G.type)]
// CHECK:STDOUT:   %G: @Inner.%G.type (%G.type) = struct_value () [symbolic = %G (constants.%G)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @Inner.%F.type (%F.type) = fn_decl @F [symbolic = @Inner.%F.loc8_19.2 (constants.%F.2)] {
// CHECK:STDOUT:       %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc8_15.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:       %int.make_type_signed: init type = call constants.%Int(%.loc8_15.1) [template = constants.%i32]
// CHECK:STDOUT:       %.loc8_15.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:       %.loc8_15.3: type = converted %int.make_type_signed, %.loc8_15.2 [template = constants.%i32]
// CHECK:STDOUT:       %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:       %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %G.decl: @Inner.%G.type (%G.type) = fn_decl @G [symbolic = @Inner.%G (constants.%G)] {
// CHECK:STDOUT:       %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc9_15.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:       %int.make_type_signed.loc9: init type = call constants.%Int(%.loc9_15.1) [template = constants.%i32]
// CHECK:STDOUT:       %.loc9_15.2: type = value_of_initializer %int.make_type_signed.loc9 [template = constants.%i32]
// CHECK:STDOUT:       %.loc9_15.3: type = converted %int.make_type_signed.loc9, %.loc9_15.2 [template = constants.%i32]
// CHECK:STDOUT:       %return.param.loc9: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:       %.loc9_12: ref %i32 = return_slot %return.param.loc9
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc10: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.2
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     complete_type_witness = %.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Class.%F.loc5_13.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc8_28: Core.IntLiteral = int_value 0 [template = constants.%.4]
// CHECK:STDOUT:     %.loc8_29.1: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:     %.loc8_29.2: <bound method> = bound_method %.loc8_28, %.loc8_29.1 [template = constants.%.29]
// CHECK:STDOUT:     %.loc8_29.3: <specific function> = specific_function %.loc8_29.2, @Convert.2(constants.%.1) [template = constants.%.30]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %.loc8_29.3(%.loc8_28) [template = constants.%.31]
// CHECK:STDOUT:     %.loc8_29.4: %i32 = value_of_initializer %int.convert_checked [template = constants.%.31]
// CHECK:STDOUT:     %.loc8_29.5: %i32 = converted %.loc8_28, %.loc8_29.4 [template = constants.%.31]
// CHECK:STDOUT:     return %.loc8_29.5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@Class.%F.loc5_13.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.loc13_46.1: type = bind_symbolic_name F, 0 [symbolic = %F.loc13_46.1 (constants.%F.1)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Inner(%F.loc13_46.1) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F.loc13_46.2: @G.%F.type (%F.type) = struct_value () [symbolic = %F.loc13_46.2 (constants.%F.2)]
// CHECK:STDOUT:   %.loc13_46.3: <specific function> = specific_function %F.loc13_46.2, @F(%F.loc13_46.1) [symbolic = %.loc13_46.3 (constants.%.32)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc13_46.1: @G.%F.type (%F.type) = specific_constant @Inner.%F.decl, @Inner(constants.%F.1) [symbolic = %F.loc13_46.2 (constants.%F.2)]
// CHECK:STDOUT:     %F.ref: @G.%F.type (%F.type) = name_ref F, %.loc13_46.1 [symbolic = %F.loc13_46.2 (constants.%F.2)]
// CHECK:STDOUT:     %.loc13_46.2: <specific function> = specific_function %F.ref, @F(constants.%F.1) [symbolic = %.loc13_46.3 (constants.%.32)]
// CHECK:STDOUT:     %F.call: init %i32 = call %.loc13_46.2()
// CHECK:STDOUT:     %.loc13_49.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc13_49.2: %i32 = converted %F.call, %.loc13_49.1
// CHECK:STDOUT:     return %.loc13_49.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%F.1) {
// CHECK:STDOUT:   %F.loc5_13.2 => constants.%F.1
// CHECK:STDOUT:   %F.patt.loc5_13.2 => constants.%F.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%F.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.loc8_19.1 => constants.%F.1
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F.loc8_19.2 => constants.%F.2
// CHECK:STDOUT:   %G.type => constants.%G.type
// CHECK:STDOUT:   %G => constants.%G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%F.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%F.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(%F.loc8_19.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(@G.%F.loc13_46.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(@G.%F.loc13_46.1) {}
// CHECK:STDOUT:
