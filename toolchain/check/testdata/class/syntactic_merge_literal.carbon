// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/syntactic_merge_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/syntactic_merge_literal.carbon

// --- int_match.carbon

library "[[@TEST_NAME]]";

class C(a:! i32) {}
class D(b:! C(1_000));
class D(b:! C(1_000)) {}

// --- fail_int_mismatch.carbon

library "[[@TEST_NAME]]";

class C(a:! i32) {}
class D(b:! C(1000));
// CHECK:STDERR: fail_int_mismatch.carbon:[[@LINE+7]]:15: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: class D(b:! C(1_000)) {}
// CHECK:STDERR:               ^~~~~
// CHECK:STDERR: fail_int_mismatch.carbon:[[@LINE-4]]:15: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: class D(b:! C(1000));
// CHECK:STDERR:               ^~~~
// CHECK:STDERR:
class D(b:! C(1_000)) {}

// CHECK:STDOUT: --- int_match.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %a: %i32 = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %i32 = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.506: type = class_type @C, @C(%a) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_1000.ff9: Core.IntLiteral = int_value 1000 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1000.ff9, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1000.ff9, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1000.1b6: %i32 = int_value 1000 [concrete]
// CHECK:STDOUT:   %C.262: type = class_type @C, @C(%int_1000.1b6) [concrete]
// CHECK:STDOUT:   %b: %C.262 = bind_symbolic_name b, 0 [symbolic]
// CHECK:STDOUT:   %b.patt: %C.262 = symbolic_binding_pattern b, 0 [symbolic]
// CHECK:STDOUT:   %D.type: type = generic_class_type @D [concrete]
// CHECK:STDOUT:   %D.generic: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %D: type = class_type @D, @D(%b) [symbolic]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %a.patt.loc4_9.1: %i32 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc4_9.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc4_9.1: %i32 = bind_symbolic_name a, 0 [symbolic = %a.loc4_9.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc5: %D.type = class_decl @D [concrete = constants.%D.generic] {
// CHECK:STDOUT:     %b.patt.loc6: %C.262 = symbolic_binding_pattern b, 0 [symbolic = constants.%b.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_20.1: type = splice_block %C.loc5 [concrete = constants.%C.262] {
// CHECK:STDOUT:       %C.ref.loc5: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %int_1000.loc5: Core.IntLiteral = int_value 1000 [concrete = constants.%int_1000.ff9]
// CHECK:STDOUT:       %impl.elem0.loc5: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:       %bound_method.loc5_20.1: <bound method> = bound_method %int_1000.loc5, %impl.elem0.loc5 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:       %specific_fn.loc5: <specific function> = specific_function %impl.elem0.loc5, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc5_20.2: <bound method> = bound_method %int_1000.loc5, %specific_fn.loc5 [concrete = constants.%bound_method]
// CHECK:STDOUT:       %int.convert_checked.loc5: init %i32 = call %bound_method.loc5_20.2(%int_1000.loc5) [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %.loc5_20.2: %i32 = value_of_initializer %int.convert_checked.loc5 [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %.loc5_20.3: %i32 = converted %int_1000.loc5, %.loc5_20.2 [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %C.loc5: type = class_type @C, @C(constants.%int_1000.1b6) [concrete = constants.%C.262]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc5_9.1: %C.262 = bind_symbolic_name b, 0 [symbolic = %b.loc5_9.2 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc6: %D.type = class_decl @D [concrete = constants.%D.generic] {
// CHECK:STDOUT:     %b.patt.loc6: %C.262 = symbolic_binding_pattern b, 0 [symbolic = constants.%b.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_20.1: type = splice_block %C.loc6 [concrete = constants.%C.262] {
// CHECK:STDOUT:       %C.ref.loc6: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %int_1000.loc6: Core.IntLiteral = int_value 1000 [concrete = constants.%int_1000.ff9]
// CHECK:STDOUT:       %impl.elem0.loc6: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:       %bound_method.loc6_20.1: <bound method> = bound_method %int_1000.loc6, %impl.elem0.loc6 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:       %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc6_20.2: <bound method> = bound_method %int_1000.loc6, %specific_fn.loc6 [concrete = constants.%bound_method]
// CHECK:STDOUT:       %int.convert_checked.loc6: init %i32 = call %bound_method.loc6_20.2(%int_1000.loc6) [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %.loc6_20.2: %i32 = value_of_initializer %int.convert_checked.loc6 [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %.loc6_20.3: %i32 = converted %int_1000.loc6, %.loc6_20.2 [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %C.loc6: type = class_type @C, @C(constants.%int_1000.1b6) [concrete = constants.%C.262]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc6: %C.262 = bind_symbolic_name b, 0 [symbolic = constants.%b]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%a.loc4_9.1: %i32) {
// CHECK:STDOUT:   %a.loc4_9.2: %i32 = bind_symbolic_name a, 0 [symbolic = %a.loc4_9.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc4_9.2: %i32 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc4_9.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.506
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D(%b.loc5_9.1: %C.262) {
// CHECK:STDOUT:   %b.loc5_9.2: %C.262 = bind_symbolic_name b, 0 [symbolic = %b.loc5_9.2 (constants.%b)]
// CHECK:STDOUT:   %b.patt.loc5: %C.262 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc5 (constants.%b.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%D
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%a) {
// CHECK:STDOUT:   %a.loc4_9.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc4_9.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1000.1b6) {
// CHECK:STDOUT:   %a.loc4_9.2 => constants.%int_1000.1b6
// CHECK:STDOUT:   %a.patt.loc4_9.2 => constants.%int_1000.1b6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%b) {
// CHECK:STDOUT:   %b.loc5_9.2 => constants.%b
// CHECK:STDOUT:   %b.patt.loc5 => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_int_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %a: %i32 = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %i32 = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.506: type = class_type @C, @C(%a) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_1000.ff9: Core.IntLiteral = int_value 1000 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1000.ff9, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1000.ff9, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1000.1b6: %i32 = int_value 1000 [concrete]
// CHECK:STDOUT:   %C.262: type = class_type @C, @C(%int_1000.1b6) [concrete]
// CHECK:STDOUT:   %b: %C.262 = bind_symbolic_name b, 0 [symbolic]
// CHECK:STDOUT:   %b.patt: %C.262 = symbolic_binding_pattern b, 0 [symbolic]
// CHECK:STDOUT:   %D.type: type = generic_class_type @D [concrete]
// CHECK:STDOUT:   %D.generic: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %.type: type = generic_class_type @.1 [concrete]
// CHECK:STDOUT:   %.generic: %.type = struct_value () [concrete]
// CHECK:STDOUT:   %.3d2: type = class_type @.1, @.1(%b) [symbolic]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %a.patt.loc4_9.1: %i32 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc4_9.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc4_9.1: %i32 = bind_symbolic_name a, 0 [symbolic = %a.loc4_9.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = class_decl @D [concrete = constants.%D.generic] {
// CHECK:STDOUT:     %b.patt.loc5_9.1: %C.262 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc5_9.2 (constants.%b.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_19.1: type = splice_block %C [concrete = constants.%C.262] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %int_1000: Core.IntLiteral = int_value 1000 [concrete = constants.%int_1000.ff9]
// CHECK:STDOUT:       %impl.elem0: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:       %bound_method.loc5_19.1: <bound method> = bound_method %int_1000, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc5_19.2: <bound method> = bound_method %int_1000, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:       %int.convert_checked: init %i32 = call %bound_method.loc5_19.2(%int_1000) [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %.loc5_19.2: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %.loc5_19.3: %i32 = converted %int_1000, %.loc5_19.2 [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%int_1000.1b6) [concrete = constants.%C.262]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc5_9.1: %C.262 = bind_symbolic_name b, 0 [symbolic = %b.loc5_9.2 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = class_decl @.1 [concrete = constants.%.generic] {
// CHECK:STDOUT:     %b.patt.loc13_9.1: %C.262 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc13_9.2 (constants.%b.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_20.1: type = splice_block %C [concrete = constants.%C.262] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %int_1000: Core.IntLiteral = int_value 1000 [concrete = constants.%int_1000.ff9]
// CHECK:STDOUT:       %impl.elem0: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:       %bound_method.loc13_20.1: <bound method> = bound_method %int_1000, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc13_20.2: <bound method> = bound_method %int_1000, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:       %int.convert_checked: init %i32 = call %bound_method.loc13_20.2(%int_1000) [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %.loc13_20.2: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %.loc13_20.3: %i32 = converted %int_1000, %.loc13_20.2 [concrete = constants.%int_1000.1b6]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%int_1000.1b6) [concrete = constants.%C.262]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc13_9.1: %C.262 = bind_symbolic_name b, 0 [symbolic = %b.loc13_9.2 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%a.loc4_9.1: %i32) {
// CHECK:STDOUT:   %a.loc4_9.2: %i32 = bind_symbolic_name a, 0 [symbolic = %a.loc4_9.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc4_9.2: %i32 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc4_9.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.506
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D(%b.loc5_9.1: %C.262) {
// CHECK:STDOUT:   %b.loc5_9.2: %C.262 = bind_symbolic_name b, 0 [symbolic = %b.loc5_9.2 (constants.%b)]
// CHECK:STDOUT:   %b.patt.loc5_9.2: %C.262 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc5_9.2 (constants.%b.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @.1(%b.loc13_9.1: %C.262) {
// CHECK:STDOUT:   %b.loc13_9.2: %C.262 = bind_symbolic_name b, 0 [symbolic = %b.loc13_9.2 (constants.%b)]
// CHECK:STDOUT:   %b.patt.loc13_9.2: %C.262 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc13_9.2 (constants.%b.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%.3d2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%a) {
// CHECK:STDOUT:   %a.loc4_9.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc4_9.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1000.1b6) {
// CHECK:STDOUT:   %a.loc4_9.2 => constants.%int_1000.1b6
// CHECK:STDOUT:   %a.patt.loc4_9.2 => constants.%int_1000.1b6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%b) {
// CHECK:STDOUT:   %b.loc5_9.2 => constants.%b
// CHECK:STDOUT:   %b.patt.loc5_9.2 => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%b) {
// CHECK:STDOUT:   %b.loc13_9.2 => constants.%b
// CHECK:STDOUT:   %b.patt.loc13_9.2 => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
