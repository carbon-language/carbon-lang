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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.6b6 = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %i32 = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.adf: type = class_type @C, @C(%a) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_1000.ff9: Core.IntLiteral = int_value 1000 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %int_1000.55f: %i32 = int_value 1000 [concrete]
// CHECK:STDOUT:   %C.847: type = class_type @C, @C(%int_1000.55f) [concrete]
// CHECK:STDOUT:   %pattern_type.053: type = pattern_type %C.847 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.053 = symbolic_binding_pattern b, 0 [symbolic]
// CHECK:STDOUT:   %b: %C.847 = symbolic_binding b, 0 [symbolic]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %a.patt.loc4_10.1: %pattern_type.6b6 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc4_10.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc4_10.2: %i32 = symbolic_binding a, 0 [symbolic = %a.loc4_10.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc5: %D.type = class_decl @D [concrete = constants.%D.generic] {
// CHECK:STDOUT:     %b.patt.loc6: %pattern_type.053 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc5 (constants.%b.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5: type = splice_block %C.loc5 [concrete = constants.%C.847] {
// CHECK:STDOUT:       %.Self.loc5: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref.loc5: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %int_1000.loc5: Core.IntLiteral = int_value 1000 [concrete = constants.%int_1000.ff9]
// CHECK:STDOUT:       %C.loc5: type = class_type @C, @C(constants.%int_1000.55f) [concrete = constants.%C.847]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc5_10.2: %C.847 = symbolic_binding b, 0 [symbolic = %b.loc5_10.1 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc6: %D.type = class_decl @D [concrete = constants.%D.generic] {
// CHECK:STDOUT:     %b.patt.loc6: %pattern_type.053 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc5 (constants.%b.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %C.loc6 [concrete = constants.%C.847] {
// CHECK:STDOUT:       %.Self.loc6: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref.loc6: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %int_1000.loc6: Core.IntLiteral = int_value 1000 [concrete = constants.%int_1000.ff9]
// CHECK:STDOUT:       %C.loc6: type = class_type @C, @C(constants.%int_1000.55f) [concrete = constants.%C.847]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc6: %C.847 = symbolic_binding b, 0 [symbolic = %b.loc5_10.1 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%a.loc4_10.2: %i32) {
// CHECK:STDOUT:   %a.patt.loc4_10.2: %pattern_type.6b6 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc4_10.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc4_10.1: %i32 = symbolic_binding a, 0 [symbolic = %a.loc4_10.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.adf
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D(%b.loc5_10.2: %C.847) {
// CHECK:STDOUT:   %b.patt.loc5: %pattern_type.053 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc5 (constants.%b.patt)]
// CHECK:STDOUT:   %b.loc5_10.1: %C.847 = symbolic_binding b, 0 [symbolic = %b.loc5_10.1 (constants.%b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%D
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc4_10.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc4_10.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1000.55f) {
// CHECK:STDOUT:   %a.patt.loc4_10.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc4_10.1 => constants.%int_1000.55f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%b) {
// CHECK:STDOUT:   %b.patt.loc5 => constants.%b.patt
// CHECK:STDOUT:   %b.loc5_10.1 => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_int_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.6b6 = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %i32 = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.adf: type = class_type @C, @C(%a) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_1000.ff9: Core.IntLiteral = int_value 1000 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %int_1000.55f: %i32 = int_value 1000 [concrete]
// CHECK:STDOUT:   %C.847: type = class_type @C, @C(%int_1000.55f) [concrete]
// CHECK:STDOUT:   %pattern_type.053: type = pattern_type %C.847 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.053 = symbolic_binding_pattern b, 0 [symbolic]
// CHECK:STDOUT:   %b: %C.847 = symbolic_binding b, 0 [symbolic]
// CHECK:STDOUT:   %D.type.ff75f2.1: type = generic_class_type @D.loc5 [concrete]
// CHECK:STDOUT:   %D.generic.7223e9.1: %D.type.ff75f2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %D.type.ff75f2.2: type = generic_class_type @D.loc13 [concrete]
// CHECK:STDOUT:   %D.generic.7223e9.2: %D.type.ff75f2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %D.5b0064.2: type = class_type @D.loc13, @D.loc13(%b) [symbolic]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %a.patt.loc4_10.1: %pattern_type.6b6 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc4_10.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc4_10.2: %i32 = symbolic_binding a, 0 [symbolic = %a.loc4_10.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc5: %D.type.ff75f2.1 = class_decl @D.loc5 [concrete = constants.%D.generic.7223e9.1] {
// CHECK:STDOUT:     %b.patt.loc5_10.1: %pattern_type.053 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc5_10.2 (constants.%b.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5: type = splice_block %C [concrete = constants.%C.847] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %int_1000: Core.IntLiteral = int_value 1000 [concrete = constants.%int_1000.ff9]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%int_1000.55f) [concrete = constants.%C.847]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc5_10.2: %C.847 = symbolic_binding b, 0 [symbolic = %b.loc5_10.1 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc13: %D.type.ff75f2.2 = class_decl @D.loc13 [concrete = constants.%D.generic.7223e9.2] {
// CHECK:STDOUT:     %b.patt.loc13_10.1: %pattern_type.053 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc13_10.2 (constants.%b.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13: type = splice_block %C [concrete = constants.%C.847] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %int_1000: Core.IntLiteral = int_value 1000 [concrete = constants.%int_1000.ff9]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%int_1000.55f) [concrete = constants.%C.847]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc13_10.2: %C.847 = symbolic_binding b, 0 [symbolic = %b.loc13_10.1 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%a.loc4_10.2: %i32) {
// CHECK:STDOUT:   %a.patt.loc4_10.2: %pattern_type.6b6 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc4_10.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc4_10.1: %i32 = symbolic_binding a, 0 [symbolic = %a.loc4_10.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.adf
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D.loc5(%b.loc5_10.2: %C.847) {
// CHECK:STDOUT:   %b.patt.loc5_10.2: %pattern_type.053 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc5_10.2 (constants.%b.patt)]
// CHECK:STDOUT:   %b.loc5_10.1: %C.847 = symbolic_binding b, 0 [symbolic = %b.loc5_10.1 (constants.%b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D.loc13(%b.loc13_10.2: %C.847) {
// CHECK:STDOUT:   %b.patt.loc13_10.2: %pattern_type.053 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc13_10.2 (constants.%b.patt)]
// CHECK:STDOUT:   %b.loc13_10.1: %C.847 = symbolic_binding b, 0 [symbolic = %b.loc13_10.1 (constants.%b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%D.5b0064.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc4_10.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc4_10.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1000.55f) {
// CHECK:STDOUT:   %a.patt.loc4_10.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc4_10.1 => constants.%int_1000.55f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.loc5(constants.%b) {
// CHECK:STDOUT:   %b.patt.loc5_10.2 => constants.%b.patt
// CHECK:STDOUT:   %b.loc5_10.1 => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.loc13(constants.%b) {
// CHECK:STDOUT:   %b.patt.loc13_10.2 => constants.%b.patt
// CHECK:STDOUT:   %b.loc13_10.1 => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
