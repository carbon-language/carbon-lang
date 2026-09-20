// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/redeclare.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/redeclare.carbon

// --- valid.carbon

library "[[@TEST_NAME]]";

class Generic(T: type);

class Generic(T: type) {
}

// --- fail_mismatch_param_list.carbon

library "[[@TEST_NAME]]";

class A;
// CHECK:STDERR: fail_mismatch_param_list.carbon:[[@LINE+3]]:1-18: error: redeclaration differs because of parameter list [RedeclParamListDiffers]
// CHECK:STDERR: fail_mismatch_param_list.carbon:[[@LINE-2]]:1-8: note: previously declared without parameter list [RedeclParamListPrevious]
// CHECK:STDERR:
class A(T: type) {}

// --- fail_mismatch_implicit_param_list.carbon

library "[[@TEST_NAME]]";

class A {}

class B(N: A);
// CHECK:STDERR: fail_mismatch_implicit_param_list.carbon:[[@LINE+3]]:1-24: error: redeclaration differs because of implicit parameter list [RedeclParamListDiffers]
// CHECK:STDERR: fail_mismatch_implicit_param_list.carbon:[[@LINE-2]]:1-14: note: previously declared without implicit parameter list [RedeclParamListPrevious]
// CHECK:STDERR:
class B[T: type](N: T) {}

// --- fail_mismatch_param_count.carbon

library "[[@TEST_NAME]]";

class A {}

class C(T: type);
// CHECK:STDERR: fail_mismatch_param_count.carbon:[[@LINE+3]]:1-24: error: redeclaration differs because of parameter count of 2 [RedeclParamCountDiffers]
// CHECK:STDERR: fail_mismatch_param_count.carbon:[[@LINE-2]]:1-17: note: previously declared with parameter count of 1 [RedeclParamCountPrevious]
// CHECK:STDERR:
class C(T: type, U: A) {}

// --- fail_mismatch_param_type.carbon

library "[[@TEST_NAME]]";

class A {}

class D(T: type);
// CHECK:STDERR: fail_mismatch_param_type.carbon:[[@LINE+3]]:9-12: error: type `<pattern for A>` of parameter 1 in redeclaration differs from previous parameter type `<pattern for type>` [RedeclParamDiffersType]
// CHECK:STDERR: fail_mismatch_param_type.carbon:[[@LINE-2]]:9-15: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR:
class D(T: A) {}

// --- fail_mismatch_param_name.carbon

library "[[@TEST_NAME]]";

class E(T: type);
// CHECK:STDERR: fail_mismatch_param_name.carbon:[[@LINE+3]]:9-15: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
// CHECK:STDERR: fail_mismatch_param_name.carbon:[[@LINE-2]]:9-15: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR:
class E(U: type) {}

// CHECK:STDOUT: --- valid.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type: type = generic_class_type @Generic [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic: type = class_type @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
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
// CHECK:STDOUT:     .Generic = %Generic.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Generic.decl.loc4: %Generic.type = class_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %T.patt.loc6: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_18.1: type = splice_block %.loc4_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc4: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_16.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.decl.loc6: %Generic.type = class_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %T.patt.loc6: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_18.1: type = splice_block %.loc6_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc6: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc6_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6: type = symbolic_binding T, 0 [symbolic = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Generic(%T.loc4_16.2: type) {
// CHECK:STDOUT:   %T.patt.loc4: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_16.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Generic
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_list.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.5a2: type = class_type @A.loc4 [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A.loc8 [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.47a: type = class_type @A.loc8, @A.loc8(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
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
// CHECK:STDOUT:     .A = %A.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl.loc4: type = class_decl @A.loc4 [concrete = constants.%A.5a2] {} {}
// CHECK:STDOUT:   %A.decl.loc8: %A.type = class_decl @A.loc8 [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt.loc8_10.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_12.1: type = splice_block %.loc8_12.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc8_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A.loc4;
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A.loc8(%T.loc8_10.2: type) {
// CHECK:STDOUT:   %T.patt.loc8_10.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.47a
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.loc8(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc8_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_10.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_implicit_param_list.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %N.patt.3a1: %pattern_type.9ef = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N.37f: %A = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %B.type.7937a9.1: type = generic_class_type @B.loc6 [concrete]
// CHECK:STDOUT:   %B.generic.60ca48.1: %B.type.7937a9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %N.patt.9f0: %pattern_type.51d = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %N.bd9: %T = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %B.type.7937a9.2: type = generic_class_type @B.loc10 [concrete]
// CHECK:STDOUT:   %B.generic.60ca48.2: %B.type.7937a9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %B.2b1: type = class_type @B.loc10, @B.loc10(%T, %N.bd9) [symbolic]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl.loc6: %B.type.7937a9.1 = class_decl @B.loc6 [concrete = constants.%B.generic.60ca48.1] {
// CHECK:STDOUT:     %N.patt.loc6_10.1: %pattern_type.9ef = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_10.2 (constants.%N.patt.3a1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %A.ref [concrete = constants.%A] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_10.2: %A = symbolic_binding N, 0 [symbolic = %N.loc6_10.1 (constants.%N.37f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc10: %B.type.7937a9.2 = class_decl @B.loc10 [concrete = constants.%B.generic.60ca48.2] {
// CHECK:STDOUT:     %T.patt.loc10_10.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_10.2 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc10_19.1: @B.loc10.%pattern_type (%pattern_type.51d) = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc10_19.2 (constants.%N.patt.9f0)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_12.1: type = splice_block %.loc10_12.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc10_10: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc10_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc10_10.1 (constants.%T)]
// CHECK:STDOUT:     %.loc10_21: type = splice_block %T.ref [symbolic = %T.loc10_10.1 (constants.%T)] {
// CHECK:STDOUT:       %.Self.frozen.loc10_19: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc10_10.2 [symbolic = %T.loc10_10.1 (constants.%T)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc10_19.2: @B.loc10.%T.loc10_10.1 (%T) = symbolic_binding N, 1 [symbolic = %N.loc10_19.1 (constants.%N.bd9)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @B.loc6(%N.loc6_10.2: %A) {
// CHECK:STDOUT:   %N.patt.loc6_10.2: %pattern_type.9ef = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_10.2 (constants.%N.patt.3a1)]
// CHECK:STDOUT:   %N.loc6_10.1: %A = symbolic_binding N, 0 [symbolic = %N.loc6_10.1 (constants.%N.37f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @B.loc10(%T.loc10_10.2: type, %N.loc10_19.2: @B.loc10.%T.loc10_10.1 (%T)) {
// CHECK:STDOUT:   %T.patt.loc10_10.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc10_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc10_10.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc10_10.1 [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %N.patt.loc10_19.2: @B.loc10.%pattern_type (%pattern_type.51d) = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc10_19.2 (constants.%N.patt.9f0)]
// CHECK:STDOUT:   %N.loc10_19.1: @B.loc10.%T.loc10_10.1 (%T) = symbolic_binding N, 1 [symbolic = %N.loc10_19.1 (constants.%N.bd9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%B.2b1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.loc6(constants.%N.37f) {
// CHECK:STDOUT:   %N.patt.loc6_10.2 => constants.%N.patt.3a1
// CHECK:STDOUT:   %N.loc6_10.1 => constants.%N.37f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.loc10(constants.%T, constants.%N.bd9) {
// CHECK:STDOUT:   %T.patt.loc10_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc10_10.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %N.patt.loc10_19.2 => constants.%N.patt.9f0
// CHECK:STDOUT:   %N.loc10_19.1 => constants.%N.bd9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_count.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type.a601fe.1: type = generic_class_type @C.loc6 [concrete]
// CHECK:STDOUT:   %C.generic.4b4597.1: %C.type.a601fe.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.9ef = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U: %A = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %C.type.a601fe.2: type = generic_class_type @C.loc10 [concrete]
// CHECK:STDOUT:   %C.generic.4b4597.2: %C.type.a601fe.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.394: type = class_type @C.loc10, @C.loc10(%T, %U) [symbolic]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .C = %C.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %C.decl.loc6: %C.type.a601fe.1 = class_decl @C.loc6 [concrete = constants.%C.generic.4b4597.1] {
// CHECK:STDOUT:     %T.patt.loc6_10.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_12.1: type = splice_block %.loc6_12.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc6_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc10: %C.type.a601fe.2 = class_decl @C.loc10 [concrete = constants.%C.generic.4b4597.2] {
// CHECK:STDOUT:     %T.patt.loc10_10.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_10.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc10_19.1: %pattern_type.9ef = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc10_19.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_12.1: type = splice_block %.loc10_12.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc10_10: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc10_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc10_10.1 (constants.%T)]
// CHECK:STDOUT:     %.loc10_21: type = splice_block %A.ref [concrete = constants.%A] {
// CHECK:STDOUT:       %.Self.frozen.loc10_19: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc10_19.2: %A = symbolic_binding U, 1 [symbolic = %U.loc10_19.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C.loc6(%T.loc6_10.2: type) {
// CHECK:STDOUT:   %T.patt.loc6_10.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C.loc10(%T.loc10_10.2: type, %U.loc10_19.2: %A) {
// CHECK:STDOUT:   %T.patt.loc10_10.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc10_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc10_10.1 (constants.%T)]
// CHECK:STDOUT:   %U.patt.loc10_19.2: %pattern_type.9ef = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc10_19.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc10_19.1: %A = symbolic_binding U, 1 [symbolic = %U.loc10_19.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.394
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.loc6(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_10.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.loc10(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.patt.loc10_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc10_10.1 => constants.%T
// CHECK:STDOUT:   %U.patt.loc10_19.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc10_19.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.d47: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %D.type.ff75f2.1: type = generic_class_type @D.loc6 [concrete]
// CHECK:STDOUT:   %D.generic.7223e9.1: %D.type.ff75f2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %T.patt.3a1: %pattern_type.9ef = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.37f: %A = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %D.type.ff75f2.2: type = generic_class_type @D.loc10 [concrete]
// CHECK:STDOUT:   %D.generic.7223e9.2: %D.type.ff75f2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %D.ad0: type = class_type @D.loc10, @D.loc10(%T.37f) [symbolic]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .D = %D.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %D.decl.loc6: %D.type.ff75f2.1 = class_decl @D.loc6 [concrete = constants.%D.generic.7223e9.1] {
// CHECK:STDOUT:     %T.patt.loc6_10.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_10.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_12.1: type = splice_block %.loc6_12.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc6_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_10.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc10: %D.type.ff75f2.2 = class_decl @D.loc10 [concrete = constants.%D.generic.7223e9.2] {
// CHECK:STDOUT:     %T.patt.loc10_10.1: %pattern_type.9ef = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_10.2 (constants.%T.patt.3a1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10: type = splice_block %A.ref [concrete = constants.%A] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_10.2: %A = symbolic_binding T, 0 [symbolic = %T.loc10_10.1 (constants.%T.37f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D.loc6(%T.loc6_10.2: type) {
// CHECK:STDOUT:   %T.patt.loc6_10.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_10.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   %T.loc6_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_10.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D.loc10(%T.loc10_10.2: %A) {
// CHECK:STDOUT:   %T.patt.loc10_10.2: %pattern_type.9ef = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_10.2 (constants.%T.patt.3a1)]
// CHECK:STDOUT:   %T.loc10_10.1: %A = symbolic_binding T, 0 [symbolic = %T.loc10_10.1 (constants.%T.37f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%D.ad0
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.loc6(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc6_10.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc6_10.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.loc10(constants.%T.37f) {
// CHECK:STDOUT:   %T.patt.loc10_10.2 => constants.%T.patt.3a1
// CHECK:STDOUT:   %T.loc10_10.1 => constants.%T.37f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %E.type.0b1ef1.1: type = generic_class_type @E.loc4 [concrete]
// CHECK:STDOUT:   %E.generic.5469ef.1: %E.type.0b1ef1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %E.type.0b1ef1.2: type = generic_class_type @E.loc8 [concrete]
// CHECK:STDOUT:   %E.generic.5469ef.2: %E.type.0b1ef1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %E.7de02e.2: type = class_type @E.loc8, @E.loc8(%U) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
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
// CHECK:STDOUT:     .E = %E.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %E.decl.loc4: %E.type.0b1ef1.1 = class_decl @E.loc4 [concrete = constants.%E.generic.5469ef.1] {
// CHECK:STDOUT:     %T.patt.loc4_10.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_12.1: type = splice_block %.loc4_12.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.decl.loc8: %E.type.0b1ef1.2 = class_decl @E.loc8 [concrete = constants.%E.generic.5469ef.2] {
// CHECK:STDOUT:     %U.patt.loc8_10.1: %pattern_type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_10.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_12.1: type = splice_block %.loc8_12.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc8_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_10.2: type = symbolic_binding U, 0 [symbolic = %U.loc8_10.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @E.loc4(%T.loc4_10.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_10.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @E.loc8(%U.loc8_10.2: type) {
// CHECK:STDOUT:   %U.patt.loc8_10.2: %pattern_type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_10.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc8_10.1: type = symbolic_binding U, 0 [symbolic = %U.loc8_10.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%E.7de02e.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @E.loc4(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_10.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @E.loc8(constants.%U) {
// CHECK:STDOUT:   %U.patt.loc8_10.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc8_10.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
