// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/generic_vs_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/generic_vs_params.carbon

// --- params.carbon

library "[[@TEST_NAME]]";

class NotGenericNoParams {}
class NotGenericButParams() {}
class GenericAndParams(T:! type) {}

class C(T:! type) {
  class GenericNoParams {}
  class GenericAndParams(U:! type) {}
}

class X {}

var a: NotGenericNoParams = {};
var b: NotGenericButParams() = {};
var c: GenericAndParams(X) = {};
var d: C(X).GenericNoParams = {};
var e: C(X).GenericAndParams(X) = {};

// --- fail_non_generic_implicit_params.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_non_generic_implicit_params.carbon:[[@LINE+4]]:9: error: parameters of generic types must be constant [GenericParamMustBeConstant]
// CHECK:STDERR: class A[T: type]() {}
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR:
class A[T: type]() {}

// --- fail_non_generic_params.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_non_generic_params.carbon:[[@LINE+4]]:9: error: parameters of generic types must be constant [GenericParamMustBeConstant]
// CHECK:STDERR: class A(T: type) {}
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR:
class A(T: type) {}

// This is testing a use of the invalid type.
fn F(T:! type) {
  A(T);
}

// --- fail_implicit_params_only_empty.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_implicit_params_only_empty.carbon:[[@LINE+4]]:10: error: expected explicit parameters after implicit parameters [GenericMissingExplicitParameters]
// CHECK:STDERR: class Foo[];
// CHECK:STDERR:          ^~
// CHECK:STDERR:
class Foo[];

// --- fail_implicit_params_only.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_implicit_params_only.carbon:[[@LINE+4]]:10: error: expected explicit parameters after implicit parameters [GenericMissingExplicitParameters]
// CHECK:STDERR: class Foo[T:! type];
// CHECK:STDERR:          ^~~~~~~~~~
// CHECK:STDERR:
class Foo[T:! type];

// CHECK:STDOUT: --- params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NotGenericNoParams: type = class_type @NotGenericNoParams [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %NotGenericButParams.type: type = generic_class_type @NotGenericButParams [concrete]
// CHECK:STDOUT:   %NotGenericButParams.generic: %NotGenericButParams.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotGenericButParams: type = class_type @NotGenericButParams [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.type.fa0: type = generic_class_type @GenericAndParams.loc21 [concrete]
// CHECK:STDOUT:   %GenericAndParams.generic.e6f: %GenericAndParams.type.fa0 = struct_value () [concrete]
// CHECK:STDOUT:   %GenericAndParams.2b4: type = class_type @GenericAndParams.loc21, @GenericAndParams.loc21(%T) [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.5a3: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %GenericNoParams.6cf: type = class_type @GenericNoParams, @GenericNoParams(%T) [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.type.208: type = generic_class_type @GenericAndParams.loc42, @C(%T) [symbolic]
// CHECK:STDOUT:   %GenericAndParams.generic.cc2: %GenericAndParams.type.208 = struct_value () [symbolic]
// CHECK:STDOUT:   %GenericAndParams.015: type = class_type @GenericAndParams.loc42, @GenericAndParams.loc42(%T, %U) [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %pattern_type.673: type = pattern_type %NotGenericNoParams [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %NotGenericNoParams.val: %NotGenericNoParams = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.749: type = pattern_type %NotGenericButParams [concrete]
// CHECK:STDOUT:   %NotGenericButParams.val: %NotGenericButParams = struct_value () [concrete]
// CHECK:STDOUT:   %GenericAndParams.d3d: type = class_type @GenericAndParams.loc21, @GenericAndParams.loc21(%X) [concrete]
// CHECK:STDOUT:   %pattern_type.856: type = pattern_type %GenericAndParams.d3d [concrete]
// CHECK:STDOUT:   %GenericAndParams.val.835: %GenericAndParams.d3d = struct_value () [concrete]
// CHECK:STDOUT:   %C.513: type = class_type @C, @C(%X) [concrete]
// CHECK:STDOUT:   %GenericNoParams.efa: type = class_type @GenericNoParams, @GenericNoParams(%X) [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.23d: type = generic_class_type @GenericAndParams.loc42, @C(%X) [concrete]
// CHECK:STDOUT:   %GenericAndParams.generic.fd2: %GenericAndParams.type.23d = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.1fb: type = pattern_type %GenericNoParams.efa [concrete]
// CHECK:STDOUT:   %GenericNoParams.val: %GenericNoParams.efa = struct_value () [concrete]
// CHECK:STDOUT:   %GenericAndParams.ed1: type = class_type @GenericAndParams.loc42, @GenericAndParams.loc42(%X, %X) [concrete]
// CHECK:STDOUT:   %pattern_type.1b8: type = pattern_type %GenericAndParams.ed1 [concrete]
// CHECK:STDOUT:   %GenericAndParams.val.e35: %GenericAndParams.ed1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NotGenericNoParams = %NotGenericNoParams.decl
// CHECK:STDOUT:     .NotGenericButParams = %NotGenericButParams.decl
// CHECK:STDOUT:     .GenericAndParams = %GenericAndParams.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotGenericNoParams.decl: type = class_decl @NotGenericNoParams [concrete = constants.%NotGenericNoParams] {} {}
// CHECK:STDOUT:   %NotGenericButParams.decl: %NotGenericButParams.type = class_decl @NotGenericButParams [concrete = constants.%NotGenericButParams.generic] {} {}
// CHECK:STDOUT:   %GenericAndParams.decl: %GenericAndParams.type.fa0 = class_decl @GenericAndParams.loc21 [concrete = constants.%GenericAndParams.generic.e6f] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc19.1: type = splice_block %.loc19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc17.2: type = symbolic_binding T, 0 [symbolic = %T.loc17.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc28.1: type = splice_block %.loc28.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc28.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26.2: type = symbolic_binding T, 0 [symbolic = %T.loc26.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.673 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.673 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %NotGenericNoParams = var %a.var_patt [concrete]
// CHECK:STDOUT:   %NotGenericNoParams.ref: type = name_ref NotGenericNoParams, %NotGenericNoParams.decl [concrete = constants.%NotGenericNoParams]
// CHECK:STDOUT:   %a: ref %NotGenericNoParams = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.749 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.749 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %NotGenericButParams = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc62: type = splice_block %NotGenericButParams [concrete = constants.%NotGenericButParams] {
// CHECK:STDOUT:     %NotGenericButParams.ref: %NotGenericButParams.type = name_ref NotGenericButParams, %NotGenericButParams.decl [concrete = constants.%NotGenericButParams.generic]
// CHECK:STDOUT:     %NotGenericButParams: type = class_type @NotGenericButParams [concrete = constants.%NotGenericButParams]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %NotGenericButParams = ref_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.856 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.856 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %GenericAndParams.d3d = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc73: type = splice_block %GenericAndParams.loc73 [concrete = constants.%GenericAndParams.d3d] {
// CHECK:STDOUT:     %GenericAndParams.ref.loc70: %GenericAndParams.type.fa0 = name_ref GenericAndParams, %GenericAndParams.decl [concrete = constants.%GenericAndParams.generic.e6f]
// CHECK:STDOUT:     %X.ref.loc72: type = name_ref X, %X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %GenericAndParams.loc73: type = class_type @GenericAndParams.loc21, @GenericAndParams.loc21(constants.%X) [concrete = constants.%GenericAndParams.d3d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %GenericAndParams.d3d = ref_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.1fb = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.1fb = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %GenericNoParams.efa = var %d.var_patt [concrete]
// CHECK:STDOUT:   %.loc85.1: type = splice_block %GenericNoParams.ref [concrete = constants.%GenericNoParams.efa] {
// CHECK:STDOUT:     %C.ref.loc81: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %X.ref.loc83: type = name_ref X, %X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C.loc84: type = class_type @C, @C(constants.%X) [concrete = constants.%C.513]
// CHECK:STDOUT:     %.loc85.2: type = specific_constant @C.%GenericNoParams.decl, @C(constants.%X) [concrete = constants.%GenericNoParams.efa]
// CHECK:STDOUT:     %GenericNoParams.ref: type = name_ref GenericNoParams, %.loc85.2 [concrete = constants.%GenericNoParams.efa]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %GenericNoParams.efa = ref_binding d, %d.var [concrete = %d.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.1b8 = ref_binding_pattern e [concrete]
// CHECK:STDOUT:     %e.var_patt: %pattern_type.1b8 = var_pattern %e.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %GenericAndParams.ed1 = var %e.var_patt [concrete]
// CHECK:STDOUT:   %.loc102: type = splice_block %GenericAndParams.loc102 [concrete = constants.%GenericAndParams.ed1] {
// CHECK:STDOUT:     %C.ref.loc94: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %X.ref.loc96: type = name_ref X, %X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C.loc97: type = class_type @C, @C(constants.%X) [concrete = constants.%C.513]
// CHECK:STDOUT:     %.loc98: %GenericAndParams.type.23d = specific_constant @C.%GenericAndParams.decl, @C(constants.%X) [concrete = constants.%GenericAndParams.generic.fd2]
// CHECK:STDOUT:     %GenericAndParams.ref.loc98: %GenericAndParams.type.23d = name_ref GenericAndParams, %.loc98 [concrete = constants.%GenericAndParams.generic.fd2]
// CHECK:STDOUT:     %X.ref.loc101: type = name_ref X, %X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %GenericAndParams.loc102: type = class_type @GenericAndParams.loc42, @GenericAndParams.loc42(constants.%X, constants.%X) [concrete = constants.%GenericAndParams.ed1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: ref %GenericAndParams.ed1 = ref_binding e, %e.var [concrete = %e.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NotGenericNoParams {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NotGenericNoParams
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NotGenericButParams {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NotGenericButParams
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @GenericAndParams.loc21(%T.loc17.2: type) {
// CHECK:STDOUT:   %T.loc17.1: type = symbolic_binding T, 0 [symbolic = %T.loc17.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%GenericAndParams.2b4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc26.2: type) {
// CHECK:STDOUT:   %T.loc26.1: type = symbolic_binding T, 0 [symbolic = %T.loc26.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericNoParams: type = class_type @GenericNoParams, @GenericNoParams(%T.loc26.1) [symbolic = %GenericNoParams (constants.%GenericNoParams.6cf)]
// CHECK:STDOUT:   %GenericAndParams.type: type = generic_class_type @GenericAndParams.loc42, @C(%T.loc26.1) [symbolic = %GenericAndParams.type (constants.%GenericAndParams.type.208)]
// CHECK:STDOUT:   %GenericAndParams.generic: @C.%GenericAndParams.type (%GenericAndParams.type.208) = struct_value () [symbolic = %GenericAndParams.generic (constants.%GenericAndParams.generic.cc2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %GenericNoParams.decl: type = class_decl @GenericNoParams [symbolic = @C.%GenericNoParams (constants.%GenericNoParams.6cf)] {} {}
// CHECK:STDOUT:     %GenericAndParams.decl: @C.%GenericAndParams.type (%GenericAndParams.type.208) = class_decl @GenericAndParams.loc42 [symbolic = @C.%GenericAndParams.generic (constants.%GenericAndParams.generic.cc2)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc40.1: type = splice_block %.loc40.2 [concrete = type] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %.loc40.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc38.2: type = symbolic_binding U, 1 [symbolic = %U.loc38.1 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.5a3
// CHECK:STDOUT:     .GenericNoParams = %GenericNoParams.decl
// CHECK:STDOUT:     .GenericAndParams = %GenericAndParams.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @GenericNoParams(@C.%T.loc26.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%GenericNoParams.6cf
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @GenericAndParams.loc42(@C.%T.loc26.2: type, %U.loc38.2: type) {
// CHECK:STDOUT:   %U.loc38.1: type = symbolic_binding U, 1 [symbolic = %U.loc38.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%GenericAndParams.015
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc55.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc55.2: init %NotGenericNoParams to file.%a.var = class_init () [concrete = constants.%NotGenericNoParams.val]
// CHECK:STDOUT:   %.loc49: init %NotGenericNoParams = converted %.loc55.1, %.loc55.2 [concrete = constants.%NotGenericNoParams.val]
// CHECK:STDOUT:   assign file.%a.var, %.loc49
// CHECK:STDOUT:   %.loc65.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc65.2: init %NotGenericButParams to file.%b.var = class_init () [concrete = constants.%NotGenericButParams.val]
// CHECK:STDOUT:   %.loc57: init %NotGenericButParams = converted %.loc65.1, %.loc65.2 [concrete = constants.%NotGenericButParams.val]
// CHECK:STDOUT:   assign file.%b.var, %.loc57
// CHECK:STDOUT:   %.loc76.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc76.2: init %GenericAndParams.d3d to file.%c.var = class_init () [concrete = constants.%GenericAndParams.val.835]
// CHECK:STDOUT:   %.loc67: init %GenericAndParams.d3d = converted %.loc76.1, %.loc76.2 [concrete = constants.%GenericAndParams.val.835]
// CHECK:STDOUT:   assign file.%c.var, %.loc67
// CHECK:STDOUT:   %.loc89.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc89.2: init %GenericNoParams.efa to file.%d.var = class_init () [concrete = constants.%GenericNoParams.val]
// CHECK:STDOUT:   %.loc78: init %GenericNoParams.efa = converted %.loc89.1, %.loc89.2 [concrete = constants.%GenericNoParams.val]
// CHECK:STDOUT:   assign file.%d.var, %.loc78
// CHECK:STDOUT:   %.loc105.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc105.2: init %GenericAndParams.ed1 to file.%e.var = class_init () [concrete = constants.%GenericAndParams.val.e35]
// CHECK:STDOUT:   %.loc91: init %GenericAndParams.ed1 = converted %.loc105.1, %.loc105.2 [concrete = constants.%GenericAndParams.val.e35]
// CHECK:STDOUT:   assign file.%e.var, %.loc91
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc21(constants.%T) {
// CHECK:STDOUT:   %T.loc17.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc26.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc42(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc38.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc21(constants.%X) {
// CHECK:STDOUT:   %T.loc17.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %T.loc26.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericNoParams => constants.%GenericNoParams.efa
// CHECK:STDOUT:   %GenericAndParams.type => constants.%GenericAndParams.type.23d
// CHECK:STDOUT:   %GenericAndParams.generic => constants.%GenericAndParams.generic.fd2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(constants.%X) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc42(constants.%X, constants.%X) {
// CHECK:STDOUT:   %U.loc38.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_generic_implicit_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [concrete = constants.%A.generic] {} {}
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
// CHECK:STDOUT: --- fail_non_generic_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [concrete = constants.%A.generic] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18.1: type = splice_block %.loc18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc16.2: type = symbolic_binding T, 0 [symbolic = %T.loc16.1 (constants.%T)]
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
// CHECK:STDOUT: generic fn @F(%T.loc16.2: type) {
// CHECK:STDOUT:   %T.loc16.1: type = symbolic_binding T, 0 [symbolic = %T.loc16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc16.2 [symbolic = %T.loc16.1 (constants.%T)]
// CHECK:STDOUT:     %A: type = class_type @A [concrete = constants.%A]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc16.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_implicit_params_only_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Foo;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_implicit_params_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9.1: type = splice_block %.loc9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7.2: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Foo(%T.loc7.2: type) {
// CHECK:STDOUT:   %T.loc7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%T) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
