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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic_vs_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic_vs_params.carbon

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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.c8d: type = generic_class_type @GenericAndParams.loc6 [concrete]
// CHECK:STDOUT:   %GenericAndParams.generic.1e4: %GenericAndParams.type.c8d = struct_value () [concrete]
// CHECK:STDOUT:   %GenericAndParams.2ce: type = class_type @GenericAndParams.loc6, @GenericAndParams.loc6(%T) [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %GenericNoParams.fbf: type = class_type @GenericNoParams, @GenericNoParams(%T) [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %GenericAndParams.type.3ce: type = generic_class_type @GenericAndParams.loc10, @C(%T) [symbolic]
// CHECK:STDOUT:   %GenericAndParams.generic.54a: %GenericAndParams.type.3ce = struct_value () [symbolic]
// CHECK:STDOUT:   %GenericAndParams.425: type = class_type @GenericAndParams.loc10, @GenericAndParams.loc10(%T, %U) [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %pattern_type.71d: type = pattern_type %NotGenericNoParams [concrete]
// CHECK:STDOUT:   %NotGenericNoParams.val: %NotGenericNoParams = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.4f9: type = pattern_type %NotGenericButParams [concrete]
// CHECK:STDOUT:   %NotGenericButParams.val: %NotGenericButParams = struct_value () [concrete]
// CHECK:STDOUT:   %GenericAndParams.2bb: type = class_type @GenericAndParams.loc6, @GenericAndParams.loc6(%X) [concrete]
// CHECK:STDOUT:   %pattern_type.569: type = pattern_type %GenericAndParams.2bb [concrete]
// CHECK:STDOUT:   %GenericAndParams.val.0b2: %GenericAndParams.2bb = struct_value () [concrete]
// CHECK:STDOUT:   %C.fac: type = class_type @C, @C(%X) [concrete]
// CHECK:STDOUT:   %GenericNoParams.fa1: type = class_type @GenericNoParams, @GenericNoParams(%X) [concrete]
// CHECK:STDOUT:   %GenericAndParams.type.c20: type = generic_class_type @GenericAndParams.loc10, @C(%X) [concrete]
// CHECK:STDOUT:   %GenericAndParams.generic.a55: %GenericAndParams.type.c20 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.518: type = pattern_type %GenericNoParams.fa1 [concrete]
// CHECK:STDOUT:   %GenericNoParams.val: %GenericNoParams.fa1 = struct_value () [concrete]
// CHECK:STDOUT:   %GenericAndParams.91f: type = class_type @GenericAndParams.loc10, @GenericAndParams.loc10(%X, %X) [concrete]
// CHECK:STDOUT:   %pattern_type.806: type = pattern_type %GenericAndParams.91f [concrete]
// CHECK:STDOUT:   %GenericAndParams.val.99b: %GenericAndParams.91f = struct_value () [concrete]
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
// CHECK:STDOUT:   %GenericAndParams.decl: %GenericAndParams.type.c8d = class_decl @GenericAndParams.loc6 [concrete = constants.%GenericAndParams.generic.1e4] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc6_24.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_24.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc8_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.71d = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.71d = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %NotGenericNoParams = var %a.var_patt [concrete]
// CHECK:STDOUT:   %NotGenericNoParams.ref: type = name_ref NotGenericNoParams, %NotGenericNoParams.decl [concrete = constants.%NotGenericNoParams]
// CHECK:STDOUT:   %a: ref %NotGenericNoParams = bind_name a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.4f9 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.4f9 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %NotGenericButParams = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc16: type = splice_block %NotGenericButParams [concrete = constants.%NotGenericButParams] {
// CHECK:STDOUT:     %NotGenericButParams.ref: %NotGenericButParams.type = name_ref NotGenericButParams, %NotGenericButParams.decl [concrete = constants.%NotGenericButParams.generic]
// CHECK:STDOUT:     %NotGenericButParams: type = class_type @NotGenericButParams [concrete = constants.%NotGenericButParams]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %NotGenericButParams = bind_name b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.569 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.569 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %GenericAndParams.2bb = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc17: type = splice_block %GenericAndParams.loc17 [concrete = constants.%GenericAndParams.2bb] {
// CHECK:STDOUT:     %GenericAndParams.ref.loc17: %GenericAndParams.type.c8d = name_ref GenericAndParams, %GenericAndParams.decl [concrete = constants.%GenericAndParams.generic.1e4]
// CHECK:STDOUT:     %X.ref.loc17: type = name_ref X, %X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %GenericAndParams.loc17: type = class_type @GenericAndParams.loc6, @GenericAndParams.loc6(constants.%X) [concrete = constants.%GenericAndParams.2bb]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %GenericAndParams.2bb = bind_name c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.518 = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.518 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %GenericNoParams.fa1 = var %d.var_patt [concrete]
// CHECK:STDOUT:   %.loc18_12.1: type = splice_block %GenericNoParams.ref [concrete = constants.%GenericNoParams.fa1] {
// CHECK:STDOUT:     %C.ref.loc18: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %X.ref.loc18: type = name_ref X, %X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C.loc18: type = class_type @C, @C(constants.%X) [concrete = constants.%C.fac]
// CHECK:STDOUT:     %.loc18_12.2: type = specific_constant @C.%GenericNoParams.decl, @C(constants.%X) [concrete = constants.%GenericNoParams.fa1]
// CHECK:STDOUT:     %GenericNoParams.ref: type = name_ref GenericNoParams, %.loc18_12.2 [concrete = constants.%GenericNoParams.fa1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %GenericNoParams.fa1 = bind_name d, %d.var [concrete = %d.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.806 = binding_pattern e [concrete]
// CHECK:STDOUT:     %e.var_patt: %pattern_type.806 = var_pattern %e.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %GenericAndParams.91f = var %e.var_patt [concrete]
// CHECK:STDOUT:   %.loc19_31: type = splice_block %GenericAndParams.loc19 [concrete = constants.%GenericAndParams.91f] {
// CHECK:STDOUT:     %C.ref.loc19: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %X.ref.loc19_10: type = name_ref X, %X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %C.loc19: type = class_type @C, @C(constants.%X) [concrete = constants.%C.fac]
// CHECK:STDOUT:     %.loc19_12: %GenericAndParams.type.c20 = specific_constant @C.%GenericAndParams.decl, @C(constants.%X) [concrete = constants.%GenericAndParams.generic.a55]
// CHECK:STDOUT:     %GenericAndParams.ref.loc19: %GenericAndParams.type.c20 = name_ref GenericAndParams, %.loc19_12 [concrete = constants.%GenericAndParams.generic.a55]
// CHECK:STDOUT:     %X.ref.loc19_30: type = name_ref X, %X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %GenericAndParams.loc19: type = class_type @GenericAndParams.loc10, @GenericAndParams.loc10(constants.%X, constants.%X) [concrete = constants.%GenericAndParams.91f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: ref %GenericAndParams.91f = bind_name e, %e.var [concrete = %e.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NotGenericNoParams {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NotGenericNoParams
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NotGenericButParams {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NotGenericButParams
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @GenericAndParams.loc6(%T.loc6_24.2: type) {
// CHECK:STDOUT:   %T.loc6_24.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_24.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%GenericAndParams.2ce
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc8_9.2: type) {
// CHECK:STDOUT:   %T.loc8_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericNoParams: type = class_type @GenericNoParams, @GenericNoParams(%T.loc8_9.1) [symbolic = %GenericNoParams (constants.%GenericNoParams.fbf)]
// CHECK:STDOUT:   %GenericAndParams.type: type = generic_class_type @GenericAndParams.loc10, @C(%T.loc8_9.1) [symbolic = %GenericAndParams.type (constants.%GenericAndParams.type.3ce)]
// CHECK:STDOUT:   %GenericAndParams.generic: @C.%GenericAndParams.type (%GenericAndParams.type.3ce) = struct_value () [symbolic = %GenericAndParams.generic (constants.%GenericAndParams.generic.54a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %GenericNoParams.decl: type = class_decl @GenericNoParams [symbolic = @C.%GenericNoParams (constants.%GenericNoParams.fbf)] {} {}
// CHECK:STDOUT:     %GenericAndParams.decl: @C.%GenericAndParams.type (%GenericAndParams.type.3ce) = class_decl @GenericAndParams.loc10 [symbolic = @C.%GenericAndParams.generic (constants.%GenericAndParams.generic.54a)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.loc10_26.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc10_26.2 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:     .GenericNoParams = %GenericNoParams.decl
// CHECK:STDOUT:     .GenericAndParams = %GenericAndParams.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @GenericNoParams(@C.%T.loc8_9.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%GenericNoParams.fbf
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @GenericAndParams.loc10(@C.%T.loc8_9.2: type, %U.loc10_26.1: type) {
// CHECK:STDOUT:   %U.loc10_26.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc10_26.2 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%GenericAndParams.425
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc15_30.2: init %NotGenericNoParams = class_init (), file.%a.var [concrete = constants.%NotGenericNoParams.val]
// CHECK:STDOUT:   %.loc15_1: init %NotGenericNoParams = converted %.loc15_30.1, %.loc15_30.2 [concrete = constants.%NotGenericNoParams.val]
// CHECK:STDOUT:   assign file.%a.var, %.loc15_1
// CHECK:STDOUT:   %.loc16_33.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc16_33.2: init %NotGenericButParams = class_init (), file.%b.var [concrete = constants.%NotGenericButParams.val]
// CHECK:STDOUT:   %.loc16_1: init %NotGenericButParams = converted %.loc16_33.1, %.loc16_33.2 [concrete = constants.%NotGenericButParams.val]
// CHECK:STDOUT:   assign file.%b.var, %.loc16_1
// CHECK:STDOUT:   %.loc17_31.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc17_31.2: init %GenericAndParams.2bb = class_init (), file.%c.var [concrete = constants.%GenericAndParams.val.0b2]
// CHECK:STDOUT:   %.loc17_1: init %GenericAndParams.2bb = converted %.loc17_31.1, %.loc17_31.2 [concrete = constants.%GenericAndParams.val.0b2]
// CHECK:STDOUT:   assign file.%c.var, %.loc17_1
// CHECK:STDOUT:   %.loc18_32.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc18_32.2: init %GenericNoParams.fa1 = class_init (), file.%d.var [concrete = constants.%GenericNoParams.val]
// CHECK:STDOUT:   %.loc18_1: init %GenericNoParams.fa1 = converted %.loc18_32.1, %.loc18_32.2 [concrete = constants.%GenericNoParams.val]
// CHECK:STDOUT:   assign file.%d.var, %.loc18_1
// CHECK:STDOUT:   %.loc19_36.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc19_36.2: init %GenericAndParams.91f = class_init (), file.%e.var [concrete = constants.%GenericAndParams.val.99b]
// CHECK:STDOUT:   %.loc19_1: init %GenericAndParams.91f = converted %.loc19_36.1, %.loc19_36.2 [concrete = constants.%GenericAndParams.val.99b]
// CHECK:STDOUT:   assign file.%e.var, %.loc19_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc6(constants.%T) {
// CHECK:STDOUT:   %T.loc6_24.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc8_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc10(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc10_26.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc6(constants.%X) {
// CHECK:STDOUT:   %T.loc6_24.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %T.loc8_9.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericNoParams => constants.%GenericNoParams.fa1
// CHECK:STDOUT:   %GenericAndParams.type => constants.%GenericAndParams.type.c20
// CHECK:STDOUT:   %GenericAndParams.generic => constants.%GenericAndParams.generic.a55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericNoParams(constants.%X) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericAndParams.loc10(constants.%X, constants.%X) {
// CHECK:STDOUT:   %U.loc10_26.2 => constants.%X
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
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
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
// CHECK:STDOUT:     %T.loc11_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc11_6.2: type) {
// CHECK:STDOUT:   %T.loc11_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_6.2 [symbolic = %T.loc11_6.1 (constants.%T)]
// CHECK:STDOUT:     %A: type = class_type @A [concrete = constants.%A]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc11_6.1 => constants.%T
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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
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
// CHECK:STDOUT:     %T.loc8_11.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_11.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Foo(%T.loc8_11.2: type) {
// CHECK:STDOUT:   %T.loc8_11.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_11.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%T) {
// CHECK:STDOUT:   %T.loc8_11.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
