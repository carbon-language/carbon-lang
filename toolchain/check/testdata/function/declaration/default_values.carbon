// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/default_values.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/default_values.carbon

// --- fail_value_not_constant.carbon
library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR: fail_value_not_constant.carbon:[[@LINE+4]]:19: error: default value is not a constant [PatternDefaultValueNotConstant]
// CHECK:STDERR: fn H(x: C, y: C = x);
// CHECK:STDERR:                   ^
// CHECK:STDERR:
fn H(x: C, y: C = x);

// --- fail_type_mismatch.carbon
library "[[@TEST_NAME]]";

class C {}
class D {}

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+7]]:13: error: cannot implicitly convert expression of type `D` to `C` [ConversionFailure]
// CHECK:STDERR: fn K(x: C = {} as D);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:13: note: type `D` does not implement interface `Core.ImplicitAs(C)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: fn K(x: C = {} as D);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
fn K(x: C = {} as D);

// --- fail_pattern_defaults_not_in_parameter_list.carbon
library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR: fail_pattern_defaults_not_in_parameter_list.carbon:[[@LINE+4]]:19: error: default values are only supported in parameter lists [PatternDefaultValueNotInParameterList]
// CHECK:STDERR: let (y: C, x: C = {}) = ({}, {});
// CHECK:STDERR:                   ^~
// CHECK:STDERR:
let (y: C, x: C = {}) = ({}, {});

// --- fail_default_values_unspecified.carbon
library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR: fail_default_values_unspecified.carbon:[[@LINE+4]]:24: error: found unspecified default parameter value in the function's first owning declaration [PatternDefaultValueNotSpecified]
// CHECK:STDERR: fn Z(x: C = {}, y: C = _);
// CHECK:STDERR:                        ^
// CHECK:STDERR:
fn Z(x: C = {}, y: C = _);

// --- fail_required_default_values_missing.carbon
library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR: fail_required_default_values_missing.carbon:[[@LINE+17]]:35: error: this pattern is missing a required default value. [RequiredPatternDefaultValueMissing]
// CHECK:STDERR: fn Z(v: C, x: C = {}, (y: C = {}, z: C), w: C = {}, k: C);
// CHECK:STDERR:                                   ^~~~
// CHECK:STDERR: fail_required_default_values_missing.carbon:[[@LINE+14]]:24: note: all patterns to the right of this first pattern with a default value must also specify a default value. [RequiredPatternDefaultValueFirstDefault]
// CHECK:STDERR: fn Z(v: C, x: C = {}, (y: C = {}, z: C), w: C = {}, k: C);
// CHECK:STDERR:                        ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_required_default_values_missing.carbon:[[@LINE+10]]:23: error: this pattern is missing a required default value. [RequiredPatternDefaultValueMissing]
// CHECK:STDERR: fn Z(v: C, x: C = {}, (y: C = {}, z: C), w: C = {}, k: C);
// CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_required_default_values_missing.carbon:[[@LINE+7]]:53: note: this pattern is also missing a required default value. [RequiredPatternDefaultValueMissingAdditional]
// CHECK:STDERR: fn Z(v: C, x: C = {}, (y: C = {}, z: C), w: C = {}, k: C);
// CHECK:STDERR:                                                     ^~~~
// CHECK:STDERR: fail_required_default_values_missing.carbon:[[@LINE+4]]:12: note: all patterns to the right of this first pattern with a default value must also specify a default value. [RequiredPatternDefaultValueFirstDefault]
// CHECK:STDERR: fn Z(v: C, x: C = {}, (y: C = {}, z: C), w: C = {}, k: C);
// CHECK:STDERR:            ^~~~
// CHECK:STDERR:
fn Z(v: C, x: C = {}, (y: C = {}, z: C), w: C = {}, k: C);

// --- fail_nesting_required_default_values_missing.carbon
library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR: fail_nesting_required_default_values_missing.carbon:[[@LINE+14]]:36: error: this pattern is missing a required default value. [RequiredPatternDefaultValueMissing]
// CHECK:STDERR: fn F(a: C, (b: C = {}, (c: C = {}, (d: C, e: C))));
// CHECK:STDERR:                                    ^~~~~~~~~~~~
// CHECK:STDERR: fail_nesting_required_default_values_missing.carbon:[[@LINE+11]]:25: note: all patterns to the right of this first pattern with a default value must also specify a default value. [RequiredPatternDefaultValueFirstDefault]
// CHECK:STDERR: fn F(a: C, (b: C = {}, (c: C = {}, (d: C, e: C))));
// CHECK:STDERR:                         ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_nesting_required_default_values_missing.carbon:[[@LINE+7]]:24: error: this pattern is missing a required default value. [RequiredPatternDefaultValueMissing]
// CHECK:STDERR: fn F(a: C, (b: C = {}, (c: C = {}, (d: C, e: C))));
// CHECK:STDERR:                        ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_nesting_required_default_values_missing.carbon:[[@LINE+4]]:13: note: all patterns to the right of this first pattern with a default value must also specify a default value. [RequiredPatternDefaultValueFirstDefault]
// CHECK:STDERR: fn F(a: C, (b: C = {}, (c: C = {}, (d: C, e: C))));
// CHECK:STDERR:             ^~~~
// CHECK:STDERR:
fn F(a: C, (b: C = {}, (c: C = {}, (d: C, e: C))));

// --- fail_explicit_conversion_not_supported.carbon
library "[[@TEST_NAME]]";

class C {}
class D {}
impl C as Core.As(D) {
  fn Convert(unused self: C) -> D { return {} as D; }
}

// CHECK:STDERR: fail_explicit_conversion_not_supported.carbon:[[@LINE+7]]:13: error: cannot implicitly convert expression of type `C` to `D` [ConversionFailure]
// CHECK:STDERR: fn F(a: D = {} as C);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR: fail_explicit_conversion_not_supported.carbon:[[@LINE+4]]:13: note: type `C` does not implement interface `Core.ImplicitAs(D)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: fn F(a: D = {} as C);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
fn F(a: D = {} as C);

// --- implicit.carbon
library "[[@TEST_NAME]]";

class C {}
class D {}
impl C as Core.ImplicitAs(D) {
  fn Convert(self: C) -> D { return self; }
}

//@dump-sem-ir-begin
fn F(a: D = {} as C);
//@dump-sem-ir-end

// --- nested.carbon
library "[[@TEST_NAME]]";

class C {}

//@dump-sem-ir-begin
fn F(a: C, (b: C, (c: C, (d: C, e: C))) = ({}, ({}, ({}, {}))));
//@dump-sem-ir-end

// --- basic.carbon
library "[[@TEST_NAME]]";

class C {}

//@dump-sem-ir-begin
fn F(x: C = {});
//@dump-sem-ir-end

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2be: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness @C.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.d8d: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type: type = fn_type @C.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert: %C.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.2be = facet_value %C, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6cd: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.ee7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6cd, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.d8d = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d8d = wrapper_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.68d: %pattern_type.d8d = default_value_pattern %a.patt, index: 0 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %C.val, %C.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.d8d = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.d8d = wrapper_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:     %.loc10_11: %pattern_type.d8d = default_value_pattern %a.patt, index: 0 [concrete = constants.%.68d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_14.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc10_14.2: ref %C = temporary_storage
// CHECK:STDOUT:     %.loc10_14.3: init %C to %.loc10_14.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc10_16.1: init %C = converted %.loc10_14.1, %.loc10_14.3 [concrete = constants.%C.val]
// CHECK:STDOUT:     %a.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %a: %D = wrapper_binding a, %a.param
// CHECK:STDOUT:     %impl.elem0: %.ee7 = impl_witness_access constants.%ImplicitAs.impl_witness, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %.loc10_16.1, %impl.elem0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %.loc10_16.2: ref %D = temporary_storage
// CHECK:STDOUT:     %.loc10_16.3: ref %C = temporary %.loc10_14.2, %.loc10_16.1 [concrete = constants.%.115]
// CHECK:STDOUT:     %.loc10_16.4: %C = acquire_value %.loc10_16.3 [concrete = constants.%C.val]
// CHECK:STDOUT:     %C.as.ImplicitAs.impl.Convert.call: init %D to %.loc10_16.2 = call %bound_method(%.loc10_16.4)
// CHECK:STDOUT:     %.loc10_16.5: init %D = converted %.loc10_16.1, %C.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:     %.loc10_16.6: ref %D = temporary %.loc10_16.2, %.loc10_16.5
// CHECK:STDOUT:     %.loc10_16.7: %D = acquire_value %.loc10_16.6
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %D) {
// CHECK:STDOUT: !default_values:
// CHECK:STDOUT:   %.loc10_16.7: %D = acquire_value %.loc10_16.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.98b = wrapper_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %b.param_patt: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.98b = wrapper_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   %c.param_patt: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = wrapper_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   %d.param_patt: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.98b = wrapper_binding_pattern d, %d.param_patt [concrete]
// CHECK:STDOUT:   %e.param_patt: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %e.patt: %pattern_type.98b = wrapper_binding_pattern e, %e.param_patt [concrete]
// CHECK:STDOUT:   %tuple.type.748: type = tuple_type (%C, %C) [concrete]
// CHECK:STDOUT:   %pattern_type.8b4: type = pattern_type %tuple.type.748 [concrete]
// CHECK:STDOUT:   %.fe8: %pattern_type.8b4 = tuple_pattern (%d.patt, %e.patt) [concrete]
// CHECK:STDOUT:   %tuple.type.d76: type = tuple_type (%C, %tuple.type.748) [concrete]
// CHECK:STDOUT:   %pattern_type.4fa: type = pattern_type %tuple.type.d76 [concrete]
// CHECK:STDOUT:   %.972: %pattern_type.4fa = tuple_pattern (%c.patt, %.fe8) [concrete]
// CHECK:STDOUT:   %tuple.type.a40: type = tuple_type (%C, %tuple.type.d76) [concrete]
// CHECK:STDOUT:   %pattern_type.f5f: type = pattern_type %tuple.type.a40 [concrete]
// CHECK:STDOUT:   %.341: %pattern_type.f5f = tuple_pattern (%b.patt, %.972) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.b6b: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple.9a3: %tuple.type.b6b = tuple_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %tuple.type.6ca: type = tuple_type (%empty_struct_type, %tuple.type.b6b) [concrete]
// CHECK:STDOUT:   %tuple.56c: %tuple.type.6ca = tuple_value (%empty_struct, %tuple.9a3) [concrete]
// CHECK:STDOUT:   %tuple.type.bcdf: type = tuple_type (%empty_struct_type, %tuple.type.6ca) [concrete]
// CHECK:STDOUT:   %tuple.4f4: %tuple.type.bcdf = tuple_value (%empty_struct, %tuple.56c) [concrete]
// CHECK:STDOUT:   %.b6d: %pattern_type.f5f = default_value_pattern %.341, index: 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %tuple.d3e: %tuple.type.748 = tuple_value (%C.val, %C.val) [concrete]
// CHECK:STDOUT:   %tuple.109: %tuple.type.d76 = tuple_value (%C.val, %tuple.d3e) [concrete]
// CHECK:STDOUT:   %tuple.800: %tuple.type.a40 = tuple_value (%C.val, %tuple.109) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.98b = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.98b = wrapper_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.98b = value_param_pattern [concrete = constants.%b.param_patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.98b = wrapper_binding_pattern b, %b.param_patt [concrete = constants.%b.patt]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.98b = value_param_pattern [concrete = constants.%c.param_patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = wrapper_binding_pattern c, %c.param_patt [concrete = constants.%c.patt]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.98b = value_param_pattern [concrete = constants.%d.param_patt]
// CHECK:STDOUT:     %d.patt: %pattern_type.98b = wrapper_binding_pattern d, %d.param_patt [concrete = constants.%d.patt]
// CHECK:STDOUT:     %e.param_patt: %pattern_type.98b = value_param_pattern [concrete = constants.%e.param_patt]
// CHECK:STDOUT:     %e.patt: %pattern_type.98b = wrapper_binding_pattern e, %e.param_patt [concrete = constants.%e.patt]
// CHECK:STDOUT:     %.loc6_37: %pattern_type.8b4 = tuple_pattern (%d.patt, %e.patt) [concrete = constants.%.fe8]
// CHECK:STDOUT:     %.loc6_38: %pattern_type.4fa = tuple_pattern (%c.patt, %.loc6_37) [concrete = constants.%.972]
// CHECK:STDOUT:     %.loc6_39: %pattern_type.f5f = tuple_pattern (%b.patt, %.loc6_38) [concrete = constants.%.341]
// CHECK:STDOUT:     %.loc6_41: %pattern_type.f5f = default_value_pattern %.loc6_39, index: 0 [concrete = constants.%.b6d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_45.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc6_50.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc6_55.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc6_59.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc6_60.1: %tuple.type.b6b = tuple_literal (%.loc6_55.1, %.loc6_59.1) [concrete = constants.%tuple.9a3]
// CHECK:STDOUT:     %.loc6_61.1: %tuple.type.6ca = tuple_literal (%.loc6_50.1, %.loc6_60.1) [concrete = constants.%tuple.56c]
// CHECK:STDOUT:     %.loc6_62.1: %tuple.type.bcdf = tuple_literal (%.loc6_45.1, %.loc6_61.1) [concrete = constants.%tuple.4f4]
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc6_9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = wrapper_binding a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc6_16: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %b: %C = wrapper_binding b, %b.param
// CHECK:STDOUT:     %c.param: %C = value_param call_param2
// CHECK:STDOUT:     %C.ref.loc6_23: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = wrapper_binding c, %c.param
// CHECK:STDOUT:     %d.param: %C = value_param call_param3
// CHECK:STDOUT:     %C.ref.loc6_30: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %d: %C = wrapper_binding d, %d.param
// CHECK:STDOUT:     %e.param: %C = value_param call_param4
// CHECK:STDOUT:     %C.ref.loc6_36: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %e: %C = wrapper_binding e, %e.param
// CHECK:STDOUT:     %tuple.loc6_37: %tuple.type.748 = tuple_value (%d.param, %e.param)
// CHECK:STDOUT:     %tuple.loc6_38: %tuple.type.d76 = tuple_value (%c.param, %tuple.loc6_37)
// CHECK:STDOUT:     %tuple.loc6_39: %tuple.type.a40 = tuple_value (%b.param, %tuple.loc6_38)
// CHECK:STDOUT:     %.loc6_45.2: ref %C = temporary_storage
// CHECK:STDOUT:     %.loc6_45.3: init %C to %.loc6_45.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc6_62.2: init %C = converted %.loc6_45.1, %.loc6_45.3 [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc6_62.3: ref %C = temporary %.loc6_45.2, %.loc6_62.2 [concrete = constants.%.115]
// CHECK:STDOUT:     %.loc6_62.4: %C = acquire_value %.loc6_62.3 [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc6_50.2: ref %C = temporary_storage
// CHECK:STDOUT:     %.loc6_50.3: init %C to %.loc6_50.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc6_61.2: init %C = converted %.loc6_50.1, %.loc6_50.3 [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc6_61.3: ref %C = temporary %.loc6_50.2, %.loc6_61.2 [concrete = constants.%.115]
// CHECK:STDOUT:     %.loc6_61.4: %C = acquire_value %.loc6_61.3 [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc6_55.2: ref %C = temporary_storage
// CHECK:STDOUT:     %.loc6_55.3: init %C to %.loc6_55.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc6_60.2: init %C = converted %.loc6_55.1, %.loc6_55.3 [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc6_60.3: ref %C = temporary %.loc6_55.2, %.loc6_60.2 [concrete = constants.%.115]
// CHECK:STDOUT:     %.loc6_60.4: %C = acquire_value %.loc6_60.3 [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc6_59.2: ref %C = temporary_storage
// CHECK:STDOUT:     %.loc6_59.3: init %C to %.loc6_59.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc6_60.5: init %C = converted %.loc6_59.1, %.loc6_59.3 [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc6_60.6: ref %C = temporary %.loc6_59.2, %.loc6_60.5 [concrete = constants.%.115]
// CHECK:STDOUT:     %.loc6_60.7: %C = acquire_value %.loc6_60.6 [concrete = constants.%C.val]
// CHECK:STDOUT:     %tuple.loc6_60: %tuple.type.748 = tuple_value (%.loc6_60.4, %.loc6_60.7) [concrete = constants.%tuple.d3e]
// CHECK:STDOUT:     %.loc6_61.5: %tuple.type.748 = converted %.loc6_60.1, %tuple.loc6_60 [concrete = constants.%tuple.d3e]
// CHECK:STDOUT:     %tuple.loc6_61: %tuple.type.d76 = tuple_value (%.loc6_61.4, %.loc6_61.5) [concrete = constants.%tuple.109]
// CHECK:STDOUT:     %.loc6_62.5: %tuple.type.d76 = converted %.loc6_61.1, %tuple.loc6_61 [concrete = constants.%tuple.109]
// CHECK:STDOUT:     %tuple.loc6_62: %tuple.type.a40 = tuple_value (%.loc6_62.4, %.loc6_62.5) [concrete = constants.%tuple.800]
// CHECK:STDOUT:     %.loc6_62.6: %tuple.type.a40 = converted %.loc6_62.1, %tuple.loc6_62 [concrete = constants.%tuple.800]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %C, %b.param: %C, %c.param: %C, %d.param: %C, %e.param: %C) {
// CHECK:STDOUT: !default_values:
// CHECK:STDOUT:   %.loc6_62.6: %tuple.type.a40 = converted %.loc6_62.1, %tuple.loc6_62 [concrete = constants.%tuple.800]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.015: %pattern_type = default_value_pattern %x.patt, index: 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %.loc6_11: %pattern_type = default_value_pattern %x.patt, index: 0 [concrete = constants.%.015]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_14.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %x.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %x: %C = wrapper_binding x, %x.param
// CHECK:STDOUT:     %.loc6_14.2: ref %C = temporary_storage
// CHECK:STDOUT:     %.loc6_14.3: init %C to %.loc6_14.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc6_14.4: init %C = converted %.loc6_14.1, %.loc6_14.3 [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc6_14.5: ref %C = temporary %.loc6_14.2, %.loc6_14.4 [concrete = constants.%.115]
// CHECK:STDOUT:     %.loc6_14.6: %C = acquire_value %.loc6_14.5 [concrete = constants.%C.val]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %C) {
// CHECK:STDOUT: !default_values:
// CHECK:STDOUT:   %.loc6_14.6: %C = acquire_value %.loc6_14.5 [concrete = constants.%C.val]
// CHECK:STDOUT: }
// CHECK:STDOUT:
