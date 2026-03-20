// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/uint.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/choice/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/choice/basic.carbon

// --- no_alternative.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
choice Never {}
//@dump-sem-ir-end

// --- one_alternative.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
choice Always {
  Sunny
}

let mood: Always = Always.Sunny;
//@dump-sem-ir-end

// --- multiple_alternatives.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
choice Ordering {
  Less,
  Equivalent,
  Greater,
  Incomparable
}

let less: Ordering = Ordering.Less;
let equiv: Ordering = Ordering.Equivalent;
let greater: Ordering = Ordering.Greater;
let inc: Ordering = Ordering.Incomparable;
//@dump-sem-ir-end

// --- fail_no_alternative_construct.carbon
library "[[@TEST_NAME]]";

choice Never {}

// TODO: Can we produce a better diagnostic?
// CHECK:STDERR: fail_no_alternative_construct.carbon:[[@LINE+4]]:20: error: cannot initialize class with 1 field from struct with 0 fields [StructInitElementCountMismatch]
// CHECK:STDERR: let never: Never = {};
// CHECK:STDERR:                    ^~
// CHECK:STDERR:
let never: Never = {};

// CHECK:STDOUT: --- no_alternative.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Never: type = class_type @Never [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.discriminant: type = struct_type {.discriminant: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.discriminant [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Never.decl: type = class_decl @Never [concrete = constants.%Never] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Never {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.discriminant [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Never
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- one_alternative.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Always: type = class_type @Always [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.discriminant: type = struct_type {.discriminant: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.discriminant [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.discriminant = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %Always.val: %Always = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Always [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Always.decl: type = class_decl @Always [concrete = constants.%Always] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %mood.patt: %pattern_type = value_binding_pattern mood [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Always.ref: type = name_ref Always, %Always.decl [concrete = constants.%Always]
// CHECK:STDOUT:   %mood: %Always = value_binding mood, @__global_init.%Sunny.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Always {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.discriminant [concrete = constants.%complete_type]
// CHECK:STDOUT:   %.loc8.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8.2: %empty_tuple.type = converted %.loc8.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8.3: %struct_type.discriminant = struct_literal (%.loc8.2) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc8.4: ref %Always = temporary_storage
// CHECK:STDOUT:   %.loc8.5: ref %empty_tuple.type = class_element_access %.loc8.4, element0
// CHECK:STDOUT:   %.loc8.6: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8.7: init %empty_tuple.type = converted %.loc8.2, %.loc8.6 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8.8: init %Always to %.loc8.4 = class_init (%.loc8.7) [concrete = constants.%Always.val]
// CHECK:STDOUT:   %.loc8.9: init %Always = converted %.loc8.3, %.loc8.8 [concrete = constants.%Always.val]
// CHECK:STDOUT:   %.loc8.10: ref %Always = temporary %.loc8.4, %.loc8.9
// CHECK:STDOUT:   %.loc8.11: %Always = acquire_value %.loc8.10
// CHECK:STDOUT:   %Sunny: %Always = value_binding Sunny, %.loc8.11
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Always
// CHECK:STDOUT:   .Sunny = %Sunny
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Always.ref: type = name_ref Always, file.%Always.decl [concrete = constants.%Always]
// CHECK:STDOUT:   %Sunny.ref: %Always = name_ref Sunny, @Always.%Sunny
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- multiple_alternatives.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ordering: type = class_type @Ordering [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %u2: type = class_type @UInt, @UInt(%int_2.ecc) [concrete]
// CHECK:STDOUT:   %struct_type.discriminant: type = struct_type {.discriminant: %u2} [concrete]
// CHECK:STDOUT:   %complete_type.de2: <witness> = complete_type_witness %struct_type.discriminant [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a92: type = facet_type <@ImplicitAs, @ImplicitAs(%u2)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.46e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.762: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.899, @Core.IntLiteral.as.ImplicitAs.impl(%int_2.ecc) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c8c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_2.ecc) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0ed: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c8c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a92 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.762) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.539: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u2, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.055: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.539, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dd9: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0ed [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0ed, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_2.ecc) [concrete]
// CHECK:STDOUT:   %bound_method.f58: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.9fd: %u2 = int_value 0 [concrete]
// CHECK:STDOUT:   %struct.559: %struct_type.discriminant = struct_value (%int_0.9fd) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.68f: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.576: %UInt.as.Copy.impl.Op.type.68f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.bd0, @UInt.as.Copy.impl(%int_2.ecc) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.f98: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%int_2.ecc) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.b2d: %UInt.as.Copy.impl.Op.type.f98 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %u2, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.bd7: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a82: type = fn_type_with_self_type %Copy.WithSelf.Op.type.bd7, %Copy.facet [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.bound.fba: <bound method> = bound_method %int_0.9fd, %UInt.as.Copy.impl.Op.b2d [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %UInt.as.Copy.impl.Op.b2d, @UInt.as.Copy.impl.Op(%int_2.ecc) [concrete]
// CHECK:STDOUT:   %bound_method.428: <bound method> = bound_method %int_0.9fd, %UInt.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Ordering.val.9ea: %Ordering = struct_value (%int_0.9fd) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.6b7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0ed [concrete]
// CHECK:STDOUT:   %bound_method.a4b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.b2c: %u2 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct.0ff: %struct_type.discriminant = struct_value (%int_1.b2c) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.bound.67d: <bound method> = bound_method %int_1.b2c, %UInt.as.Copy.impl.Op.b2d [concrete]
// CHECK:STDOUT:   %bound_method.8f6: <bound method> = bound_method %int_1.b2c, %UInt.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Ordering.val.d41: %Ordering = struct_value (%int_1.b2c) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c76: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0ed [concrete]
// CHECK:STDOUT:   %bound_method.c5b: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.788: %u2 = int_value 2 [concrete]
// CHECK:STDOUT:   %struct.6e6: %struct_type.discriminant = struct_value (%int_2.788) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.bound.f0c: <bound method> = bound_method %int_2.788, %UInt.as.Copy.impl.Op.b2d [concrete]
// CHECK:STDOUT:   %bound_method.f1a: <bound method> = bound_method %int_2.788, %UInt.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Ordering.val.e86: %Ordering = struct_value (%int_2.788) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cdf: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0ed [concrete]
// CHECK:STDOUT:   %bound_method.898: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.975: %u2 = int_value 3 [concrete]
// CHECK:STDOUT:   %struct.7bd: %struct_type.discriminant = struct_value (%int_3.975) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.bound.7b5: <bound method> = bound_method %int_3.975, %UInt.as.Copy.impl.Op.b2d [concrete]
// CHECK:STDOUT:   %bound_method.824: <bound method> = bound_method %int_3.975, %UInt.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Ordering.val.a17: %Ordering = struct_value (%int_3.975) [concrete]
// CHECK:STDOUT:   %pattern_type.a36: type = pattern_type %Ordering [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.741: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6) = import_ref Core//prelude/parts/uint, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.46e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.899 = impl_witness_table (%Core.import_ref.741), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c3c: @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op.type (%UInt.as.Copy.impl.Op.type.68f) = import_ref Core//prelude/parts/uint, loc80, loaded [symbolic = @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op (constants.%UInt.as.Copy.impl.Op.576)]
// CHECK:STDOUT:   %Copy.impl_witness_table.bd0 = impl_witness_table (%Core.import_ref.c3c), @UInt.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Ordering.decl: type = class_decl @Ordering [concrete = constants.%Ordering] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less.patt: %pattern_type.a36 = value_binding_pattern less [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ordering.ref.loc18: type = name_ref Ordering, %Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %less: %Ordering = value_binding less, @__global_init.%Less.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %equiv.patt: %pattern_type.a36 = value_binding_pattern equiv [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ordering.ref.loc27: type = name_ref Ordering, %Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %equiv: %Ordering = value_binding equiv, @__global_init.%Equivalent.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater.patt: %pattern_type.a36 = value_binding_pattern greater [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ordering.ref.loc36: type = name_ref Ordering, %Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %greater: %Ordering = value_binding greater, @__global_init.%Greater.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %inc.patt: %pattern_type.a36 = value_binding_pattern inc [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ordering.ref.loc45: type = name_ref Ordering, %Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %inc: %Ordering = value_binding inc, @__global_init.%Incomparable.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Ordering {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.discriminant [concrete = constants.%complete_type.de2]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc8.1: %.055 = impl_witness_access constants.%ImplicitAs.impl_witness.762, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0ed]
// CHECK:STDOUT:   %bound_method.loc8.1: <bound method> = bound_method %int_0, %impl.elem0.loc8.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dd9]
// CHECK:STDOUT:   %specific_fn.loc8.1: <specific function> = specific_function %impl.elem0.loc8.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_2.ecc) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8.2: <bound method> = bound_method %int_0, %specific_fn.loc8.1 [concrete = constants.%bound_method.f58]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %u2 = call %bound_method.loc8.2(%int_0) [concrete = constants.%int_0.9fd]
// CHECK:STDOUT:   %.loc8.1: %u2 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_0.9fd]
// CHECK:STDOUT:   %.loc8.2: %u2 = converted %int_0, %.loc8.1 [concrete = constants.%int_0.9fd]
// CHECK:STDOUT:   %.loc8.3: %struct_type.discriminant = struct_literal (%.loc8.2) [concrete = constants.%struct.559]
// CHECK:STDOUT:   %impl.elem0.loc8.2: %.a82 = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%UInt.as.Copy.impl.Op.b2d]
// CHECK:STDOUT:   %bound_method.loc8.3: <bound method> = bound_method %.loc8.2, %impl.elem0.loc8.2 [concrete = constants.%UInt.as.Copy.impl.Op.bound.fba]
// CHECK:STDOUT:   %specific_fn.loc8.2: <specific function> = specific_function %impl.elem0.loc8.2, @UInt.as.Copy.impl.Op(constants.%int_2.ecc) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8.4: <bound method> = bound_method %.loc8.2, %specific_fn.loc8.2 [concrete = constants.%bound_method.428]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc8: init %u2 = call %bound_method.loc8.4(%.loc8.2) [concrete = constants.%int_0.9fd]
// CHECK:STDOUT:   %.loc8.4: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc8.5: ref %u2 = class_element_access %.loc8.4, element0
// CHECK:STDOUT:   %.loc8.6: init %u2 to %.loc8.5 = in_place_init %UInt.as.Copy.impl.Op.call.loc8 [concrete = constants.%int_0.9fd]
// CHECK:STDOUT:   %.loc8.7: init %Ordering to %.loc8.4 = class_init (%.loc8.6) [concrete = constants.%Ordering.val.9ea]
// CHECK:STDOUT:   %.loc8.8: init %Ordering = converted %.loc8.3, %.loc8.7 [concrete = constants.%Ordering.val.9ea]
// CHECK:STDOUT:   %.loc8.9: ref %Ordering = temporary %.loc8.4, %.loc8.8
// CHECK:STDOUT:   %.loc8.10: %Ordering = acquire_value %.loc8.9
// CHECK:STDOUT:   %Less: %Ordering = value_binding Less, %.loc8.10
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc10.1: %.055 = impl_witness_access constants.%ImplicitAs.impl_witness.762, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0ed]
// CHECK:STDOUT:   %bound_method.loc10.1: <bound method> = bound_method %int_1, %impl.elem0.loc10.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.6b7]
// CHECK:STDOUT:   %specific_fn.loc10.1: <specific function> = specific_function %impl.elem0.loc10.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_2.ecc) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10.2: <bound method> = bound_method %int_1, %specific_fn.loc10.1 [concrete = constants.%bound_method.a4b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10: init %u2 = call %bound_method.loc10.2(%int_1) [concrete = constants.%int_1.b2c]
// CHECK:STDOUT:   %.loc10.1: %u2 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10 [concrete = constants.%int_1.b2c]
// CHECK:STDOUT:   %.loc10.2: %u2 = converted %int_1, %.loc10.1 [concrete = constants.%int_1.b2c]
// CHECK:STDOUT:   %.loc10.3: %struct_type.discriminant = struct_literal (%.loc10.2) [concrete = constants.%struct.0ff]
// CHECK:STDOUT:   %impl.elem0.loc10.2: %.a82 = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%UInt.as.Copy.impl.Op.b2d]
// CHECK:STDOUT:   %bound_method.loc10.3: <bound method> = bound_method %.loc10.2, %impl.elem0.loc10.2 [concrete = constants.%UInt.as.Copy.impl.Op.bound.67d]
// CHECK:STDOUT:   %specific_fn.loc10.2: <specific function> = specific_function %impl.elem0.loc10.2, @UInt.as.Copy.impl.Op(constants.%int_2.ecc) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10.4: <bound method> = bound_method %.loc10.2, %specific_fn.loc10.2 [concrete = constants.%bound_method.8f6]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc10: init %u2 = call %bound_method.loc10.4(%.loc10.2) [concrete = constants.%int_1.b2c]
// CHECK:STDOUT:   %.loc10.4: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc10.5: ref %u2 = class_element_access %.loc10.4, element0
// CHECK:STDOUT:   %.loc10.6: init %u2 to %.loc10.5 = in_place_init %UInt.as.Copy.impl.Op.call.loc10 [concrete = constants.%int_1.b2c]
// CHECK:STDOUT:   %.loc10.7: init %Ordering to %.loc10.4 = class_init (%.loc10.6) [concrete = constants.%Ordering.val.d41]
// CHECK:STDOUT:   %.loc10.8: init %Ordering = converted %.loc10.3, %.loc10.7 [concrete = constants.%Ordering.val.d41]
// CHECK:STDOUT:   %.loc10.9: ref %Ordering = temporary %.loc10.4, %.loc10.8
// CHECK:STDOUT:   %.loc10.10: %Ordering = acquire_value %.loc10.9
// CHECK:STDOUT:   %Equivalent: %Ordering = value_binding Equivalent, %.loc10.10
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc12.1: %.055 = impl_witness_access constants.%ImplicitAs.impl_witness.762, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0ed]
// CHECK:STDOUT:   %bound_method.loc12.1: <bound method> = bound_method %int_2, %impl.elem0.loc12.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c76]
// CHECK:STDOUT:   %specific_fn.loc12.1: <specific function> = specific_function %impl.elem0.loc12.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_2.ecc) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12.2: <bound method> = bound_method %int_2, %specific_fn.loc12.1 [concrete = constants.%bound_method.c5b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12: init %u2 = call %bound_method.loc12.2(%int_2) [concrete = constants.%int_2.788]
// CHECK:STDOUT:   %.loc12.1: %u2 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12 [concrete = constants.%int_2.788]
// CHECK:STDOUT:   %.loc12.2: %u2 = converted %int_2, %.loc12.1 [concrete = constants.%int_2.788]
// CHECK:STDOUT:   %.loc12.3: %struct_type.discriminant = struct_literal (%.loc12.2) [concrete = constants.%struct.6e6]
// CHECK:STDOUT:   %impl.elem0.loc12.2: %.a82 = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%UInt.as.Copy.impl.Op.b2d]
// CHECK:STDOUT:   %bound_method.loc12.3: <bound method> = bound_method %.loc12.2, %impl.elem0.loc12.2 [concrete = constants.%UInt.as.Copy.impl.Op.bound.f0c]
// CHECK:STDOUT:   %specific_fn.loc12.2: <specific function> = specific_function %impl.elem0.loc12.2, @UInt.as.Copy.impl.Op(constants.%int_2.ecc) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12.4: <bound method> = bound_method %.loc12.2, %specific_fn.loc12.2 [concrete = constants.%bound_method.f1a]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc12: init %u2 = call %bound_method.loc12.4(%.loc12.2) [concrete = constants.%int_2.788]
// CHECK:STDOUT:   %.loc12.4: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc12.5: ref %u2 = class_element_access %.loc12.4, element0
// CHECK:STDOUT:   %.loc12.6: init %u2 to %.loc12.5 = in_place_init %UInt.as.Copy.impl.Op.call.loc12 [concrete = constants.%int_2.788]
// CHECK:STDOUT:   %.loc12.7: init %Ordering to %.loc12.4 = class_init (%.loc12.6) [concrete = constants.%Ordering.val.e86]
// CHECK:STDOUT:   %.loc12.8: init %Ordering = converted %.loc12.3, %.loc12.7 [concrete = constants.%Ordering.val.e86]
// CHECK:STDOUT:   %.loc12.9: ref %Ordering = temporary %.loc12.4, %.loc12.8
// CHECK:STDOUT:   %.loc12.10: %Ordering = acquire_value %.loc12.9
// CHECK:STDOUT:   %Greater: %Ordering = value_binding Greater, %.loc12.10
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc14.1: %.055 = impl_witness_access constants.%ImplicitAs.impl_witness.762, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0ed]
// CHECK:STDOUT:   %bound_method.loc14.1: <bound method> = bound_method %int_3, %impl.elem0.loc14.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.cdf]
// CHECK:STDOUT:   %specific_fn.loc14.1: <specific function> = specific_function %impl.elem0.loc14.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_2.ecc) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14.2: <bound method> = bound_method %int_3, %specific_fn.loc14.1 [concrete = constants.%bound_method.898]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14: init %u2 = call %bound_method.loc14.2(%int_3) [concrete = constants.%int_3.975]
// CHECK:STDOUT:   %.loc14.1: %u2 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14 [concrete = constants.%int_3.975]
// CHECK:STDOUT:   %.loc14.2: %u2 = converted %int_3, %.loc14.1 [concrete = constants.%int_3.975]
// CHECK:STDOUT:   %.loc14.3: %struct_type.discriminant = struct_literal (%.loc14.2) [concrete = constants.%struct.7bd]
// CHECK:STDOUT:   %impl.elem0.loc14.2: %.a82 = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%UInt.as.Copy.impl.Op.b2d]
// CHECK:STDOUT:   %bound_method.loc14.3: <bound method> = bound_method %.loc14.2, %impl.elem0.loc14.2 [concrete = constants.%UInt.as.Copy.impl.Op.bound.7b5]
// CHECK:STDOUT:   %specific_fn.loc14.2: <specific function> = specific_function %impl.elem0.loc14.2, @UInt.as.Copy.impl.Op(constants.%int_2.ecc) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14.4: <bound method> = bound_method %.loc14.2, %specific_fn.loc14.2 [concrete = constants.%bound_method.824]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc14: init %u2 = call %bound_method.loc14.4(%.loc14.2) [concrete = constants.%int_3.975]
// CHECK:STDOUT:   %.loc14.4: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc14.5: ref %u2 = class_element_access %.loc14.4, element0
// CHECK:STDOUT:   %.loc14.6: init %u2 to %.loc14.5 = in_place_init %UInt.as.Copy.impl.Op.call.loc14 [concrete = constants.%int_3.975]
// CHECK:STDOUT:   %.loc14.7: init %Ordering to %.loc14.4 = class_init (%.loc14.6) [concrete = constants.%Ordering.val.a17]
// CHECK:STDOUT:   %.loc14.8: init %Ordering = converted %.loc14.3, %.loc14.7 [concrete = constants.%Ordering.val.a17]
// CHECK:STDOUT:   %.loc14.9: ref %Ordering = temporary %.loc14.4, %.loc14.8
// CHECK:STDOUT:   %.loc14.10: %Ordering = acquire_value %.loc14.9
// CHECK:STDOUT:   %Incomparable: %Ordering = value_binding Incomparable, %.loc14.10
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Ordering
// CHECK:STDOUT:   .Less = %Less
// CHECK:STDOUT:   .Equivalent = %Equivalent
// CHECK:STDOUT:   .Greater = %Greater
// CHECK:STDOUT:   .Incomparable = %Incomparable
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Ordering.ref.loc20: type = name_ref Ordering, file.%Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %Less.ref: %Ordering = name_ref Less, @Ordering.%Less
// CHECK:STDOUT:   %Ordering.ref.loc29: type = name_ref Ordering, file.%Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %Equivalent.ref: %Ordering = name_ref Equivalent, @Ordering.%Equivalent
// CHECK:STDOUT:   %Ordering.ref.loc38: type = name_ref Ordering, file.%Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %Greater.ref: %Ordering = name_ref Greater, @Ordering.%Greater
// CHECK:STDOUT:   %Ordering.ref.loc47: type = name_ref Ordering, file.%Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %Incomparable.ref: %Ordering = name_ref Incomparable, @Ordering.%Incomparable
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
