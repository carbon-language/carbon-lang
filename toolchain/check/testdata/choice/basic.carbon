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
// CHECK:STDOUT:   %.f5f: ref %Always = temporary invalid, %Always.val [concrete]
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
// CHECK:STDOUT:   %.loc6_1.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_1.2: %empty_tuple.type = converted %.loc6_1.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_1.3: %struct_type.discriminant = struct_literal (%.loc6_1.2) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc6_1.4: ref %Always = temporary_storage
// CHECK:STDOUT:   %.loc6_1.5: ref %empty_tuple.type = class_element_access %.loc6_1.4, element0
// CHECK:STDOUT:   %.loc6_1.6: init %empty_tuple.type to %.loc6_1.5 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_1.7: init %empty_tuple.type = converted %.loc6_1.2, %.loc6_1.6 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_1.8: init %empty_tuple.type to %.loc6_1.5 = in_place_init %.loc6_1.7 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_1.9: init %Always to %.loc6_1.4 = class_init (%.loc6_1.8) [concrete = constants.%Always.val]
// CHECK:STDOUT:   %.loc6_1.10: init %Always = converted %.loc6_1.3, %.loc6_1.9 [concrete = constants.%Always.val]
// CHECK:STDOUT:   %.loc6_1.11: ref %Always = temporary %.loc6_1.4, %.loc6_1.10 [concrete = constants.%.f5f]
// CHECK:STDOUT:   %.loc6_1.12: %Always = acquire_value %.loc6_1.11 [concrete = constants.%Always.val]
// CHECK:STDOUT:   %Sunny: %Always = value_binding Sunny, %.loc6_1.12
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
// CHECK:STDOUT:   %complete_type.5ef: <witness> = complete_type_witness %struct_type.discriminant [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.239: type = facet_type <@ImplicitAs, @ImplicitAs(%u2)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.be8: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.929: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.be8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.90c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b5d, @Core.IntLiteral.as.ImplicitAs.impl(%int_2.ecc) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.caf: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_2.ecc) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.a36: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.caf = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.239 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.90c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.15e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u2, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.e52: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.15e, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0d4: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.a36 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.a36, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_2.ecc) [concrete]
// CHECK:STDOUT:   %bound_method.c7c: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.39d: %u2 = int_value 0 [concrete]
// CHECK:STDOUT:   %struct.373: %struct_type.discriminant = struct_value (%int_0.39d) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.2ce: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.65a: %UInt.as.Copy.impl.Op.type.2ce = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.2aa: <witness> = impl_witness imports.%Copy.impl_witness_table.744, @UInt.as.Copy.impl(%int_2.ecc) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.667: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%int_2.ecc) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.56a: %UInt.as.Copy.impl.Op.type.667 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %u2, (%Copy.impl_witness.2aa) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c29: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.d20: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c29, %Copy.facet [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.bound.28c: <bound method> = bound_method %int_0.39d, %UInt.as.Copy.impl.Op.56a [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %UInt.as.Copy.impl.Op.56a, @UInt.as.Copy.impl.Op(%int_2.ecc) [concrete]
// CHECK:STDOUT:   %bound_method.74f: <bound method> = bound_method %int_0.39d, %UInt.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Ordering.val.e12: %Ordering = struct_value (%int_0.39d) [concrete]
// CHECK:STDOUT:   %.5a1: ref %Ordering = temporary invalid, %Ordering.val.e12 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.400: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.a36 [concrete]
// CHECK:STDOUT:   %bound_method.ddb: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0ef: %u2 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct.7b9: %struct_type.discriminant = struct_value (%int_1.0ef) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.bound.9b0: <bound method> = bound_method %int_1.0ef, %UInt.as.Copy.impl.Op.56a [concrete]
// CHECK:STDOUT:   %bound_method.f5e: <bound method> = bound_method %int_1.0ef, %UInt.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Ordering.val.b30: %Ordering = struct_value (%int_1.0ef) [concrete]
// CHECK:STDOUT:   %.f25: ref %Ordering = temporary invalid, %Ordering.val.b30 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.de2: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.a36 [concrete]
// CHECK:STDOUT:   %bound_method.70b: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.2a0: %u2 = int_value 2 [concrete]
// CHECK:STDOUT:   %struct.fcb: %struct_type.discriminant = struct_value (%int_2.2a0) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.bound.7d6: <bound method> = bound_method %int_2.2a0, %UInt.as.Copy.impl.Op.56a [concrete]
// CHECK:STDOUT:   %bound_method.dec: <bound method> = bound_method %int_2.2a0, %UInt.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Ordering.val.538: %Ordering = struct_value (%int_2.2a0) [concrete]
// CHECK:STDOUT:   %.e53: ref %Ordering = temporary invalid, %Ordering.val.538 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ff7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.a36 [concrete]
// CHECK:STDOUT:   %bound_method.a47: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.e3d: %u2 = int_value 3 [concrete]
// CHECK:STDOUT:   %struct.0f8: %struct_type.discriminant = struct_value (%int_3.e3d) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.bound.846: <bound method> = bound_method %int_3.e3d, %UInt.as.Copy.impl.Op.56a [concrete]
// CHECK:STDOUT:   %bound_method.684: <bound method> = bound_method %int_3.e3d, %UInt.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Ordering.val.e37: %Ordering = struct_value (%int_3.e3d) [concrete]
// CHECK:STDOUT:   %.c4a: ref %Ordering = temporary invalid, %Ordering.val.e37 [concrete]
// CHECK:STDOUT:   %pattern_type.2fa: type = pattern_type %Ordering [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.f04: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.be8) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.929)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b5d = impl_witness_table (%Core.import_ref.f04), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.999: @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op.type (%UInt.as.Copy.impl.Op.type.2ce) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op (constants.%UInt.as.Copy.impl.Op.65a)]
// CHECK:STDOUT:   %Copy.impl_witness_table.744 = impl_witness_table (%Core.import_ref.999), @UInt.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Ordering.decl: type = class_decl @Ordering [concrete = constants.%Ordering] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less.patt: %pattern_type.2fa = value_binding_pattern less [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ordering.ref.loc11: type = name_ref Ordering, %Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %less: %Ordering = value_binding less, @__global_init.%Less.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %equiv.patt: %pattern_type.2fa = value_binding_pattern equiv [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ordering.ref.loc12: type = name_ref Ordering, %Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %equiv: %Ordering = value_binding equiv, @__global_init.%Equivalent.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater.patt: %pattern_type.2fa = value_binding_pattern greater [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ordering.ref.loc13: type = name_ref Ordering, %Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %greater: %Ordering = value_binding greater, @__global_init.%Greater.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %inc.patt: %pattern_type.2fa = value_binding_pattern inc [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ordering.ref.loc14: type = name_ref Ordering, %Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %inc: %Ordering = value_binding inc, @__global_init.%Incomparable.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Ordering {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.discriminant [concrete = constants.%complete_type.5ef]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc5_7.1: %.e52 = impl_witness_access constants.%ImplicitAs.impl_witness.90c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a36]
// CHECK:STDOUT:   %bound_method.loc5_7.1: <bound method> = bound_method %int_0, %impl.elem0.loc5_7.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0d4]
// CHECK:STDOUT:   %specific_fn.loc5_7.1: <specific function> = specific_function %impl.elem0.loc5_7.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_2.ecc) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_7.2: <bound method> = bound_method %int_0, %specific_fn.loc5_7.1 [concrete = constants.%bound_method.c7c]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5: init %u2 = call %bound_method.loc5_7.2(%int_0) [concrete = constants.%int_0.39d]
// CHECK:STDOUT:   %.loc5_7.1: %u2 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5 [concrete = constants.%int_0.39d]
// CHECK:STDOUT:   %.loc5_7.2: %u2 = converted %int_0, %.loc5_7.1 [concrete = constants.%int_0.39d]
// CHECK:STDOUT:   %.loc5_7.3: %struct_type.discriminant = struct_literal (%.loc5_7.2) [concrete = constants.%struct.373]
// CHECK:STDOUT:   %impl.elem0.loc5_7.2: %.d20 = impl_witness_access constants.%Copy.impl_witness.2aa, element0 [concrete = constants.%UInt.as.Copy.impl.Op.56a]
// CHECK:STDOUT:   %bound_method.loc5_7.3: <bound method> = bound_method %.loc5_7.2, %impl.elem0.loc5_7.2 [concrete = constants.%UInt.as.Copy.impl.Op.bound.28c]
// CHECK:STDOUT:   %specific_fn.loc5_7.2: <specific function> = specific_function %impl.elem0.loc5_7.2, @UInt.as.Copy.impl.Op(constants.%int_2.ecc) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_7.4: <bound method> = bound_method %.loc5_7.2, %specific_fn.loc5_7.2 [concrete = constants.%bound_method.74f]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc5: init %u2 = call %bound_method.loc5_7.4(%.loc5_7.2) [concrete = constants.%int_0.39d]
// CHECK:STDOUT:   %.loc5_7.4: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc5_7.5: ref %u2 = class_element_access %.loc5_7.4, element0
// CHECK:STDOUT:   %.loc5_7.6: init %u2 to %.loc5_7.5 = in_place_init %UInt.as.Copy.impl.Op.call.loc5 [concrete = constants.%int_0.39d]
// CHECK:STDOUT:   %.loc5_7.7: init %Ordering to %.loc5_7.4 = class_init (%.loc5_7.6) [concrete = constants.%Ordering.val.e12]
// CHECK:STDOUT:   %.loc5_7.8: init %Ordering = converted %.loc5_7.3, %.loc5_7.7 [concrete = constants.%Ordering.val.e12]
// CHECK:STDOUT:   %.loc5_7.9: ref %Ordering = temporary %.loc5_7.4, %.loc5_7.8 [concrete = constants.%.5a1]
// CHECK:STDOUT:   %.loc5_7.10: %Ordering = acquire_value %.loc5_7.9 [concrete = constants.%Ordering.val.e12]
// CHECK:STDOUT:   %Less: %Ordering = value_binding Less, %.loc5_7.10
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc6_13.1: %.e52 = impl_witness_access constants.%ImplicitAs.impl_witness.90c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a36]
// CHECK:STDOUT:   %bound_method.loc6_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc6_13.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.400]
// CHECK:STDOUT:   %specific_fn.loc6_13.1: <specific function> = specific_function %impl.elem0.loc6_13.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_2.ecc) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_13.2: <bound method> = bound_method %int_1, %specific_fn.loc6_13.1 [concrete = constants.%bound_method.ddb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %u2 = call %bound_method.loc6_13.2(%int_1) [concrete = constants.%int_1.0ef]
// CHECK:STDOUT:   %.loc6_13.1: %u2 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%int_1.0ef]
// CHECK:STDOUT:   %.loc6_13.2: %u2 = converted %int_1, %.loc6_13.1 [concrete = constants.%int_1.0ef]
// CHECK:STDOUT:   %.loc6_13.3: %struct_type.discriminant = struct_literal (%.loc6_13.2) [concrete = constants.%struct.7b9]
// CHECK:STDOUT:   %impl.elem0.loc6_13.2: %.d20 = impl_witness_access constants.%Copy.impl_witness.2aa, element0 [concrete = constants.%UInt.as.Copy.impl.Op.56a]
// CHECK:STDOUT:   %bound_method.loc6_13.3: <bound method> = bound_method %.loc6_13.2, %impl.elem0.loc6_13.2 [concrete = constants.%UInt.as.Copy.impl.Op.bound.9b0]
// CHECK:STDOUT:   %specific_fn.loc6_13.2: <specific function> = specific_function %impl.elem0.loc6_13.2, @UInt.as.Copy.impl.Op(constants.%int_2.ecc) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_13.4: <bound method> = bound_method %.loc6_13.2, %specific_fn.loc6_13.2 [concrete = constants.%bound_method.f5e]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc6: init %u2 = call %bound_method.loc6_13.4(%.loc6_13.2) [concrete = constants.%int_1.0ef]
// CHECK:STDOUT:   %.loc6_13.4: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc6_13.5: ref %u2 = class_element_access %.loc6_13.4, element0
// CHECK:STDOUT:   %.loc6_13.6: init %u2 to %.loc6_13.5 = in_place_init %UInt.as.Copy.impl.Op.call.loc6 [concrete = constants.%int_1.0ef]
// CHECK:STDOUT:   %.loc6_13.7: init %Ordering to %.loc6_13.4 = class_init (%.loc6_13.6) [concrete = constants.%Ordering.val.b30]
// CHECK:STDOUT:   %.loc6_13.8: init %Ordering = converted %.loc6_13.3, %.loc6_13.7 [concrete = constants.%Ordering.val.b30]
// CHECK:STDOUT:   %.loc6_13.9: ref %Ordering = temporary %.loc6_13.4, %.loc6_13.8 [concrete = constants.%.f25]
// CHECK:STDOUT:   %.loc6_13.10: %Ordering = acquire_value %.loc6_13.9 [concrete = constants.%Ordering.val.b30]
// CHECK:STDOUT:   %Equivalent: %Ordering = value_binding Equivalent, %.loc6_13.10
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc7_10.1: %.e52 = impl_witness_access constants.%ImplicitAs.impl_witness.90c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a36]
// CHECK:STDOUT:   %bound_method.loc7_10.1: <bound method> = bound_method %int_2, %impl.elem0.loc7_10.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.de2]
// CHECK:STDOUT:   %specific_fn.loc7_10.1: <specific function> = specific_function %impl.elem0.loc7_10.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_2.ecc) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_10.2: <bound method> = bound_method %int_2, %specific_fn.loc7_10.1 [concrete = constants.%bound_method.70b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %u2 = call %bound_method.loc7_10.2(%int_2) [concrete = constants.%int_2.2a0]
// CHECK:STDOUT:   %.loc7_10.1: %u2 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%int_2.2a0]
// CHECK:STDOUT:   %.loc7_10.2: %u2 = converted %int_2, %.loc7_10.1 [concrete = constants.%int_2.2a0]
// CHECK:STDOUT:   %.loc7_10.3: %struct_type.discriminant = struct_literal (%.loc7_10.2) [concrete = constants.%struct.fcb]
// CHECK:STDOUT:   %impl.elem0.loc7_10.2: %.d20 = impl_witness_access constants.%Copy.impl_witness.2aa, element0 [concrete = constants.%UInt.as.Copy.impl.Op.56a]
// CHECK:STDOUT:   %bound_method.loc7_10.3: <bound method> = bound_method %.loc7_10.2, %impl.elem0.loc7_10.2 [concrete = constants.%UInt.as.Copy.impl.Op.bound.7d6]
// CHECK:STDOUT:   %specific_fn.loc7_10.2: <specific function> = specific_function %impl.elem0.loc7_10.2, @UInt.as.Copy.impl.Op(constants.%int_2.ecc) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_10.4: <bound method> = bound_method %.loc7_10.2, %specific_fn.loc7_10.2 [concrete = constants.%bound_method.dec]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc7: init %u2 = call %bound_method.loc7_10.4(%.loc7_10.2) [concrete = constants.%int_2.2a0]
// CHECK:STDOUT:   %.loc7_10.4: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc7_10.5: ref %u2 = class_element_access %.loc7_10.4, element0
// CHECK:STDOUT:   %.loc7_10.6: init %u2 to %.loc7_10.5 = in_place_init %UInt.as.Copy.impl.Op.call.loc7 [concrete = constants.%int_2.2a0]
// CHECK:STDOUT:   %.loc7_10.7: init %Ordering to %.loc7_10.4 = class_init (%.loc7_10.6) [concrete = constants.%Ordering.val.538]
// CHECK:STDOUT:   %.loc7_10.8: init %Ordering = converted %.loc7_10.3, %.loc7_10.7 [concrete = constants.%Ordering.val.538]
// CHECK:STDOUT:   %.loc7_10.9: ref %Ordering = temporary %.loc7_10.4, %.loc7_10.8 [concrete = constants.%.e53]
// CHECK:STDOUT:   %.loc7_10.10: %Ordering = acquire_value %.loc7_10.9 [concrete = constants.%Ordering.val.538]
// CHECK:STDOUT:   %Greater: %Ordering = value_binding Greater, %.loc7_10.10
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc9_1.1: %.e52 = impl_witness_access constants.%ImplicitAs.impl_witness.90c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.a36]
// CHECK:STDOUT:   %bound_method.loc9_1.1: <bound method> = bound_method %int_3, %impl.elem0.loc9_1.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ff7]
// CHECK:STDOUT:   %specific_fn.loc9_1.1: <specific function> = specific_function %impl.elem0.loc9_1.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_2.ecc) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_1.2: <bound method> = bound_method %int_3, %specific_fn.loc9_1.1 [concrete = constants.%bound_method.a47]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %u2 = call %bound_method.loc9_1.2(%int_3) [concrete = constants.%int_3.e3d]
// CHECK:STDOUT:   %.loc9_1.1: %u2 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_3.e3d]
// CHECK:STDOUT:   %.loc9_1.2: %u2 = converted %int_3, %.loc9_1.1 [concrete = constants.%int_3.e3d]
// CHECK:STDOUT:   %.loc9_1.3: %struct_type.discriminant = struct_literal (%.loc9_1.2) [concrete = constants.%struct.0f8]
// CHECK:STDOUT:   %impl.elem0.loc9_1.2: %.d20 = impl_witness_access constants.%Copy.impl_witness.2aa, element0 [concrete = constants.%UInt.as.Copy.impl.Op.56a]
// CHECK:STDOUT:   %bound_method.loc9_1.3: <bound method> = bound_method %.loc9_1.2, %impl.elem0.loc9_1.2 [concrete = constants.%UInt.as.Copy.impl.Op.bound.846]
// CHECK:STDOUT:   %specific_fn.loc9_1.2: <specific function> = specific_function %impl.elem0.loc9_1.2, @UInt.as.Copy.impl.Op(constants.%int_2.ecc) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_1.4: <bound method> = bound_method %.loc9_1.2, %specific_fn.loc9_1.2 [concrete = constants.%bound_method.684]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc9: init %u2 = call %bound_method.loc9_1.4(%.loc9_1.2) [concrete = constants.%int_3.e3d]
// CHECK:STDOUT:   %.loc9_1.4: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc9_1.5: ref %u2 = class_element_access %.loc9_1.4, element0
// CHECK:STDOUT:   %.loc9_1.6: init %u2 to %.loc9_1.5 = in_place_init %UInt.as.Copy.impl.Op.call.loc9 [concrete = constants.%int_3.e3d]
// CHECK:STDOUT:   %.loc9_1.7: init %Ordering to %.loc9_1.4 = class_init (%.loc9_1.6) [concrete = constants.%Ordering.val.e37]
// CHECK:STDOUT:   %.loc9_1.8: init %Ordering = converted %.loc9_1.3, %.loc9_1.7 [concrete = constants.%Ordering.val.e37]
// CHECK:STDOUT:   %.loc9_1.9: ref %Ordering = temporary %.loc9_1.4, %.loc9_1.8 [concrete = constants.%.c4a]
// CHECK:STDOUT:   %.loc9_1.10: %Ordering = acquire_value %.loc9_1.9 [concrete = constants.%Ordering.val.e37]
// CHECK:STDOUT:   %Incomparable: %Ordering = value_binding Incomparable, %.loc9_1.10
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
// CHECK:STDOUT:   %Ordering.ref.loc11: type = name_ref Ordering, file.%Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %Less.ref: %Ordering = name_ref Less, @Ordering.%Less
// CHECK:STDOUT:   %Ordering.ref.loc12: type = name_ref Ordering, file.%Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %Equivalent.ref: %Ordering = name_ref Equivalent, @Ordering.%Equivalent
// CHECK:STDOUT:   %Ordering.ref.loc13: type = name_ref Ordering, file.%Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %Greater.ref: %Ordering = name_ref Greater, @Ordering.%Greater
// CHECK:STDOUT:   %Ordering.ref.loc14: type = name_ref Ordering, file.%Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %Incomparable.ref: %Ordering = name_ref Incomparable, @Ordering.%Incomparable
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
