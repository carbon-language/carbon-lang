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
// CHECK:STDOUT: imports {
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
// CHECK:STDOUT:   %Always.val: %Always = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Always [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Always.decl: type = class_decl @Always [concrete = constants.%Always] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %mood.patt: %pattern_type = binding_pattern mood [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Always.ref: type = name_ref Always, %Always.decl [concrete = constants.%Always]
// CHECK:STDOUT:   %mood: %Always = bind_name mood, @__global_init.%Sunny.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Always {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.discriminant [concrete = constants.%complete_type]
// CHECK:STDOUT:   %.loc6_1.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_1.2: %empty_tuple.type = converted %.loc6_1.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_1.3: %struct_type.discriminant = struct_literal (%.loc6_1.2)
// CHECK:STDOUT:   %.loc6_1.4: ref %Always = temporary_storage
// CHECK:STDOUT:   %.loc6_1.5: ref %empty_tuple.type = class_element_access %.loc6_1.4, element0
// CHECK:STDOUT:   %.loc6_1.6: init %empty_tuple.type = tuple_init () to %.loc6_1.5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_1.7: init %empty_tuple.type = converted %.loc6_1.2, %.loc6_1.6 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_1.8: init %Always = class_init (%.loc6_1.7), %.loc6_1.4 [concrete = constants.%Always.val]
// CHECK:STDOUT:   %.loc6_1.9: ref %Always = temporary %.loc6_1.4, %.loc6_1.8
// CHECK:STDOUT:   %.loc6_1.10: ref %Always = converted %.loc6_1.3, %.loc6_1.9
// CHECK:STDOUT:   %.loc6_1.11: %Always = bind_value %.loc6_1.10
// CHECK:STDOUT:   %Sunny: %Always = bind_name Sunny, %.loc6_1.11
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
// CHECK:STDOUT:   %u2: type = class_type @UInt, @UInt(%int_2.ecc) [concrete]
// CHECK:STDOUT:   %struct_type.discriminant: type = struct_type {.discriminant: %u2} [concrete]
// CHECK:STDOUT:   %complete_type.de2: <witness> = complete_type_witness %struct_type.discriminant [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a92: type = facet_type <@ImplicitAs, @ImplicitAs(%u2)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.f0e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%u2) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.367: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8f6: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.367 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.b0d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b63, @Core.IntLiteral.as.ImplicitAs.impl(%int_2.ecc) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.534: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_2.ecc) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.534 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a92 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.b0d) [concrete]
// CHECK:STDOUT:   %.a66: type = fn_type_with_self_type %ImplicitAs.Convert.type.f0e, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.289: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_2.ecc) [concrete]
// CHECK:STDOUT:   %bound_method.7b7: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.9fd: %u2 = int_value 0 [concrete]
// CHECK:STDOUT:   %Ordering.val.a29: %Ordering = struct_value (%int_0.9fd) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.fd2: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b2 [concrete]
// CHECK:STDOUT:   %bound_method.e28: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.b2c: %u2 = int_value 1 [concrete]
// CHECK:STDOUT:   %Ordering.val.927: %Ordering = struct_value (%int_1.b2c) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.53a: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b2 [concrete]
// CHECK:STDOUT:   %bound_method.aa0: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.788: %u2 = int_value 2 [concrete]
// CHECK:STDOUT:   %Ordering.val.968: %Ordering = struct_value (%int_2.788) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f9: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b2 [concrete]
// CHECK:STDOUT:   %bound_method.b35: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.975: %u2 = int_value 3 [concrete]
// CHECK:STDOUT:   %Ordering.val.8a7: %Ordering = struct_value (%int_3.975) [concrete]
// CHECK:STDOUT:   %pattern_type.308: type = pattern_type %Ordering [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.104: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.367) = import_ref Core//prelude/parts/uint, loc16_40, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8f6)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b63 = impl_witness_table (%Core.import_ref.104), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Ordering.decl: type = class_decl @Ordering [concrete = constants.%Ordering] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less.patt: %pattern_type.308 = binding_pattern less [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ordering.ref.loc11: type = name_ref Ordering, %Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %less: %Ordering = bind_name less, @__global_init.%Less.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %equiv.patt: %pattern_type.308 = binding_pattern equiv [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ordering.ref.loc12: type = name_ref Ordering, %Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %equiv: %Ordering = bind_name equiv, @__global_init.%Equivalent.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater.patt: %pattern_type.308 = binding_pattern greater [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ordering.ref.loc13: type = name_ref Ordering, %Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %greater: %Ordering = bind_name greater, @__global_init.%Greater.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %inc.patt: %pattern_type.308 = binding_pattern inc [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ordering.ref.loc14: type = name_ref Ordering, %Ordering.decl [concrete = constants.%Ordering]
// CHECK:STDOUT:   %inc: %Ordering = bind_name inc, @__global_init.%Incomparable.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Ordering {
// CHECK:STDOUT:   %int_2.loc9: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %u2: type = class_type @UInt, @UInt(constants.%int_2.ecc) [concrete = constants.%u2]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.discriminant [concrete = constants.%complete_type.de2]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc5: %.a66 = impl_witness_access constants.%ImplicitAs.impl_witness.b0d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b2]
// CHECK:STDOUT:   %bound_method.loc5_7.1: <bound method> = bound_method %int_0, %impl.elem0.loc5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.289]
// CHECK:STDOUT:   %specific_fn.loc5: <specific function> = specific_function %impl.elem0.loc5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_2.ecc) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_7.2: <bound method> = bound_method %int_0, %specific_fn.loc5 [concrete = constants.%bound_method.7b7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5: init %u2 = call %bound_method.loc5_7.2(%int_0) [concrete = constants.%int_0.9fd]
// CHECK:STDOUT:   %.loc5_7.1: %u2 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5 [concrete = constants.%int_0.9fd]
// CHECK:STDOUT:   %.loc5_7.2: %u2 = converted %int_0, %.loc5_7.1 [concrete = constants.%int_0.9fd]
// CHECK:STDOUT:   %.loc5_7.3: %struct_type.discriminant = struct_literal (%.loc5_7.2)
// CHECK:STDOUT:   %.loc5_7.4: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc5_7.5: ref %u2 = class_element_access %.loc5_7.4, element0
// CHECK:STDOUT:   %.loc5_7.6: init %u2 = initialize_from %.loc5_7.2 to %.loc5_7.5 [concrete = constants.%int_0.9fd]
// CHECK:STDOUT:   %.loc5_7.7: init %Ordering = class_init (%.loc5_7.6), %.loc5_7.4 [concrete = constants.%Ordering.val.a29]
// CHECK:STDOUT:   %.loc5_7.8: ref %Ordering = temporary %.loc5_7.4, %.loc5_7.7
// CHECK:STDOUT:   %.loc5_7.9: ref %Ordering = converted %.loc5_7.3, %.loc5_7.8
// CHECK:STDOUT:   %.loc5_7.10: %Ordering = bind_value %.loc5_7.9
// CHECK:STDOUT:   %Less: %Ordering = bind_name Less, %.loc5_7.10
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc6: %.a66 = impl_witness_access constants.%ImplicitAs.impl_witness.b0d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b2]
// CHECK:STDOUT:   %bound_method.loc6_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc6 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.fd2]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_2.ecc) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_13.2: <bound method> = bound_method %int_1, %specific_fn.loc6 [concrete = constants.%bound_method.e28]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %u2 = call %bound_method.loc6_13.2(%int_1) [concrete = constants.%int_1.b2c]
// CHECK:STDOUT:   %.loc6_13.1: %u2 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%int_1.b2c]
// CHECK:STDOUT:   %.loc6_13.2: %u2 = converted %int_1, %.loc6_13.1 [concrete = constants.%int_1.b2c]
// CHECK:STDOUT:   %.loc6_13.3: %struct_type.discriminant = struct_literal (%.loc6_13.2)
// CHECK:STDOUT:   %.loc6_13.4: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc6_13.5: ref %u2 = class_element_access %.loc6_13.4, element0
// CHECK:STDOUT:   %.loc6_13.6: init %u2 = initialize_from %.loc6_13.2 to %.loc6_13.5 [concrete = constants.%int_1.b2c]
// CHECK:STDOUT:   %.loc6_13.7: init %Ordering = class_init (%.loc6_13.6), %.loc6_13.4 [concrete = constants.%Ordering.val.927]
// CHECK:STDOUT:   %.loc6_13.8: ref %Ordering = temporary %.loc6_13.4, %.loc6_13.7
// CHECK:STDOUT:   %.loc6_13.9: ref %Ordering = converted %.loc6_13.3, %.loc6_13.8
// CHECK:STDOUT:   %.loc6_13.10: %Ordering = bind_value %.loc6_13.9
// CHECK:STDOUT:   %Equivalent: %Ordering = bind_name Equivalent, %.loc6_13.10
// CHECK:STDOUT:   %int_2.loc7: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc7: %.a66 = impl_witness_access constants.%ImplicitAs.impl_witness.b0d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b2]
// CHECK:STDOUT:   %bound_method.loc7_10.1: <bound method> = bound_method %int_2.loc7, %impl.elem0.loc7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.53a]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_2.ecc) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_10.2: <bound method> = bound_method %int_2.loc7, %specific_fn.loc7 [concrete = constants.%bound_method.aa0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %u2 = call %bound_method.loc7_10.2(%int_2.loc7) [concrete = constants.%int_2.788]
// CHECK:STDOUT:   %.loc7_10.1: %u2 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%int_2.788]
// CHECK:STDOUT:   %.loc7_10.2: %u2 = converted %int_2.loc7, %.loc7_10.1 [concrete = constants.%int_2.788]
// CHECK:STDOUT:   %.loc7_10.3: %struct_type.discriminant = struct_literal (%.loc7_10.2)
// CHECK:STDOUT:   %.loc7_10.4: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc7_10.5: ref %u2 = class_element_access %.loc7_10.4, element0
// CHECK:STDOUT:   %.loc7_10.6: init %u2 = initialize_from %.loc7_10.2 to %.loc7_10.5 [concrete = constants.%int_2.788]
// CHECK:STDOUT:   %.loc7_10.7: init %Ordering = class_init (%.loc7_10.6), %.loc7_10.4 [concrete = constants.%Ordering.val.968]
// CHECK:STDOUT:   %.loc7_10.8: ref %Ordering = temporary %.loc7_10.4, %.loc7_10.7
// CHECK:STDOUT:   %.loc7_10.9: ref %Ordering = converted %.loc7_10.3, %.loc7_10.8
// CHECK:STDOUT:   %.loc7_10.10: %Ordering = bind_value %.loc7_10.9
// CHECK:STDOUT:   %Greater: %Ordering = bind_name Greater, %.loc7_10.10
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc9: %.a66 = impl_witness_access constants.%ImplicitAs.impl_witness.b0d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b2]
// CHECK:STDOUT:   %bound_method.loc9_1.1: <bound method> = bound_method %int_3, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f9]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_2.ecc) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_1.2: <bound method> = bound_method %int_3, %specific_fn.loc9 [concrete = constants.%bound_method.b35]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %u2 = call %bound_method.loc9_1.2(%int_3) [concrete = constants.%int_3.975]
// CHECK:STDOUT:   %.loc9_1.1: %u2 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_3.975]
// CHECK:STDOUT:   %.loc9_1.2: %u2 = converted %int_3, %.loc9_1.1 [concrete = constants.%int_3.975]
// CHECK:STDOUT:   %.loc9_1.3: %struct_type.discriminant = struct_literal (%.loc9_1.2)
// CHECK:STDOUT:   %.loc9_1.4: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc9_1.5: ref %u2 = class_element_access %.loc9_1.4, element0
// CHECK:STDOUT:   %.loc9_1.6: init %u2 = initialize_from %.loc9_1.2 to %.loc9_1.5 [concrete = constants.%int_3.975]
// CHECK:STDOUT:   %.loc9_1.7: init %Ordering = class_init (%.loc9_1.6), %.loc9_1.4 [concrete = constants.%Ordering.val.8a7]
// CHECK:STDOUT:   %.loc9_1.8: ref %Ordering = temporary %.loc9_1.4, %.loc9_1.7
// CHECK:STDOUT:   %.loc9_1.9: ref %Ordering = converted %.loc9_1.3, %.loc9_1.8
// CHECK:STDOUT:   %.loc9_1.10: %Ordering = bind_value %.loc9_1.9
// CHECK:STDOUT:   %Incomparable: %Ordering = bind_name Incomparable, %.loc9_1.10
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
