// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/choice/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/choice/basic.carbon

// --- no_alternative.carbon
library "[[@TEST_NAME]]";

choice Never {}

// --- one_alternative.carbon
library "[[@TEST_NAME]]";

choice Always {
  Sunny
}

fn G() {
  let mood: Always = Always.Sunny;
}

// --- multiple_alternatives.carbon
library "[[@TEST_NAME]]";

choice Ordering {
  Less,
  Equivalent,
  Greater,
  Incomparable
}

fn H() {
  let less: Ordering = Ordering.Less;
  let equiv: Ordering = Ordering.Equivalent;
  let greater: Ordering = Ordering.Greater;
  let inc: Ordering = Ordering.Incomparable;
}

// CHECK:STDOUT: --- no_alternative.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Never: type = class_type @Never [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.discriminant: type = struct_type {.discriminant: %empty_tuple.type} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.discriminant [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Never = %Never.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Never.decl: type = class_decl @Never [template = constants.%Never] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Never {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.discriminant [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- one_alternative.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Always: type = class_type @Always [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.discriminant: type = struct_type {.discriminant: %empty_tuple.type} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.discriminant [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %Always.val: %Always = struct_value (%empty_tuple) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Always = %Always.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Always.decl: type = class_decl @Always [template = constants.%Always] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Always {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.discriminant [template = constants.%complete_type]
// CHECK:STDOUT:   %.loc5_1.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_1.2: %empty_tuple.type = converted %.loc5_1.1, %empty_tuple [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_1.3: %struct_type.discriminant = struct_literal (%.loc5_1.2)
// CHECK:STDOUT:   %.loc5_1.4: ref %Always = temporary_storage
// CHECK:STDOUT:   %.loc5_1.5: ref %empty_tuple.type = class_element_access %.loc5_1.4, element0
// CHECK:STDOUT:   %.loc5_1.6: init %empty_tuple.type = tuple_init () to %.loc5_1.5 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_1.7: init %empty_tuple.type = converted %.loc5_1.2, %.loc5_1.6 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_1.8: init %Always = class_init (%.loc5_1.7), %.loc5_1.4 [template = constants.%Always.val]
// CHECK:STDOUT:   %.loc5_1.9: ref %Always = temporary %.loc5_1.4, %.loc5_1.8
// CHECK:STDOUT:   %.loc5_1.10: ref %Always = converted %.loc5_1.3, %.loc5_1.9
// CHECK:STDOUT:   %.loc5_1.11: %Always = bind_value %.loc5_1.10
// CHECK:STDOUT:   %Sunny: %Always = bind_name Sunny, %.loc5_1.11
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %mood.patt: %Always = binding_pattern mood
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Always.ref.loc8_22: type = name_ref Always, file.%Always.decl [template = constants.%Always]
// CHECK:STDOUT:   %Sunny.ref: %Always = name_ref Sunny, @Always.%Sunny
// CHECK:STDOUT:   %Always.ref.loc8_13: type = name_ref Always, file.%Always.decl [template = constants.%Always]
// CHECK:STDOUT:   %mood: %Always = bind_name mood, %Sunny.ref
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- multiple_alternatives.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ordering: type = class_type @Ordering [template]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %u2: type = class_type @UInt, @UInt(%int_2.ecc) [template]
// CHECK:STDOUT:   %struct_type.discriminant: type = struct_type {.discriminant: %u2} [template]
// CHECK:STDOUT:   %complete_type.de2: <witness> = complete_type_witness %struct_type.discriminant [template]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.f0e: type = fn_type @Convert.1, @ImplicitAs(%u2) [template]
// CHECK:STDOUT:   %impl_witness.f5e: <witness> = impl_witness (imports.%import_ref.c3d), @impl.1(%int_2.ecc) [template]
// CHECK:STDOUT:   %Convert.type.70b: type = fn_type @Convert.2, @impl.1(%int_2.ecc) [template]
// CHECK:STDOUT:   %Convert.474: %Convert.type.70b = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.5bb: <bound method> = bound_method %int_0.5c6, %Convert.474 [template]
// CHECK:STDOUT:   %Convert.specific_fn.dd9: <specific function> = specific_function %Convert.bound.5bb, @Convert.2(%int_2.ecc) [template]
// CHECK:STDOUT:   %int_0.9fd: %u2 = int_value 0 [template]
// CHECK:STDOUT:   %Ordering.val.a29: %Ordering = struct_value (%int_0.9fd) [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.0dd: <bound method> = bound_method %int_1.5b8, %Convert.474 [template]
// CHECK:STDOUT:   %Convert.specific_fn.95d: <specific function> = specific_function %Convert.bound.0dd, @Convert.2(%int_2.ecc) [template]
// CHECK:STDOUT:   %int_1.b2c: %u2 = int_value 1 [template]
// CHECK:STDOUT:   %Ordering.val.927: %Ordering = struct_value (%int_1.b2c) [template]
// CHECK:STDOUT:   %Convert.bound.122: <bound method> = bound_method %int_2.ecc, %Convert.474 [template]
// CHECK:STDOUT:   %Convert.specific_fn.190: <specific function> = specific_function %Convert.bound.122, @Convert.2(%int_2.ecc) [template]
// CHECK:STDOUT:   %int_2.788: %u2 = int_value 2 [template]
// CHECK:STDOUT:   %Ordering.val.968: %Ordering = struct_value (%int_2.788) [template]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Convert.bound.a5c: <bound method> = bound_method %int_3.1ba, %Convert.474 [template]
// CHECK:STDOUT:   %Convert.specific_fn.9bc: <specific function> = specific_function %Convert.bound.a5c, @Convert.2(%int_2.ecc) [template]
// CHECK:STDOUT:   %int_3.975: %u2 = int_value 3 [template]
// CHECK:STDOUT:   %Ordering.val.8a7: %Ordering = struct_value (%int_3.975) [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.bcd
// CHECK:STDOUT:     .ImplicitAs = %import_ref.d44
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Ordering = %Ordering.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Ordering.decl: type = class_decl @Ordering [template = constants.%Ordering] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Ordering {
// CHECK:STDOUT:   %int_2.loc8: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:   %u2: type = class_type @UInt, @UInt(constants.%int_2.ecc) [template = constants.%u2]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.discriminant [template = constants.%complete_type.de2]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc4: %Convert.type.f0e = impl_witness_access constants.%impl_witness.f5e, element0 [template = constants.%Convert.474]
// CHECK:STDOUT:   %Convert.bound.loc4: <bound method> = bound_method %int_0, %impl.elem0.loc4 [template = constants.%Convert.bound.5bb]
// CHECK:STDOUT:   %Convert.specific_fn.loc4: <specific function> = specific_function %Convert.bound.loc4, @Convert.2(constants.%int_2.ecc) [template = constants.%Convert.specific_fn.dd9]
// CHECK:STDOUT:   %int.convert_checked.loc4: init %u2 = call %Convert.specific_fn.loc4(%int_0) [template = constants.%int_0.9fd]
// CHECK:STDOUT:   %.loc4_7.1: %u2 = value_of_initializer %int.convert_checked.loc4 [template = constants.%int_0.9fd]
// CHECK:STDOUT:   %.loc4_7.2: %u2 = converted %int_0, %.loc4_7.1 [template = constants.%int_0.9fd]
// CHECK:STDOUT:   %.loc4_7.3: %struct_type.discriminant = struct_literal (%.loc4_7.2)
// CHECK:STDOUT:   %.loc4_7.4: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc4_7.5: ref %u2 = class_element_access %.loc4_7.4, element0
// CHECK:STDOUT:   %.loc4_7.6: init %u2 = initialize_from %.loc4_7.2 to %.loc4_7.5 [template = constants.%int_0.9fd]
// CHECK:STDOUT:   %.loc4_7.7: init %Ordering = class_init (%.loc4_7.6), %.loc4_7.4 [template = constants.%Ordering.val.a29]
// CHECK:STDOUT:   %.loc4_7.8: ref %Ordering = temporary %.loc4_7.4, %.loc4_7.7
// CHECK:STDOUT:   %.loc4_7.9: ref %Ordering = converted %.loc4_7.3, %.loc4_7.8
// CHECK:STDOUT:   %.loc4_7.10: %Ordering = bind_value %.loc4_7.9
// CHECK:STDOUT:   %Less: %Ordering = bind_name Less, %.loc4_7.10
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc5: %Convert.type.f0e = impl_witness_access constants.%impl_witness.f5e, element0 [template = constants.%Convert.474]
// CHECK:STDOUT:   %Convert.bound.loc5: <bound method> = bound_method %int_1, %impl.elem0.loc5 [template = constants.%Convert.bound.0dd]
// CHECK:STDOUT:   %Convert.specific_fn.loc5: <specific function> = specific_function %Convert.bound.loc5, @Convert.2(constants.%int_2.ecc) [template = constants.%Convert.specific_fn.95d]
// CHECK:STDOUT:   %int.convert_checked.loc5: init %u2 = call %Convert.specific_fn.loc5(%int_1) [template = constants.%int_1.b2c]
// CHECK:STDOUT:   %.loc5_13.1: %u2 = value_of_initializer %int.convert_checked.loc5 [template = constants.%int_1.b2c]
// CHECK:STDOUT:   %.loc5_13.2: %u2 = converted %int_1, %.loc5_13.1 [template = constants.%int_1.b2c]
// CHECK:STDOUT:   %.loc5_13.3: %struct_type.discriminant = struct_literal (%.loc5_13.2)
// CHECK:STDOUT:   %.loc5_13.4: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc5_13.5: ref %u2 = class_element_access %.loc5_13.4, element0
// CHECK:STDOUT:   %.loc5_13.6: init %u2 = initialize_from %.loc5_13.2 to %.loc5_13.5 [template = constants.%int_1.b2c]
// CHECK:STDOUT:   %.loc5_13.7: init %Ordering = class_init (%.loc5_13.6), %.loc5_13.4 [template = constants.%Ordering.val.927]
// CHECK:STDOUT:   %.loc5_13.8: ref %Ordering = temporary %.loc5_13.4, %.loc5_13.7
// CHECK:STDOUT:   %.loc5_13.9: ref %Ordering = converted %.loc5_13.3, %.loc5_13.8
// CHECK:STDOUT:   %.loc5_13.10: %Ordering = bind_value %.loc5_13.9
// CHECK:STDOUT:   %Equivalent: %Ordering = bind_name Equivalent, %.loc5_13.10
// CHECK:STDOUT:   %int_2.loc6: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc6: %Convert.type.f0e = impl_witness_access constants.%impl_witness.f5e, element0 [template = constants.%Convert.474]
// CHECK:STDOUT:   %Convert.bound.loc6: <bound method> = bound_method %int_2.loc6, %impl.elem0.loc6 [template = constants.%Convert.bound.122]
// CHECK:STDOUT:   %Convert.specific_fn.loc6: <specific function> = specific_function %Convert.bound.loc6, @Convert.2(constants.%int_2.ecc) [template = constants.%Convert.specific_fn.190]
// CHECK:STDOUT:   %int.convert_checked.loc6: init %u2 = call %Convert.specific_fn.loc6(%int_2.loc6) [template = constants.%int_2.788]
// CHECK:STDOUT:   %.loc6_10.1: %u2 = value_of_initializer %int.convert_checked.loc6 [template = constants.%int_2.788]
// CHECK:STDOUT:   %.loc6_10.2: %u2 = converted %int_2.loc6, %.loc6_10.1 [template = constants.%int_2.788]
// CHECK:STDOUT:   %.loc6_10.3: %struct_type.discriminant = struct_literal (%.loc6_10.2)
// CHECK:STDOUT:   %.loc6_10.4: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc6_10.5: ref %u2 = class_element_access %.loc6_10.4, element0
// CHECK:STDOUT:   %.loc6_10.6: init %u2 = initialize_from %.loc6_10.2 to %.loc6_10.5 [template = constants.%int_2.788]
// CHECK:STDOUT:   %.loc6_10.7: init %Ordering = class_init (%.loc6_10.6), %.loc6_10.4 [template = constants.%Ordering.val.968]
// CHECK:STDOUT:   %.loc6_10.8: ref %Ordering = temporary %.loc6_10.4, %.loc6_10.7
// CHECK:STDOUT:   %.loc6_10.9: ref %Ordering = converted %.loc6_10.3, %.loc6_10.8
// CHECK:STDOUT:   %.loc6_10.10: %Ordering = bind_value %.loc6_10.9
// CHECK:STDOUT:   %Greater: %Ordering = bind_name Greater, %.loc6_10.10
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc8: %Convert.type.f0e = impl_witness_access constants.%impl_witness.f5e, element0 [template = constants.%Convert.474]
// CHECK:STDOUT:   %Convert.bound.loc8: <bound method> = bound_method %int_3, %impl.elem0.loc8 [template = constants.%Convert.bound.a5c]
// CHECK:STDOUT:   %Convert.specific_fn.loc8: <specific function> = specific_function %Convert.bound.loc8, @Convert.2(constants.%int_2.ecc) [template = constants.%Convert.specific_fn.9bc]
// CHECK:STDOUT:   %int.convert_checked.loc8: init %u2 = call %Convert.specific_fn.loc8(%int_3) [template = constants.%int_3.975]
// CHECK:STDOUT:   %.loc8_1.1: %u2 = value_of_initializer %int.convert_checked.loc8 [template = constants.%int_3.975]
// CHECK:STDOUT:   %.loc8_1.2: %u2 = converted %int_3, %.loc8_1.1 [template = constants.%int_3.975]
// CHECK:STDOUT:   %.loc8_1.3: %struct_type.discriminant = struct_literal (%.loc8_1.2)
// CHECK:STDOUT:   %.loc8_1.4: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc8_1.5: ref %u2 = class_element_access %.loc8_1.4, element0
// CHECK:STDOUT:   %.loc8_1.6: init %u2 = initialize_from %.loc8_1.2 to %.loc8_1.5 [template = constants.%int_3.975]
// CHECK:STDOUT:   %.loc8_1.7: init %Ordering = class_init (%.loc8_1.6), %.loc8_1.4 [template = constants.%Ordering.val.8a7]
// CHECK:STDOUT:   %.loc8_1.8: ref %Ordering = temporary %.loc8_1.4, %.loc8_1.7
// CHECK:STDOUT:   %.loc8_1.9: ref %Ordering = converted %.loc8_1.3, %.loc8_1.8
// CHECK:STDOUT:   %.loc8_1.10: %Ordering = bind_value %.loc8_1.9
// CHECK:STDOUT:   %Incomparable: %Ordering = bind_name Incomparable, %.loc8_1.10
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less.patt: %Ordering = binding_pattern less
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ordering.ref.loc11_24: type = name_ref Ordering, file.%Ordering.decl [template = constants.%Ordering]
// CHECK:STDOUT:   %Less.ref: %Ordering = name_ref Less, @Ordering.%Less
// CHECK:STDOUT:   %Ordering.ref.loc11_13: type = name_ref Ordering, file.%Ordering.decl [template = constants.%Ordering]
// CHECK:STDOUT:   %less: %Ordering = bind_name less, %Less.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %equiv.patt: %Ordering = binding_pattern equiv
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ordering.ref.loc12_25: type = name_ref Ordering, file.%Ordering.decl [template = constants.%Ordering]
// CHECK:STDOUT:   %Equivalent.ref: %Ordering = name_ref Equivalent, @Ordering.%Equivalent
// CHECK:STDOUT:   %Ordering.ref.loc12_14: type = name_ref Ordering, file.%Ordering.decl [template = constants.%Ordering]
// CHECK:STDOUT:   %equiv: %Ordering = bind_name equiv, %Equivalent.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater.patt: %Ordering = binding_pattern greater
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ordering.ref.loc13_27: type = name_ref Ordering, file.%Ordering.decl [template = constants.%Ordering]
// CHECK:STDOUT:   %Greater.ref: %Ordering = name_ref Greater, @Ordering.%Greater
// CHECK:STDOUT:   %Ordering.ref.loc13_16: type = name_ref Ordering, file.%Ordering.decl [template = constants.%Ordering]
// CHECK:STDOUT:   %greater: %Ordering = bind_name greater, %Greater.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %inc.patt: %Ordering = binding_pattern inc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ordering.ref.loc14_23: type = name_ref Ordering, file.%Ordering.decl [template = constants.%Ordering]
// CHECK:STDOUT:   %Incomparable.ref: %Ordering = name_ref Incomparable, @Ordering.%Incomparable
// CHECK:STDOUT:   %Ordering.ref.loc14_12: type = name_ref Ordering, file.%Ordering.decl [template = constants.%Ordering]
// CHECK:STDOUT:   %inc: %Ordering = bind_name inc, %Incomparable.ref
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
