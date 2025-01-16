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

// --- fail_no_alternative_construct.carbon
library "[[@TEST_NAME]]";

choice Never {}

fn F() {
  // TODO: Can we produce a better diagnostic?
  // CHECK:STDERR: fail_no_alternative_construct.carbon:[[@LINE+3]]:22: error: cannot initialize class with 1 field from struct with 0 fields [StructInitElementCountMismatch]
  // CHECK:STDERR:   let never: Never = {};
  // CHECK:STDERR:                      ^~
  let never: Never = {};
}

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

// constants {
//   %S: type = class_type @S [template]
//   %TAlt: type = class_type @TAlt [template]
//   %empty_struct_type: type = struct_type {} [template]
//   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
//   %T.type: type = fn_type @T [template]
//   %T: %T.type = struct_value () [template]
// }
//
// imports {
//   %Core: <namespace> = namespace file.%Core.import, [template] {
//     import Core//prelude
//     import Core//prelude/...
//   }
// }
//
// file {
//   package: <namespace> = namespace [template] {
//     .Core = imports.%Core
//     .S = %S.decl
//   }
//   %Core.import = import Core
//   %S.decl: type = class_decl @S [template = constants.%S] {} {}
// }
//
// class @S {
//   %TAlt.decl: type = class_decl @TAlt [template = constants.%TAlt] {} {}
//   %T.decl: %T.type = fn_decl @T [template = constants.%T] {
//     %return.patt: %S = return_slot_pattern
//     %return.param_patt: %S = out_param_pattern %return.patt, runtime_param0
//   } {
//     %S.ref: type = name_ref S, file.%S.decl [template = constants.%S]
//     %return.param: ref %S = out_param runtime_param0
//     %return: ref %S = return_slot %return.param
//   }
//   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
//
// !members:
//   .Self = constants.%S
//   .TAlt = %TAlt.decl
//   .T = %T.decl
//   complete_type_witness = %complete_type
// }
//
// class @TAlt {
//   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
//
// !members:
//   .Self = constants.%TAlt
//   complete_type_witness = %complete_type
// }
//
// fn @T() -> %S;

// CHECK:STDOUT: --- no_alternative.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Never: type = class_type @Never [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.choice_storage: type = struct_type {.choice_storage: %empty_tuple.type} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.choice_storage [template]
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.choice_storage [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_alternative_construct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Never: type = class_type @Never [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.choice_storage: type = struct_type {.choice_storage: %empty_tuple.type} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.choice_storage [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Never.decl: type = class_decl @Never [template = constants.%Never] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Never {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.choice_storage [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_23.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_23.2: ref %Never = temporary_storage
// CHECK:STDOUT:   %.loc10_23.3: ref %Never = temporary %.loc10_23.2, <error>
// CHECK:STDOUT:   %.loc10_24.1: ref %Never = converted %.loc10_23.1, %.loc10_23.3
// CHECK:STDOUT:   %.loc10_24.2: %Never = bind_value %.loc10_24.1
// CHECK:STDOUT:   %never: %Never = bind_name never, %.loc10_24.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- one_alternative.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Always: type = class_type @Always [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type) [template]
// CHECK:STDOUT:   %struct_type.choice_storage: type = struct_type {.choice_storage: %tuple.type} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.choice_storage [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple) [template]
// CHECK:STDOUT:   %Always.val: %Always = struct_value (%tuple) [template]
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.choice_storage [template = constants.%complete_type]
// CHECK:STDOUT:   %.loc5_1.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_1.2: %empty_tuple.type = converted %.loc5_1.1, %empty_tuple [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_1.3: %tuple.type = tuple_literal (%.loc5_1.2)
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%.loc5_1.2) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc5_1.4: %tuple.type = converted %.loc5_1.3, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %.loc5_1.5: %struct_type.choice_storage = struct_literal (%.loc5_1.4)
// CHECK:STDOUT:   %tuple.elem0.loc5_1.1: %empty_tuple.type = tuple_access %.loc5_1.4, element0 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_1.6: ref %Always = temporary_storage
// CHECK:STDOUT:   %.loc5_1.7: ref %tuple.type = class_element_access %.loc5_1.6, element0
// CHECK:STDOUT:   %tuple.elem0.loc5_1.2: ref %empty_tuple.type = tuple_access %.loc5_1.7, element0
// CHECK:STDOUT:   %.loc5_1.8: init %empty_tuple.type = tuple_init () to %tuple.elem0.loc5_1.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_1.9: init %empty_tuple.type = converted %tuple.elem0.loc5_1.1, %.loc5_1.8 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_1.10: init %tuple.type = tuple_init (%.loc5_1.9) to %.loc5_1.7 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc5_1.11: init %tuple.type = converted %.loc5_1.4, %.loc5_1.10 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc5_1.12: init %tuple.type = initialize_from %.loc5_1.11 to %.loc5_1.7 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc5_1.13: init %Always = class_init (%.loc5_1.12), %.loc5_1.6 [template = constants.%Always.val]
// CHECK:STDOUT:   %.loc5_1.14: ref %Always = temporary %.loc5_1.6, %.loc5_1.13
// CHECK:STDOUT:   %.loc5_1.15: ref %Always = converted %.loc5_1.5, %.loc5_1.14
// CHECK:STDOUT:   %.loc5_1.16: %Always = bind_value %.loc5_1.15
// CHECK:STDOUT:   %Sunny: %Always = bind_name Sunny, %.loc5_1.16
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Sunny = %Sunny
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Always.ref: type = name_ref Always, file.%Always.decl [template = constants.%Always]
// CHECK:STDOUT:   %Sunny.ref: %Always = name_ref Sunny, @Always.%Sunny
// CHECK:STDOUT:   %mood: %Always = bind_name mood, %Sunny.ref
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- multiple_alternatives.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ordering: type = class_type @Ordering [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %u2: type = class_type @UInt, @UInt(%int_2.ecc) [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type, %empty_tuple.type, %empty_tuple.type) [template]
// CHECK:STDOUT:   %struct_type.discriminant.choice_storage.5c4: type = struct_type {.discriminant: %u2, .choice_storage: %tuple.type} [template]
// CHECK:STDOUT:   %complete_type.de2: <witness> = complete_type_witness %struct_type.discriminant.choice_storage.5c4 [template]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.f0e: type = fn_type @Convert.1, @ImplicitAs(%u2) [template]
// CHECK:STDOUT:   %impl_witness.f5e: <witness> = impl_witness (imports.%import_ref.c3d), @impl.1(%int_2.ecc) [template]
// CHECK:STDOUT:   %Convert.type.70b: type = fn_type @Convert.2, @impl.1(%int_2.ecc) [template]
// CHECK:STDOUT:   %Convert.474: %Convert.type.70b = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.5bb: <bound method> = bound_method %int_0.5c6, %Convert.474 [template]
// CHECK:STDOUT:   %Convert.specific_fn.dd9: <specific function> = specific_function %Convert.bound.5bb, @Convert.2(%int_2.ecc) [template]
// CHECK:STDOUT:   %int_0.9fd: %u2 = int_value 0 [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple, %empty_tuple, %empty_tuple) [template]
// CHECK:STDOUT:   %Ordering.val.1cf: %Ordering = struct_value (%int_0.9fd, %tuple) [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.0dd: <bound method> = bound_method %int_1.5b8, %Convert.474 [template]
// CHECK:STDOUT:   %Convert.specific_fn.95d: <specific function> = specific_function %Convert.bound.0dd, @Convert.2(%int_2.ecc) [template]
// CHECK:STDOUT:   %int_1.b2c: %u2 = int_value 1 [template]
// CHECK:STDOUT:   %Ordering.val.b7b: %Ordering = struct_value (%int_1.b2c, %tuple) [template]
// CHECK:STDOUT:   %Convert.bound.122: <bound method> = bound_method %int_2.ecc, %Convert.474 [template]
// CHECK:STDOUT:   %Convert.specific_fn.190: <specific function> = specific_function %Convert.bound.122, @Convert.2(%int_2.ecc) [template]
// CHECK:STDOUT:   %int_2.788: %u2 = int_value 2 [template]
// CHECK:STDOUT:   %Ordering.val.66b: %Ordering = struct_value (%int_2.788, %tuple) [template]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Convert.bound.a5c: <bound method> = bound_method %int_3.1ba, %Convert.474 [template]
// CHECK:STDOUT:   %Convert.specific_fn.9bc: <specific function> = specific_function %Convert.bound.a5c, @Convert.2(%int_2.ecc) [template]
// CHECK:STDOUT:   %int_3.975: %u2 = int_value 3 [template]
// CHECK:STDOUT:   %Ordering.val.8a3: %Ordering = struct_value (%int_3.975, %tuple) [template]
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.discriminant.choice_storage.5c4 [template = constants.%complete_type.de2]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc4: %Convert.type.f0e = impl_witness_access constants.%impl_witness.f5e, element0 [template = constants.%Convert.474]
// CHECK:STDOUT:   %Convert.bound.loc4: <bound method> = bound_method %int_0, %impl.elem0.loc4 [template = constants.%Convert.bound.5bb]
// CHECK:STDOUT:   %Convert.specific_fn.loc4: <specific function> = specific_function %Convert.bound.loc4, @Convert.2(constants.%int_2.ecc) [template = constants.%Convert.specific_fn.dd9]
// CHECK:STDOUT:   %int.convert_checked.loc4: init %u2 = call %Convert.specific_fn.loc4(%int_0) [template = constants.%int_0.9fd]
// CHECK:STDOUT:   %.loc4_7.1: %u2 = value_of_initializer %int.convert_checked.loc4 [template = constants.%int_0.9fd]
// CHECK:STDOUT:   %.loc4_7.2: %u2 = converted %int_0, %.loc4_7.1 [template = constants.%int_0.9fd]
// CHECK:STDOUT:   %.loc4_7.3: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple.loc4_7.1: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_7.4: %empty_tuple.type = converted %.loc4_7.3, %empty_tuple.loc4_7.1 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_7.5: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_7.6: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_7.7: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_7.8: %tuple.type = tuple_literal (%.loc4_7.4, %.loc4_7.5, %.loc4_7.6, %.loc4_7.7)
// CHECK:STDOUT:   %empty_tuple.loc4_7.2: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_7.9: %empty_tuple.type = converted %.loc4_7.5, %empty_tuple.loc4_7.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc4_7.3: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_7.10: %empty_tuple.type = converted %.loc4_7.6, %empty_tuple.loc4_7.3 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc4_7.4: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_7.11: %empty_tuple.type = converted %.loc4_7.7, %empty_tuple.loc4_7.4 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.loc4: %tuple.type = tuple_value (%.loc4_7.4, %.loc4_7.9, %.loc4_7.10, %.loc4_7.11) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc4_7.12: %tuple.type = converted %.loc4_7.8, %tuple.loc4 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc4_7.13: %struct_type.discriminant.choice_storage.5c4 = struct_literal (%.loc4_7.2, %.loc4_7.12)
// CHECK:STDOUT:   %.loc4_7.14: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc4_7.15: ref %u2 = class_element_access %.loc4_7.14, element0
// CHECK:STDOUT:   %.loc4_7.16: init %u2 = initialize_from %.loc4_7.2 to %.loc4_7.15 [template = constants.%int_0.9fd]
// CHECK:STDOUT:   %tuple.elem0.loc4_7.1: %empty_tuple.type = tuple_access %.loc4_7.12, element0 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_7.17: ref %tuple.type = class_element_access %.loc4_7.14, element1
// CHECK:STDOUT:   %tuple.elem0.loc4_7.2: ref %empty_tuple.type = tuple_access %.loc4_7.17, element0
// CHECK:STDOUT:   %.loc4_7.18: init %empty_tuple.type = tuple_init () to %tuple.elem0.loc4_7.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_7.19: init %empty_tuple.type = converted %tuple.elem0.loc4_7.1, %.loc4_7.18 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc4_7.1: %empty_tuple.type = tuple_access %.loc4_7.12, element1 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc4_7.2: ref %empty_tuple.type = tuple_access %.loc4_7.17, element1
// CHECK:STDOUT:   %.loc4_7.20: init %empty_tuple.type = tuple_init () to %tuple.elem1.loc4_7.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_7.21: init %empty_tuple.type = converted %tuple.elem1.loc4_7.1, %.loc4_7.20 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem2.loc4_7.1: %empty_tuple.type = tuple_access %.loc4_7.12, element2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem2.loc4_7.2: ref %empty_tuple.type = tuple_access %.loc4_7.17, element2
// CHECK:STDOUT:   %.loc4_7.22: init %empty_tuple.type = tuple_init () to %tuple.elem2.loc4_7.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_7.23: init %empty_tuple.type = converted %tuple.elem2.loc4_7.1, %.loc4_7.22 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem3.loc4_7.1: %empty_tuple.type = tuple_access %.loc4_7.12, element3 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem3.loc4_7.2: ref %empty_tuple.type = tuple_access %.loc4_7.17, element3
// CHECK:STDOUT:   %.loc4_7.24: init %empty_tuple.type = tuple_init () to %tuple.elem3.loc4_7.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_7.25: init %empty_tuple.type = converted %tuple.elem3.loc4_7.1, %.loc4_7.24 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_7.26: init %tuple.type = tuple_init (%.loc4_7.19, %.loc4_7.21, %.loc4_7.23, %.loc4_7.25) to %.loc4_7.17 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc4_7.27: init %tuple.type = converted %.loc4_7.12, %.loc4_7.26 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc4_7.28: init %Ordering = class_init (%.loc4_7.16, %.loc4_7.27), %.loc4_7.14 [template = constants.%Ordering.val.1cf]
// CHECK:STDOUT:   %.loc4_7.29: ref %Ordering = temporary %.loc4_7.14, %.loc4_7.28
// CHECK:STDOUT:   %.loc4_7.30: ref %Ordering = converted %.loc4_7.13, %.loc4_7.29
// CHECK:STDOUT:   %.loc4_7.31: %Ordering = bind_value %.loc4_7.30
// CHECK:STDOUT:   %Less: %Ordering = bind_name Less, %.loc4_7.31
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc5: %Convert.type.f0e = impl_witness_access constants.%impl_witness.f5e, element0 [template = constants.%Convert.474]
// CHECK:STDOUT:   %Convert.bound.loc5: <bound method> = bound_method %int_1, %impl.elem0.loc5 [template = constants.%Convert.bound.0dd]
// CHECK:STDOUT:   %Convert.specific_fn.loc5: <specific function> = specific_function %Convert.bound.loc5, @Convert.2(constants.%int_2.ecc) [template = constants.%Convert.specific_fn.95d]
// CHECK:STDOUT:   %int.convert_checked.loc5: init %u2 = call %Convert.specific_fn.loc5(%int_1) [template = constants.%int_1.b2c]
// CHECK:STDOUT:   %.loc5_13.1: %u2 = value_of_initializer %int.convert_checked.loc5 [template = constants.%int_1.b2c]
// CHECK:STDOUT:   %.loc5_13.2: %u2 = converted %int_1, %.loc5_13.1 [template = constants.%int_1.b2c]
// CHECK:STDOUT:   %.loc5_13.3: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc5_13.4: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple.loc5_13.1: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_13.5: %empty_tuple.type = converted %.loc5_13.4, %empty_tuple.loc5_13.1 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_13.6: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc5_13.7: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc5_13.8: %tuple.type = tuple_literal (%.loc5_13.3, %.loc5_13.5, %.loc5_13.6, %.loc5_13.7)
// CHECK:STDOUT:   %empty_tuple.loc5_13.2: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_13.9: %empty_tuple.type = converted %.loc5_13.3, %empty_tuple.loc5_13.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc5_13.3: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_13.10: %empty_tuple.type = converted %.loc5_13.6, %empty_tuple.loc5_13.3 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc5_13.4: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_13.11: %empty_tuple.type = converted %.loc5_13.7, %empty_tuple.loc5_13.4 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.loc5: %tuple.type = tuple_value (%.loc5_13.9, %.loc5_13.5, %.loc5_13.10, %.loc5_13.11) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc5_13.12: %tuple.type = converted %.loc5_13.8, %tuple.loc5 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc5_13.13: %struct_type.discriminant.choice_storage.5c4 = struct_literal (%.loc5_13.2, %.loc5_13.12)
// CHECK:STDOUT:   %.loc5_13.14: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc5_13.15: ref %u2 = class_element_access %.loc5_13.14, element0
// CHECK:STDOUT:   %.loc5_13.16: init %u2 = initialize_from %.loc5_13.2 to %.loc5_13.15 [template = constants.%int_1.b2c]
// CHECK:STDOUT:   %tuple.elem0.loc5_13.1: %empty_tuple.type = tuple_access %.loc5_13.12, element0 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_13.17: ref %tuple.type = class_element_access %.loc5_13.14, element1
// CHECK:STDOUT:   %tuple.elem0.loc5_13.2: ref %empty_tuple.type = tuple_access %.loc5_13.17, element0
// CHECK:STDOUT:   %.loc5_13.18: init %empty_tuple.type = tuple_init () to %tuple.elem0.loc5_13.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_13.19: init %empty_tuple.type = converted %tuple.elem0.loc5_13.1, %.loc5_13.18 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc5_13.1: %empty_tuple.type = tuple_access %.loc5_13.12, element1 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc5_13.2: ref %empty_tuple.type = tuple_access %.loc5_13.17, element1
// CHECK:STDOUT:   %.loc5_13.20: init %empty_tuple.type = tuple_init () to %tuple.elem1.loc5_13.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_13.21: init %empty_tuple.type = converted %tuple.elem1.loc5_13.1, %.loc5_13.20 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem2.loc5_13.1: %empty_tuple.type = tuple_access %.loc5_13.12, element2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem2.loc5_13.2: ref %empty_tuple.type = tuple_access %.loc5_13.17, element2
// CHECK:STDOUT:   %.loc5_13.22: init %empty_tuple.type = tuple_init () to %tuple.elem2.loc5_13.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_13.23: init %empty_tuple.type = converted %tuple.elem2.loc5_13.1, %.loc5_13.22 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem3.loc5_13.1: %empty_tuple.type = tuple_access %.loc5_13.12, element3 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem3.loc5_13.2: ref %empty_tuple.type = tuple_access %.loc5_13.17, element3
// CHECK:STDOUT:   %.loc5_13.24: init %empty_tuple.type = tuple_init () to %tuple.elem3.loc5_13.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_13.25: init %empty_tuple.type = converted %tuple.elem3.loc5_13.1, %.loc5_13.24 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_13.26: init %tuple.type = tuple_init (%.loc5_13.19, %.loc5_13.21, %.loc5_13.23, %.loc5_13.25) to %.loc5_13.17 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc5_13.27: init %tuple.type = converted %.loc5_13.12, %.loc5_13.26 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc5_13.28: init %Ordering = class_init (%.loc5_13.16, %.loc5_13.27), %.loc5_13.14 [template = constants.%Ordering.val.b7b]
// CHECK:STDOUT:   %.loc5_13.29: ref %Ordering = temporary %.loc5_13.14, %.loc5_13.28
// CHECK:STDOUT:   %.loc5_13.30: ref %Ordering = converted %.loc5_13.13, %.loc5_13.29
// CHECK:STDOUT:   %.loc5_13.31: %Ordering = bind_value %.loc5_13.30
// CHECK:STDOUT:   %Equivalent: %Ordering = bind_name Equivalent, %.loc5_13.31
// CHECK:STDOUT:   %int_2.loc6: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc6: %Convert.type.f0e = impl_witness_access constants.%impl_witness.f5e, element0 [template = constants.%Convert.474]
// CHECK:STDOUT:   %Convert.bound.loc6: <bound method> = bound_method %int_2.loc6, %impl.elem0.loc6 [template = constants.%Convert.bound.122]
// CHECK:STDOUT:   %Convert.specific_fn.loc6: <specific function> = specific_function %Convert.bound.loc6, @Convert.2(constants.%int_2.ecc) [template = constants.%Convert.specific_fn.190]
// CHECK:STDOUT:   %int.convert_checked.loc6: init %u2 = call %Convert.specific_fn.loc6(%int_2.loc6) [template = constants.%int_2.788]
// CHECK:STDOUT:   %.loc6_10.1: %u2 = value_of_initializer %int.convert_checked.loc6 [template = constants.%int_2.788]
// CHECK:STDOUT:   %.loc6_10.2: %u2 = converted %int_2.loc6, %.loc6_10.1 [template = constants.%int_2.788]
// CHECK:STDOUT:   %.loc6_10.3: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_10.4: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_10.5: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple.loc6_10.1: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_10.6: %empty_tuple.type = converted %.loc6_10.5, %empty_tuple.loc6_10.1 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_10.7: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_10.8: %tuple.type = tuple_literal (%.loc6_10.3, %.loc6_10.4, %.loc6_10.6, %.loc6_10.7)
// CHECK:STDOUT:   %empty_tuple.loc6_10.2: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_10.9: %empty_tuple.type = converted %.loc6_10.3, %empty_tuple.loc6_10.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc6_10.3: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_10.10: %empty_tuple.type = converted %.loc6_10.4, %empty_tuple.loc6_10.3 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc6_10.4: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_10.11: %empty_tuple.type = converted %.loc6_10.7, %empty_tuple.loc6_10.4 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.loc6: %tuple.type = tuple_value (%.loc6_10.9, %.loc6_10.10, %.loc6_10.6, %.loc6_10.11) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_10.12: %tuple.type = converted %.loc6_10.8, %tuple.loc6 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_10.13: %struct_type.discriminant.choice_storage.5c4 = struct_literal (%.loc6_10.2, %.loc6_10.12)
// CHECK:STDOUT:   %.loc6_10.14: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc6_10.15: ref %u2 = class_element_access %.loc6_10.14, element0
// CHECK:STDOUT:   %.loc6_10.16: init %u2 = initialize_from %.loc6_10.2 to %.loc6_10.15 [template = constants.%int_2.788]
// CHECK:STDOUT:   %tuple.elem0.loc6_10.1: %empty_tuple.type = tuple_access %.loc6_10.12, element0 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_10.17: ref %tuple.type = class_element_access %.loc6_10.14, element1
// CHECK:STDOUT:   %tuple.elem0.loc6_10.2: ref %empty_tuple.type = tuple_access %.loc6_10.17, element0
// CHECK:STDOUT:   %.loc6_10.18: init %empty_tuple.type = tuple_init () to %tuple.elem0.loc6_10.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_10.19: init %empty_tuple.type = converted %tuple.elem0.loc6_10.1, %.loc6_10.18 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc6_10.1: %empty_tuple.type = tuple_access %.loc6_10.12, element1 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc6_10.2: ref %empty_tuple.type = tuple_access %.loc6_10.17, element1
// CHECK:STDOUT:   %.loc6_10.20: init %empty_tuple.type = tuple_init () to %tuple.elem1.loc6_10.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_10.21: init %empty_tuple.type = converted %tuple.elem1.loc6_10.1, %.loc6_10.20 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem2.loc6_10.1: %empty_tuple.type = tuple_access %.loc6_10.12, element2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem2.loc6_10.2: ref %empty_tuple.type = tuple_access %.loc6_10.17, element2
// CHECK:STDOUT:   %.loc6_10.22: init %empty_tuple.type = tuple_init () to %tuple.elem2.loc6_10.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_10.23: init %empty_tuple.type = converted %tuple.elem2.loc6_10.1, %.loc6_10.22 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem3.loc6_10.1: %empty_tuple.type = tuple_access %.loc6_10.12, element3 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem3.loc6_10.2: ref %empty_tuple.type = tuple_access %.loc6_10.17, element3
// CHECK:STDOUT:   %.loc6_10.24: init %empty_tuple.type = tuple_init () to %tuple.elem3.loc6_10.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_10.25: init %empty_tuple.type = converted %tuple.elem3.loc6_10.1, %.loc6_10.24 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_10.26: init %tuple.type = tuple_init (%.loc6_10.19, %.loc6_10.21, %.loc6_10.23, %.loc6_10.25) to %.loc6_10.17 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_10.27: init %tuple.type = converted %.loc6_10.12, %.loc6_10.26 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_10.28: init %Ordering = class_init (%.loc6_10.16, %.loc6_10.27), %.loc6_10.14 [template = constants.%Ordering.val.66b]
// CHECK:STDOUT:   %.loc6_10.29: ref %Ordering = temporary %.loc6_10.14, %.loc6_10.28
// CHECK:STDOUT:   %.loc6_10.30: ref %Ordering = converted %.loc6_10.13, %.loc6_10.29
// CHECK:STDOUT:   %.loc6_10.31: %Ordering = bind_value %.loc6_10.30
// CHECK:STDOUT:   %Greater: %Ordering = bind_name Greater, %.loc6_10.31
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc8: %Convert.type.f0e = impl_witness_access constants.%impl_witness.f5e, element0 [template = constants.%Convert.474]
// CHECK:STDOUT:   %Convert.bound.loc8: <bound method> = bound_method %int_3, %impl.elem0.loc8 [template = constants.%Convert.bound.a5c]
// CHECK:STDOUT:   %Convert.specific_fn.loc8: <specific function> = specific_function %Convert.bound.loc8, @Convert.2(constants.%int_2.ecc) [template = constants.%Convert.specific_fn.9bc]
// CHECK:STDOUT:   %int.convert_checked.loc8: init %u2 = call %Convert.specific_fn.loc8(%int_3) [template = constants.%int_3.975]
// CHECK:STDOUT:   %.loc8_1.1: %u2 = value_of_initializer %int.convert_checked.loc8 [template = constants.%int_3.975]
// CHECK:STDOUT:   %.loc8_1.2: %u2 = converted %int_3, %.loc8_1.1 [template = constants.%int_3.975]
// CHECK:STDOUT:   %.loc8_1.3: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc8_1.4: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc8_1.5: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc8_1.6: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple.loc8_1.1: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_1.7: %empty_tuple.type = converted %.loc8_1.6, %empty_tuple.loc8_1.1 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_1.8: %tuple.type = tuple_literal (%.loc8_1.3, %.loc8_1.4, %.loc8_1.5, %.loc8_1.7)
// CHECK:STDOUT:   %empty_tuple.loc8_1.2: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_1.9: %empty_tuple.type = converted %.loc8_1.3, %empty_tuple.loc8_1.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc8_1.3: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_1.10: %empty_tuple.type = converted %.loc8_1.4, %empty_tuple.loc8_1.3 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc8_1.4: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_1.11: %empty_tuple.type = converted %.loc8_1.5, %empty_tuple.loc8_1.4 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.loc8: %tuple.type = tuple_value (%.loc8_1.9, %.loc8_1.10, %.loc8_1.11, %.loc8_1.7) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc8_1.12: %tuple.type = converted %.loc8_1.8, %tuple.loc8 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc8_1.13: %struct_type.discriminant.choice_storage.5c4 = struct_literal (%.loc8_1.2, %.loc8_1.12)
// CHECK:STDOUT:   %.loc8_1.14: ref %Ordering = temporary_storage
// CHECK:STDOUT:   %.loc8_1.15: ref %u2 = class_element_access %.loc8_1.14, element0
// CHECK:STDOUT:   %.loc8_1.16: init %u2 = initialize_from %.loc8_1.2 to %.loc8_1.15 [template = constants.%int_3.975]
// CHECK:STDOUT:   %tuple.elem0.loc8_1.1: %empty_tuple.type = tuple_access %.loc8_1.12, element0 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_1.17: ref %tuple.type = class_element_access %.loc8_1.14, element1
// CHECK:STDOUT:   %tuple.elem0.loc8_1.2: ref %empty_tuple.type = tuple_access %.loc8_1.17, element0
// CHECK:STDOUT:   %.loc8_1.18: init %empty_tuple.type = tuple_init () to %tuple.elem0.loc8_1.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_1.19: init %empty_tuple.type = converted %tuple.elem0.loc8_1.1, %.loc8_1.18 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc8_1.1: %empty_tuple.type = tuple_access %.loc8_1.12, element1 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc8_1.2: ref %empty_tuple.type = tuple_access %.loc8_1.17, element1
// CHECK:STDOUT:   %.loc8_1.20: init %empty_tuple.type = tuple_init () to %tuple.elem1.loc8_1.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_1.21: init %empty_tuple.type = converted %tuple.elem1.loc8_1.1, %.loc8_1.20 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem2.loc8_1.1: %empty_tuple.type = tuple_access %.loc8_1.12, element2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem2.loc8_1.2: ref %empty_tuple.type = tuple_access %.loc8_1.17, element2
// CHECK:STDOUT:   %.loc8_1.22: init %empty_tuple.type = tuple_init () to %tuple.elem2.loc8_1.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_1.23: init %empty_tuple.type = converted %tuple.elem2.loc8_1.1, %.loc8_1.22 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem3.loc8_1.1: %empty_tuple.type = tuple_access %.loc8_1.12, element3 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem3.loc8_1.2: ref %empty_tuple.type = tuple_access %.loc8_1.17, element3
// CHECK:STDOUT:   %.loc8_1.24: init %empty_tuple.type = tuple_init () to %tuple.elem3.loc8_1.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_1.25: init %empty_tuple.type = converted %tuple.elem3.loc8_1.1, %.loc8_1.24 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_1.26: init %tuple.type = tuple_init (%.loc8_1.19, %.loc8_1.21, %.loc8_1.23, %.loc8_1.25) to %.loc8_1.17 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc8_1.27: init %tuple.type = converted %.loc8_1.12, %.loc8_1.26 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc8_1.28: init %Ordering = class_init (%.loc8_1.16, %.loc8_1.27), %.loc8_1.14 [template = constants.%Ordering.val.8a3]
// CHECK:STDOUT:   %.loc8_1.29: ref %Ordering = temporary %.loc8_1.14, %.loc8_1.28
// CHECK:STDOUT:   %.loc8_1.30: ref %Ordering = converted %.loc8_1.13, %.loc8_1.29
// CHECK:STDOUT:   %.loc8_1.31: %Ordering = bind_value %.loc8_1.30
// CHECK:STDOUT:   %Incomparable: %Ordering = bind_name Incomparable, %.loc8_1.31
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Less = %Less
// CHECK:STDOUT:   .Equivalent = %Equivalent
// CHECK:STDOUT:   .Greater = %Greater
// CHECK:STDOUT:   .Incomparable = %Incomparable
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Ordering.ref.loc11: type = name_ref Ordering, file.%Ordering.decl [template = constants.%Ordering]
// CHECK:STDOUT:   %Less.ref: %Ordering = name_ref Less, @Ordering.%Less
// CHECK:STDOUT:   %less: %Ordering = bind_name less, %Less.ref
// CHECK:STDOUT:   %Ordering.ref.loc12: type = name_ref Ordering, file.%Ordering.decl [template = constants.%Ordering]
// CHECK:STDOUT:   %Equivalent.ref: %Ordering = name_ref Equivalent, @Ordering.%Equivalent
// CHECK:STDOUT:   %equiv: %Ordering = bind_name equiv, %Equivalent.ref
// CHECK:STDOUT:   %Ordering.ref.loc13: type = name_ref Ordering, file.%Ordering.decl [template = constants.%Ordering]
// CHECK:STDOUT:   %Greater.ref: %Ordering = name_ref Greater, @Ordering.%Greater
// CHECK:STDOUT:   %greater: %Ordering = bind_name greater, %Greater.ref
// CHECK:STDOUT:   %Ordering.ref.loc14: type = name_ref Ordering, file.%Ordering.decl [template = constants.%Ordering]
// CHECK:STDOUT:   %Incomparable.ref: %Ordering = name_ref Incomparable, @Ordering.%Incomparable
// CHECK:STDOUT:   %inc: %Ordering = bind_name inc, %Incomparable.ref
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
