// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_adapt_with_subobjects.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_adapt_with_subobjects.carbon

// --- fail_adapt_with_base.carbon

library "[[@TEST_NAME]]";

base class Base {}

class AdaptWithBase {
  // CHECK:STDERR: fail_adapt_with_base.carbon:[[@LINE+3]]:3: error: adapter with base class
  // CHECK:STDERR:   adapt i32;
  // CHECK:STDERR:   ^~~~~~~~~~
  adapt i32;
  // CHECK:STDERR: fail_adapt_with_base.carbon:[[@LINE+4]]:3: `base` declaration is here
  // CHECK:STDERR:   extend base: Base;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: Base;
}

// --- fail_adapt_with_fields.carbon

library "[[@TEST_NAME]]";

class AdaptWithField {
  // CHECK:STDERR: fail_adapt_with_fields.carbon:[[@LINE+3]]:3: error: adapter with fields
  // CHECK:STDERR:   adapt i32;
  // CHECK:STDERR:   ^~~~~~~~~~
  adapt i32;
  // CHECK:STDERR: fail_adapt_with_fields.carbon:[[@LINE+4]]:7: first field declaration is here
  // CHECK:STDERR:   var n: i32;
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  var n: i32;
}

class AdaptWithFields {
  // CHECK:STDERR: fail_adapt_with_fields.carbon:[[@LINE+3]]:3: error: adapter with fields
  // CHECK:STDERR:   adapt i32;
  // CHECK:STDERR:   ^~~~~~~~~~
  adapt i32;
  // CHECK:STDERR: fail_adapt_with_fields.carbon:[[@LINE+4]]:7: first field declaration is here
  // CHECK:STDERR:   var a: i32;
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  var a: i32;
  var b: i32;
  var c: i32;
}

// --- fail_adapt_with_base_and_fields.carbon

library "[[@TEST_NAME]]";

base class Base {}

class AdaptWithBaseAndFields {
  extend base: Base;
  var n: i32;
  // CHECK:STDERR: fail_adapt_with_base_and_fields.carbon:[[@LINE+6]]:3: error: adapter with base class
  // CHECK:STDERR:   adapt {};
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_adapt_with_base_and_fields.carbon:[[@LINE-5]]:3: `base` declaration is here
  // CHECK:STDERR:   extend base: Base;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  adapt {};
}

// CHECK:STDOUT: --- fail_adapt_with_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %AdaptWithBase: type = class_type @AdaptWithBase [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %AdaptWithBase, %Base [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .AdaptWithBase = %AdaptWithBase.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %AdaptWithBase.decl: type = class_decl @AdaptWithBase [template = constants.%AdaptWithBase] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptWithBase {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc10_12.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc10_12.2: type = converted %int.make_type_32, %.loc10_12.1 [template = i32]
// CHECK:STDOUT:   adapt_decl i32
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc15: %.5 = base_decl %Base, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptWithBase
// CHECK:STDOUT:   .base = %.loc15
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_with_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AdaptWithField: type = class_type @AdaptWithField [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %AdaptWithField, i32 [template]
// CHECK:STDOUT:   %AdaptWithFields: type = class_type @AdaptWithFields [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type %AdaptWithFields, i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .AdaptWithField = %AdaptWithField.decl
// CHECK:STDOUT:     .AdaptWithFields = %AdaptWithFields.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %AdaptWithField.decl: type = class_decl @AdaptWithField [template = constants.%AdaptWithField] {}
// CHECK:STDOUT:   %AdaptWithFields.decl: type = class_decl @AdaptWithFields [template = constants.%AdaptWithFields] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptWithField {
// CHECK:STDOUT:   %int.make_type_32.loc8: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc8_12.1: type = value_of_initializer %int.make_type_32.loc8 [template = i32]
// CHECK:STDOUT:   %.loc8_12.2: type = converted %int.make_type_32.loc8, %.loc8_12.1 [template = i32]
// CHECK:STDOUT:   adapt_decl i32
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_10.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_10.2: type = converted %int.make_type_32.loc13, %.loc13_10.1 [template = i32]
// CHECK:STDOUT:   %.loc13_8: %.2 = field_decl n, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptWithField
// CHECK:STDOUT:   .n = %.loc13_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptWithFields {
// CHECK:STDOUT:   %int.make_type_32.loc20: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc20_12.1: type = value_of_initializer %int.make_type_32.loc20 [template = i32]
// CHECK:STDOUT:   %.loc20_12.2: type = converted %int.make_type_32.loc20, %.loc20_12.1 [template = i32]
// CHECK:STDOUT:   adapt_decl i32
// CHECK:STDOUT:   %int.make_type_32.loc25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc25_10.1: type = value_of_initializer %int.make_type_32.loc25 [template = i32]
// CHECK:STDOUT:   %.loc25_10.2: type = converted %int.make_type_32.loc25, %.loc25_10.1 [template = i32]
// CHECK:STDOUT:   %.loc25_8: %.3 = field_decl a, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc26: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc26_10.1: type = value_of_initializer %int.make_type_32.loc26 [template = i32]
// CHECK:STDOUT:   %.loc26_10.2: type = converted %int.make_type_32.loc26, %.loc26_10.1 [template = i32]
// CHECK:STDOUT:   %.loc26_8: %.3 = field_decl b, element1 [template]
// CHECK:STDOUT:   %int.make_type_32.loc27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc27_10.1: type = value_of_initializer %int.make_type_32.loc27 [template = i32]
// CHECK:STDOUT:   %.loc27_10.2: type = converted %int.make_type_32.loc27, %.loc27_10.1 [template = i32]
// CHECK:STDOUT:   %.loc27_8: %.3 = field_decl c, element2 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptWithFields
// CHECK:STDOUT:   .a = %.loc25_8
// CHECK:STDOUT:   .b = %.loc26_8
// CHECK:STDOUT:   .c = %.loc27_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_with_base_and_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %AdaptWithBaseAndFields: type = class_type @AdaptWithBaseAndFields [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %AdaptWithBaseAndFields, %Base [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %AdaptWithBaseAndFields, i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .AdaptWithBaseAndFields = %AdaptWithBaseAndFields.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %AdaptWithBaseAndFields.decl: type = class_decl @AdaptWithBaseAndFields [template = constants.%AdaptWithBaseAndFields] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptWithBaseAndFields {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc7: %.5 = base_decl %Base, element0 [template]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc8_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc8_10.2: type = converted %int.make_type_32, %.loc8_10.1 [template = i32]
// CHECK:STDOUT:   %.loc8_8: %.6 = field_decl n, element1 [template]
// CHECK:STDOUT:   %.loc15_10: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc15_11: type = converted %.loc15_10, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   adapt_decl %.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptWithBaseAndFields
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .n = %.loc8_8
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
