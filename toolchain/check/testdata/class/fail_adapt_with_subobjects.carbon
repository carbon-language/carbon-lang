// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fail_adapt_with_base.carbon

library "fail_adapt_with_base" api;

base class Base {}

class AdaptWithBase {
  // CHECK:STDERR: fail_adapt_with_base.carbon:[[@LINE+3]]:3: ERROR: Adapter cannot have a base class.
  // CHECK:STDERR:   adapt i32;
  // CHECK:STDERR:   ^~~~~~~~~~
  adapt i32;
  // CHECK:STDERR: fail_adapt_with_base.carbon:[[@LINE+4]]:3: `base` declaration is here.
  // CHECK:STDERR:   extend base: Base;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: Base;
}

// --- fail_adapt_with_fields.carbon

library "fail_adapt_with_fields" api;

class AdaptWithField {
  // CHECK:STDERR: fail_adapt_with_fields.carbon:[[@LINE+3]]:3: ERROR: Adapter cannot have fields.
  // CHECK:STDERR:   adapt i32;
  // CHECK:STDERR:   ^~~~~~~~~~
  adapt i32;
  // CHECK:STDERR: fail_adapt_with_fields.carbon:[[@LINE+4]]:7: First field declaration is here.
  // CHECK:STDERR:   var n: i32;
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  var n: i32;
}

class AdaptWithFields {
  // CHECK:STDERR: fail_adapt_with_fields.carbon:[[@LINE+3]]:3: ERROR: Adapter cannot have fields.
  // CHECK:STDERR:   adapt i32;
  // CHECK:STDERR:   ^~~~~~~~~~
  adapt i32;
  // CHECK:STDERR: fail_adapt_with_fields.carbon:[[@LINE+4]]:7: First field declaration is here.
  // CHECK:STDERR:   var a: i32;
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  var a: i32;
  var b: i32;
  var c: i32;
}

// --- fail_adapt_with_base_and_fields.carbon

library "fail_adapt_with_base_and_fields" api;

base class Base {}

class AdaptWithBaseAndFields {
  extend base: Base;
  var n: i32;
  // CHECK:STDERR: fail_adapt_with_base_and_fields.carbon:[[@LINE+6]]:3: ERROR: Adapter cannot have a base class.
  // CHECK:STDERR:   adapt {};
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_adapt_with_base_and_fields.carbon:[[@LINE-5]]:3: `base` declaration is here.
  // CHECK:STDERR:   extend base: Base;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  adapt {};
}

// CHECK:STDOUT: --- fail_adapt_with_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %AdaptWithBase: type = class_type @AdaptWithBase [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type AdaptWithBase, Base [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .AdaptWithBase = %AdaptWithBase.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %AdaptWithBase.decl: type = class_decl @AdaptWithBase [template = constants.%AdaptWithBase] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptWithBase {
// CHECK:STDOUT:   adapt_decl i32
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc15: <unbound element of class AdaptWithBase> = base_decl Base, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptWithBase
// CHECK:STDOUT:   .base = %.loc15
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_with_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AdaptWithField: type = class_type @AdaptWithField [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type AdaptWithField, i32 [template]
// CHECK:STDOUT:   %AdaptWithFields: type = class_type @AdaptWithFields [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type AdaptWithFields, i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .AdaptWithField = %AdaptWithField.decl
// CHECK:STDOUT:     .AdaptWithFields = %AdaptWithFields.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %AdaptWithField.decl: type = class_decl @AdaptWithField [template = constants.%AdaptWithField] {}
// CHECK:STDOUT:   %AdaptWithFields.decl: type = class_decl @AdaptWithFields [template = constants.%AdaptWithFields] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptWithField {
// CHECK:STDOUT:   adapt_decl i32
// CHECK:STDOUT:   %.loc13: <unbound element of class AdaptWithField> = field_decl n, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptWithField
// CHECK:STDOUT:   .n = %.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptWithFields {
// CHECK:STDOUT:   adapt_decl i32
// CHECK:STDOUT:   %.loc25: <unbound element of class AdaptWithFields> = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc26: <unbound element of class AdaptWithFields> = field_decl b, element1 [template]
// CHECK:STDOUT:   %.loc27: <unbound element of class AdaptWithFields> = field_decl c, element2 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptWithFields
// CHECK:STDOUT:   .a = %.loc25
// CHECK:STDOUT:   .b = %.loc26
// CHECK:STDOUT:   .c = %.loc27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_with_base_and_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %AdaptWithBaseAndFields: type = class_type @AdaptWithBaseAndFields [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type AdaptWithBaseAndFields, Base [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type AdaptWithBaseAndFields, i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .AdaptWithBaseAndFields = %AdaptWithBaseAndFields.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %AdaptWithBaseAndFields.decl: type = class_decl @AdaptWithBaseAndFields [template = constants.%AdaptWithBaseAndFields] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptWithBaseAndFields {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc7: <unbound element of class AdaptWithBaseAndFields> = base_decl Base, element0 [template]
// CHECK:STDOUT:   %.loc8: <unbound element of class AdaptWithBaseAndFields> = field_decl n, element1 [template]
// CHECK:STDOUT:   %.loc15_10: {} = struct_literal ()
// CHECK:STDOUT:   %.loc15_11: type = converted %.loc15_10, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   adapt_decl {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptWithBaseAndFields
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .n = %.loc8
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
