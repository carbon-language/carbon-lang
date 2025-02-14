// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/adapter/fail_adapt_with_subobjects.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/adapter/fail_adapt_with_subobjects.carbon

// --- fail_adapt_with_base.carbon

library "[[@TEST_NAME]]";

base class Base {}

class AdaptWithBase {
  // CHECK:STDERR: fail_adapt_with_base.carbon:[[@LINE+3]]:3: error: adapter with base class [AdaptWithBase]
  // CHECK:STDERR:   adapt i32;
  // CHECK:STDERR:   ^~~~~~~~~~
  adapt i32;
  // CHECK:STDERR: fail_adapt_with_base.carbon:[[@LINE+4]]:3: note: `base` declaration is here [AdaptWithBaseHere]
  // CHECK:STDERR:   extend base: Base;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: Base;
}

// --- fail_adapt_with_fields.carbon

library "[[@TEST_NAME]]";

class AdaptWithField {
  // CHECK:STDERR: fail_adapt_with_fields.carbon:[[@LINE+3]]:3: error: adapter with fields [AdaptWithFields]
  // CHECK:STDERR:   adapt i32;
  // CHECK:STDERR:   ^~~~~~~~~~
  adapt i32;
  // CHECK:STDERR: fail_adapt_with_fields.carbon:[[@LINE+4]]:7: note: first field declaration is here [AdaptWithFieldHere]
  // CHECK:STDERR:   var n: i32;
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  var n: i32;
}

class AdaptWithFields {
  // CHECK:STDERR: fail_adapt_with_fields.carbon:[[@LINE+3]]:3: error: adapter with fields [AdaptWithFields]
  // CHECK:STDERR:   adapt i32;
  // CHECK:STDERR:   ^~~~~~~~~~
  adapt i32;
  // CHECK:STDERR: fail_adapt_with_fields.carbon:[[@LINE+4]]:7: note: first field declaration is here [AdaptWithFieldHere]
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
  // CHECK:STDERR: fail_adapt_with_base_and_fields.carbon:[[@LINE+7]]:3: error: adapter with base class [AdaptWithBase]
  // CHECK:STDERR:   adapt {};
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_adapt_with_base_and_fields.carbon:[[@LINE-5]]:3: note: `base` declaration is here [AdaptWithBaseHere]
  // CHECK:STDERR:   extend base: Base;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  adapt {};
}

// CHECK:STDOUT: --- fail_adapt_with_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %AdaptWithBase: type = class_type @AdaptWithBase [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %AdaptWithBase.elem: type = unbound_element_type %AdaptWithBase, %Base [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .AdaptWithBase = %AdaptWithBase.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %AdaptWithBase.decl: type = class_decl @AdaptWithBase [concrete = constants.%AdaptWithBase] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptWithBase {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   adapt_decl %i32 [concrete]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc15: %AdaptWithBase.elem = base_decl %Base.ref, element<none> [concrete]
// CHECK:STDOUT:   complete_type_witness = <error>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptWithBase
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc15
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_with_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AdaptWithField: type = class_type @AdaptWithField [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %AdaptWithField.elem: type = unbound_element_type %AdaptWithField, %i32 [concrete]
// CHECK:STDOUT:   %AdaptWithFields: type = class_type @AdaptWithFields [concrete]
// CHECK:STDOUT:   %AdaptWithFields.elem: type = unbound_element_type %AdaptWithFields, %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .AdaptWithField = %AdaptWithField.decl
// CHECK:STDOUT:     .AdaptWithFields = %AdaptWithFields.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %AdaptWithField.decl: type = class_decl @AdaptWithField [concrete = constants.%AdaptWithField] {} {}
// CHECK:STDOUT:   %AdaptWithFields.decl: type = class_decl @AdaptWithFields [concrete = constants.%AdaptWithFields] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptWithField {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   adapt_decl %i32 [concrete]
// CHECK:STDOUT:   %.loc13_8: %AdaptWithField.elem = field_decl n, element<none> [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc13_3: %AdaptWithField.elem = var_pattern %.loc13_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %AdaptWithField.elem = var <none>
// CHECK:STDOUT:   complete_type_witness = <error>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptWithField
// CHECK:STDOUT:   .n = %.loc13_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptWithFields {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   adapt_decl %i32 [concrete]
// CHECK:STDOUT:   %.loc25_8: %AdaptWithFields.elem = field_decl a, element<none> [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc25_3: %AdaptWithFields.elem = var_pattern %.loc25_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc25: ref %AdaptWithFields.elem = var <none>
// CHECK:STDOUT:   %.loc26_8: %AdaptWithFields.elem = field_decl b, element<none> [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc26_3: %AdaptWithFields.elem = var_pattern %.loc26_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc26: ref %AdaptWithFields.elem = var <none>
// CHECK:STDOUT:   %.loc27_8: %AdaptWithFields.elem = field_decl c, element<none> [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc27_3: %AdaptWithFields.elem = var_pattern %.loc27_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc27: ref %AdaptWithFields.elem = var <none>
// CHECK:STDOUT:   complete_type_witness = <error>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptWithFields
// CHECK:STDOUT:   .a = %.loc25_8
// CHECK:STDOUT:   .b = %.loc26_8
// CHECK:STDOUT:   .c = %.loc27_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_with_base_and_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %AdaptWithBaseAndFields: type = class_type @AdaptWithBaseAndFields [concrete]
// CHECK:STDOUT:   %AdaptWithBaseAndFields.elem.767: type = unbound_element_type %AdaptWithBaseAndFields, %Base [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %AdaptWithBaseAndFields.elem.ddf: type = unbound_element_type %AdaptWithBaseAndFields, %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .AdaptWithBaseAndFields = %AdaptWithBaseAndFields.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %AdaptWithBaseAndFields.decl: type = class_decl @AdaptWithBaseAndFields [concrete = constants.%AdaptWithBaseAndFields] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptWithBaseAndFields {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc7: %AdaptWithBaseAndFields.elem.767 = base_decl %Base.ref, element<none> [concrete]
// CHECK:STDOUT:   %.loc8_8: %AdaptWithBaseAndFields.elem.ddf = field_decl n, element<none> [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc8_3: %AdaptWithBaseAndFields.elem.ddf = var_pattern %.loc8_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %AdaptWithBaseAndFields.elem.ddf = var <none>
// CHECK:STDOUT:   %.loc16_10: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc16_11: type = converted %.loc16_10, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   adapt_decl %.loc16_11 [concrete]
// CHECK:STDOUT:   complete_type_witness = <error>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptWithBaseAndFields
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .n = %.loc8_8
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
