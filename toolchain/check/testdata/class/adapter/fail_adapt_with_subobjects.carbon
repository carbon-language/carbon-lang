// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.31c: <witness> = impl_witness @Base.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.11f: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %pattern_type.1b9: type = pattern_type %ptr.11f [concrete]
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op.type: type = fn_type @Base.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op: %Base.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %AdaptWithBase: type = class_type @AdaptWithBase [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %AdaptWithBase.elem: type = unbound_element_type %AdaptWithBase, %Base [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.cad: <witness> = impl_witness @AdaptWithBase.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.52c: type = ptr_type %AdaptWithBase [concrete]
// CHECK:STDOUT:   %pattern_type.d4d: type = pattern_type %ptr.52c [concrete]
// CHECK:STDOUT:   %AdaptWithBase.as.Destroy.impl.Op.type: type = fn_type @AdaptWithBase.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %AdaptWithBase.as.Destroy.impl.Op: %AdaptWithBase.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
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
// CHECK:STDOUT: impl @Base.as.Destroy.impl: @Base.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op.decl: %Base.as.Destroy.impl.Op.type = fn_decl @Base.as.Destroy.impl.Op [concrete = constants.%Base.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.1b9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1b9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.11f = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %ptr.11f = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Base.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Base.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @AdaptWithBase.as.Destroy.impl: @AdaptWithBase.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %AdaptWithBase.as.Destroy.impl.Op.decl: %AdaptWithBase.as.Destroy.impl.Op.type = fn_decl @AdaptWithBase.as.Destroy.impl.Op [concrete = constants.%AdaptWithBase.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.d4d = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d4d = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc6: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.52c = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%AdaptWithBase [concrete = constants.%AdaptWithBase]
// CHECK:STDOUT:     %self: %ptr.52c = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %AdaptWithBase.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @AdaptWithBase.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:   impl_decl @Base.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Base.as.Destroy.impl.%Base.as.Destroy.impl.Op.decl), @Base.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.31c]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%AdaptWithBase [concrete = constants.%AdaptWithBase]
// CHECK:STDOUT:   impl_decl @AdaptWithBase.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@AdaptWithBase.as.Destroy.impl.%AdaptWithBase.as.Destroy.impl.Op.decl), @AdaptWithBase.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.cad]
// CHECK:STDOUT:   complete_type_witness = <error>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptWithBase
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc15
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.as.Destroy.impl.Op(%self.param: %ptr.11f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AdaptWithBase.as.Destroy.impl.Op(%self.param: %ptr.52c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_with_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AdaptWithField: type = class_type @AdaptWithField [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %AdaptWithField.elem: type = unbound_element_type %AdaptWithField, %i32 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.46a: <witness> = impl_witness @AdaptWithField.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.849: type = ptr_type %AdaptWithField [concrete]
// CHECK:STDOUT:   %pattern_type.55a: type = pattern_type %ptr.849 [concrete]
// CHECK:STDOUT:   %AdaptWithField.as.Destroy.impl.Op.type: type = fn_type @AdaptWithField.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %AdaptWithField.as.Destroy.impl.Op: %AdaptWithField.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AdaptWithFields: type = class_type @AdaptWithFields [concrete]
// CHECK:STDOUT:   %AdaptWithFields.elem: type = unbound_element_type %AdaptWithFields, %i32 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.eea: <witness> = impl_witness @AdaptWithFields.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.521: type = ptr_type %AdaptWithFields [concrete]
// CHECK:STDOUT:   %pattern_type.1a9: type = pattern_type %ptr.521 [concrete]
// CHECK:STDOUT:   %AdaptWithFields.as.Destroy.impl.Op.type: type = fn_type @AdaptWithFields.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %AdaptWithFields.as.Destroy.impl.Op: %AdaptWithFields.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT: impl @AdaptWithField.as.Destroy.impl: @AdaptWithField.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %AdaptWithField.as.Destroy.impl.Op.decl: %AdaptWithField.as.Destroy.impl.Op.type = fn_decl @AdaptWithField.as.Destroy.impl.Op [concrete = constants.%AdaptWithField.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.55a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.55a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.849 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%AdaptWithField [concrete = constants.%AdaptWithField]
// CHECK:STDOUT:     %self: %ptr.849 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %AdaptWithField.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @AdaptWithField.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @AdaptWithFields.as.Destroy.impl: @AdaptWithFields.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %AdaptWithFields.as.Destroy.impl.Op.decl: %AdaptWithFields.as.Destroy.impl.Op.type = fn_decl @AdaptWithFields.as.Destroy.impl.Op [concrete = constants.%AdaptWithFields.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.1a9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1a9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc16: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.521 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%AdaptWithFields [concrete = constants.%AdaptWithFields]
// CHECK:STDOUT:     %self: %ptr.521 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %AdaptWithFields.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @AdaptWithFields.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptWithField {
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   adapt_decl %i32.loc8 [concrete]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc13: %AdaptWithField.elem = field_decl n, element<none> [concrete]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%AdaptWithField [concrete = constants.%AdaptWithField]
// CHECK:STDOUT:   impl_decl @AdaptWithField.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@AdaptWithField.as.Destroy.impl.%AdaptWithField.as.Destroy.impl.Op.decl), @AdaptWithField.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.46a]
// CHECK:STDOUT:   complete_type_witness = <error>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptWithField
// CHECK:STDOUT:   .n = %.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptWithFields {
// CHECK:STDOUT:   %int_32.loc20: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc20: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   adapt_decl %i32.loc20 [concrete]
// CHECK:STDOUT:   %int_32.loc25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc25: %AdaptWithFields.elem = field_decl a, element<none> [concrete]
// CHECK:STDOUT:   %int_32.loc26: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc26: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc26: %AdaptWithFields.elem = field_decl b, element<none> [concrete]
// CHECK:STDOUT:   %int_32.loc27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc27: %AdaptWithFields.elem = field_decl c, element<none> [concrete]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%AdaptWithFields [concrete = constants.%AdaptWithFields]
// CHECK:STDOUT:   impl_decl @AdaptWithFields.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@AdaptWithFields.as.Destroy.impl.%AdaptWithFields.as.Destroy.impl.Op.decl), @AdaptWithFields.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.eea]
// CHECK:STDOUT:   complete_type_witness = <error>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptWithFields
// CHECK:STDOUT:   .a = %.loc25
// CHECK:STDOUT:   .b = %.loc26
// CHECK:STDOUT:   .c = %.loc27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AdaptWithField.as.Destroy.impl.Op(%self.param: %ptr.849) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AdaptWithFields.as.Destroy.impl.Op(%self.param: %ptr.521) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_with_base_and_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.31c: <witness> = impl_witness @Base.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.11f: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %pattern_type.1b9: type = pattern_type %ptr.11f [concrete]
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op.type: type = fn_type @Base.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op: %Base.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %AdaptWithBaseAndFields: type = class_type @AdaptWithBaseAndFields [concrete]
// CHECK:STDOUT:   %AdaptWithBaseAndFields.elem.767: type = unbound_element_type %AdaptWithBaseAndFields, %Base [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %AdaptWithBaseAndFields.elem.ddf: type = unbound_element_type %AdaptWithBaseAndFields, %i32 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.00a: <witness> = impl_witness @AdaptWithBaseAndFields.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.cbb: type = ptr_type %AdaptWithBaseAndFields [concrete]
// CHECK:STDOUT:   %pattern_type.926: type = pattern_type %ptr.cbb [concrete]
// CHECK:STDOUT:   %AdaptWithBaseAndFields.as.Destroy.impl.Op.type: type = fn_type @AdaptWithBaseAndFields.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %AdaptWithBaseAndFields.as.Destroy.impl.Op: %AdaptWithBaseAndFields.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
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
// CHECK:STDOUT: impl @Base.as.Destroy.impl: @Base.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op.decl: %Base.as.Destroy.impl.Op.type = fn_decl @Base.as.Destroy.impl.Op [concrete = constants.%Base.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.1b9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1b9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.11f = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %ptr.11f = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Base.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Base.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @AdaptWithBaseAndFields.as.Destroy.impl: @AdaptWithBaseAndFields.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %AdaptWithBaseAndFields.as.Destroy.impl.Op.decl: %AdaptWithBaseAndFields.as.Destroy.impl.Op.type = fn_decl @AdaptWithBaseAndFields.as.Destroy.impl.Op [concrete = constants.%AdaptWithBaseAndFields.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.926 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.926 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc6: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.cbb = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%AdaptWithBaseAndFields [concrete = constants.%AdaptWithBaseAndFields]
// CHECK:STDOUT:     %self: %ptr.cbb = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %AdaptWithBaseAndFields.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @AdaptWithBaseAndFields.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:   impl_decl @Base.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Base.as.Destroy.impl.%Base.as.Destroy.impl.Op.decl), @Base.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.31c]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptWithBaseAndFields {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc7: %AdaptWithBaseAndFields.elem.767 = base_decl %Base.ref, element<none> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc8: %AdaptWithBaseAndFields.elem.ddf = field_decl n, element<none> [concrete]
// CHECK:STDOUT:   %.loc16_10: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc16_11: type = converted %.loc16_10, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   adapt_decl %.loc16_11 [concrete]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%AdaptWithBaseAndFields [concrete = constants.%AdaptWithBaseAndFields]
// CHECK:STDOUT:   impl_decl @AdaptWithBaseAndFields.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@AdaptWithBaseAndFields.as.Destroy.impl.%AdaptWithBaseAndFields.as.Destroy.impl.Op.decl), @AdaptWithBaseAndFields.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.00a]
// CHECK:STDOUT:   complete_type_witness = <error>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptWithBaseAndFields
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .n = %.loc8
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.as.Destroy.impl.Op(%self.param: %ptr.11f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AdaptWithBaseAndFields.as.Destroy.impl.Op(%self.param: %ptr.cbb) = "no_op";
// CHECK:STDOUT:
