// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class Base {}
class Final {}

class DeriveFromNonType {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:3: ERROR: Cannot implicitly convert from `i32` to `type`.
  // CHECK:STDERR:   extend base: 32;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  extend base: 32;
}

class DeriveFromi32 {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:3: ERROR: Deriving from final type `i32`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   extend base: i32;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  extend base: i32;
}

// It's not really important whether this conversion produces an error or not,
// but it shouldn't crash.
fn ConvertToBadBasei32(p: DeriveFromi32*) -> i32* { return p; }

class DeriveFromTuple {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:3: ERROR: Deriving from final type `(Base,)`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   extend base: (Base,);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  extend base: (Base,);
}

fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }

// TODO: Should we allow this?
// We do allow `{.base = {.a: i32, .b: i32}}`.
class DeriveFromStruct {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:3: ERROR: Deriving from final type `{.a: i32, .b: i32}`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   extend base: {.a: i32, .b: i32};
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  extend base: {.a: i32, .b: i32};
}

fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }

base class Incomplete;

class DeriveFromIncomplete {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+6]]:3: ERROR: Base `Incomplete` is an incomplete type.
  // CHECK:STDERR:   extend base: Incomplete;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE-6]]:1: Class was forward declared here.
  // CHECK:STDERR: base class Incomplete;
  // CHECK:STDERR: ^~~~
  extend base: Incomplete;
}

// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:82: ERROR: Cannot implicitly convert from `DeriveFromIncomplete*` to `Incomplete*`.
// CHECK:STDERR: fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }
// CHECK:STDERR:                                                                                  ^
fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }

class DeriveFromFinal {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:3: ERROR: Deriving from final type `Final`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   extend base: Final;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  extend base: Final;
}

fn ConvertToBadBaseFinal(p: DeriveFromFinal*) -> Final* { return p; }

// CHECK:STDOUT: --- fail_base_bad_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc7_18.1: type = struct_type {}
// CHECK:STDOUT:   %.loc15: type = struct_type {.base: <error>}
// CHECK:STDOUT:   %.loc22: type = struct_type {.base: i32}
// CHECK:STDOUT:   %.loc17: type = ptr_type {.base: i32}
// CHECK:STDOUT:   %.loc32_22: type = tuple_type (type)
// CHECK:STDOUT:   %.loc32_23.1: type = tuple_type (Base)
// CHECK:STDOUT:   %.loc7_18.2: type = tuple_type ()
// CHECK:STDOUT:   %.loc7_1: type = ptr_type {}
// CHECK:STDOUT:   %.loc32_23.2: type = tuple_type ({}*)
// CHECK:STDOUT:   %.loc33_1.1: type = struct_type {.base: (Base,)}
// CHECK:STDOUT:   %.loc33_1.2: type = struct_type {.base: ({}*,)}
// CHECK:STDOUT:   %.loc28: type = ptr_type {.base: (Base,)}
// CHECK:STDOUT:   %.loc43: type = ptr_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc44_1.1: type = struct_type {.base: {.a: i32, .b: i32}}
// CHECK:STDOUT:   %.loc44_1.2: type = struct_type {.base: {.a: i32, .b: i32}*}
// CHECK:STDOUT:   %.loc39: type = ptr_type {.base: {.a: i32, .b: i32}}
// CHECK:STDOUT:   %.loc50: type = ptr_type {.base: <error>}
// CHECK:STDOUT:   %.loc70_1.1: type = struct_type {.base: Final}
// CHECK:STDOUT:   %.loc70_1.2: type = struct_type {.base: {}*}
// CHECK:STDOUT:   %.loc65: type = ptr_type {.base: Final}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Base = %Base.decl, .Final = %Final.decl, .DeriveFromNonType = %DeriveFromNonType.decl, .DeriveFromi32 = %DeriveFromi32.decl, .ConvertToBadBasei32 = %ConvertToBadBasei32, .DeriveFromTuple = %DeriveFromTuple.decl, .ConvertToBadBaseTuple = %ConvertToBadBaseTuple, .DeriveFromStruct = %DeriveFromStruct.decl, .ConvertToBadBaseStruct = %ConvertToBadBaseStruct, .Incomplete = %Incomplete.decl, .DeriveFromIncomplete = %DeriveFromIncomplete.decl, .ConvertToBadBaseIncomplete = %ConvertToBadBaseIncomplete, .DeriveFromFinal = %DeriveFromFinal.decl, .ConvertToBadBaseFinal = %ConvertToBadBaseFinal}
// CHECK:STDOUT:   %Base.decl = class_decl @Base, ()
// CHECK:STDOUT:   %Base: type = class_type @Base
// CHECK:STDOUT:   %Final.decl = class_decl @Final, ()
// CHECK:STDOUT:   %Final: type = class_type @Final
// CHECK:STDOUT:   %DeriveFromNonType.decl = class_decl @DeriveFromNonType, ()
// CHECK:STDOUT:   %DeriveFromNonType: type = class_type @DeriveFromNonType
// CHECK:STDOUT:   %DeriveFromi32.decl = class_decl @DeriveFromi32, ()
// CHECK:STDOUT:   %DeriveFromi32: type = class_type @DeriveFromi32
// CHECK:STDOUT:   %ConvertToBadBasei32: <function> = fn_decl @ConvertToBadBasei32
// CHECK:STDOUT:   %DeriveFromTuple.decl = class_decl @DeriveFromTuple, ()
// CHECK:STDOUT:   %DeriveFromTuple: type = class_type @DeriveFromTuple
// CHECK:STDOUT:   %ConvertToBadBaseTuple: <function> = fn_decl @ConvertToBadBaseTuple
// CHECK:STDOUT:   %DeriveFromStruct.decl = class_decl @DeriveFromStruct, ()
// CHECK:STDOUT:   %DeriveFromStruct: type = class_type @DeriveFromStruct
// CHECK:STDOUT:   %ConvertToBadBaseStruct: <function> = fn_decl @ConvertToBadBaseStruct
// CHECK:STDOUT:   %Incomplete.decl = class_decl @Incomplete, ()
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete
// CHECK:STDOUT:   %DeriveFromIncomplete.decl = class_decl @DeriveFromIncomplete, ()
// CHECK:STDOUT:   %DeriveFromIncomplete: type = class_type @DeriveFromIncomplete
// CHECK:STDOUT:   %ConvertToBadBaseIncomplete: <function> = fn_decl @ConvertToBadBaseIncomplete
// CHECK:STDOUT:   %DeriveFromFinal.decl = class_decl @DeriveFromFinal, ()
// CHECK:STDOUT:   %DeriveFromFinal: type = class_type @DeriveFromFinal
// CHECK:STDOUT:   %ConvertToBadBaseFinal: <function> = fn_decl @ConvertToBadBaseFinal
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Final {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromNonType {
// CHECK:STDOUT:   %.loc14_16: i32 = int_literal 32
// CHECK:STDOUT:   %.loc14_18.1: type = unbound_element_type DeriveFromNonType, <error>
// CHECK:STDOUT:   %.loc14_18.2: <unbound element of class DeriveFromNonType> = base_decl <error>, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc14_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromi32 {
// CHECK:STDOUT:   %.loc21_19.1: type = unbound_element_type DeriveFromi32, i32
// CHECK:STDOUT:   %.loc21_19.2: <unbound element of class DeriveFromi32> = base_decl i32, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc21_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromTuple {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base
// CHECK:STDOUT:   %.loc32_22: (type,) = tuple_literal (%Base.ref)
// CHECK:STDOUT:   %.loc32_23.1: type = converted %.loc32_22, constants.%.loc32_23.1
// CHECK:STDOUT:   %.loc32_23.2: type = unbound_element_type DeriveFromTuple, (Base,)
// CHECK:STDOUT:   %.loc32_23.3: <unbound element of class DeriveFromTuple> = base_decl (Base,), element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc32_23.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromStruct {
// CHECK:STDOUT:   %.loc43_33: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc43_34.1: type = unbound_element_type DeriveFromStruct, {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc43_34.2: <unbound element of class DeriveFromStruct> = base_decl {.a: i32, .b: i32}, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc43_34.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromIncomplete {
// CHECK:STDOUT:   %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete
// CHECK:STDOUT:   %.loc57_26.1: type = unbound_element_type DeriveFromIncomplete, <error>
// CHECK:STDOUT:   %.loc57_26.2: <unbound element of class DeriveFromIncomplete> = base_decl <error>, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc57_26.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromFinal {
// CHECK:STDOUT:   %Final.ref: type = name_ref Final, file.%Final
// CHECK:STDOUT:   %.loc69_21.1: type = unbound_element_type DeriveFromFinal, Final
// CHECK:STDOUT:   %.loc69_21.2: <unbound element of class DeriveFromFinal> = base_decl Final, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc69_21.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBasei32(%p: DeriveFromi32*) -> i32* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromi32* = name_ref p, %p
// CHECK:STDOUT:   %.loc26_61.1: ref DeriveFromi32 = deref %p.ref
// CHECK:STDOUT:   %.loc26_61.2: ref i32 = class_element_access %.loc26_61.1, element0
// CHECK:STDOUT:   %.loc26_61.3: i32* = address_of %.loc26_61.2
// CHECK:STDOUT:   %.loc26_61.4: i32* = converted %p.ref, %.loc26_61.3
// CHECK:STDOUT:   return %.loc26_61.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseTuple(%p: DeriveFromTuple*) -> (Base,)* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromTuple* = name_ref p, %p
// CHECK:STDOUT:   %.loc35_69.1: ref DeriveFromTuple = deref %p.ref
// CHECK:STDOUT:   %.loc35_69.2: ref (Base,) = class_element_access %.loc35_69.1, element0
// CHECK:STDOUT:   %.loc35_69.3: (Base,)* = address_of %.loc35_69.2
// CHECK:STDOUT:   %.loc35_69.4: (Base,)* = converted %p.ref, %.loc35_69.3
// CHECK:STDOUT:   return %.loc35_69.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseStruct(%p: DeriveFromStruct*) -> {.a: i32, .b: i32}* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromStruct* = name_ref p, %p
// CHECK:STDOUT:   %.loc46_82.1: ref DeriveFromStruct = deref %p.ref
// CHECK:STDOUT:   %.loc46_82.2: ref {.a: i32, .b: i32} = class_element_access %.loc46_82.1, element0
// CHECK:STDOUT:   %.loc46_82.3: {.a: i32, .b: i32}* = address_of %.loc46_82.2
// CHECK:STDOUT:   %.loc46_82.4: {.a: i32, .b: i32}* = converted %p.ref, %.loc46_82.3
// CHECK:STDOUT:   return %.loc46_82.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseIncomplete(%p: DeriveFromIncomplete*) -> Incomplete* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromIncomplete* = name_ref p, %p
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseFinal(%p: DeriveFromFinal*) -> Final* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromFinal* = name_ref p, %p
// CHECK:STDOUT:   %.loc72_67.1: ref DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %.loc72_67.2: ref Final = class_element_access %.loc72_67.1, element0
// CHECK:STDOUT:   %.loc72_67.3: Final* = address_of %.loc72_67.2
// CHECK:STDOUT:   %.loc72_67.4: Final* = converted %p.ref, %.loc72_67.3
// CHECK:STDOUT:   return %.loc72_67.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
