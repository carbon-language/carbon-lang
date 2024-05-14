// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class Base {}
class Final {
  var a: i32;
}

class DeriveFromError {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:16: ERROR: Name `error` not found.
  // CHECK:STDERR:   extend base: error;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  extend base: error;
}

// This should not produce an error.
fn AccessMemberWithInvalidBaseError(p: DeriveFromError*) -> i32 { return (*p).n; }

class DeriveFromNonType {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:16: ERROR: Cannot implicitly convert from `i32` to `type`.
  // CHECK:STDERR:   extend base: 32;
  // CHECK:STDERR:                ^~
  // CHECK:STDERR:
  extend base: 32;
}

fn AccessMemberWithInvalidBasNonType(p: DeriveFromNonType*) -> i32 { return (*p).n; }

class DeriveFromi32 {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:16: ERROR: Deriving from final type `i32`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   extend base: i32;
  // CHECK:STDERR:                ^~~
  // CHECK:STDERR:
  extend base: i32;
}

// It's not really important whether this conversion produces an error or not,
// but it shouldn't crash.
// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:53: ERROR: Cannot implicitly convert from `DeriveFromi32*` to `i32*`.
// CHECK:STDERR: fn ConvertToBadBasei32(p: DeriveFromi32*) -> i32* { return p; }
// CHECK:STDERR:                                                     ^~~~~~~~~
// CHECK:STDERR:
fn ConvertToBadBasei32(p: DeriveFromi32*) -> i32* { return p; }

fn AccessMemberWithInvalidBasei32(p: DeriveFromi32*) -> i32 { return (*p).n; }

class DeriveFromTuple {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:16: ERROR: Deriving from final type `(Base,)`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   extend base: (Base,);
  // CHECK:STDERR:                ^~~~~~~
  // CHECK:STDERR:
  extend base: (Base,);
}

// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:61: ERROR: Cannot implicitly convert from `DeriveFromTuple*` to `(Base,)*`.
// CHECK:STDERR: fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }
// CHECK:STDERR:                                                             ^~~~~~~~~
// CHECK:STDERR:
fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }

fn AccessMemberWithInvalidBaseTuple(p: DeriveFromTuple*) -> i32 { return (*p).n; }

// TODO: Should we allow this?
// We do allow `{.base = {.a: i32, .b: i32}}`.
class DeriveFromStruct {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:16: ERROR: Deriving from final type `{.a: i32, .b: i32}`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   extend base: {.a: i32, .b: i32};
  // CHECK:STDERR:                ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: {.a: i32, .b: i32};
}

// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:74: ERROR: Cannot implicitly convert from `DeriveFromStruct*` to `{.a: i32, .b: i32}*`.
// CHECK:STDERR: fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
// CHECK:STDERR:
fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }

// It would be OK to reject this if we start actually looking in the struct type.
fn AccessMemberWithInvalidBaseStruct(p: DeriveFromStruct*) -> i32 { return (*p).n; }

base class Incomplete;

class DeriveFromIncomplete {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+7]]:16: ERROR: Base `Incomplete` is an incomplete type.
  // CHECK:STDERR:   extend base: Incomplete;
  // CHECK:STDERR:                ^~~~~~~~~~
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE-6]]:1: Class was forward declared here.
  // CHECK:STDERR: base class Incomplete;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: Incomplete;
}

// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:74: ERROR: Cannot implicitly convert from `DeriveFromIncomplete*` to `Incomplete*`.
// CHECK:STDERR: fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
// CHECK:STDERR:
fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }

fn AccessMemberWithInvalidBaseIncomplete(p: DeriveFromIncomplete*) -> i32 { return (*p).n; }

class DeriveFromFinal {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:16: ERROR: Deriving from final type `Final`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   extend base: Final;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  extend base: Final;
}

// For error recovery purposes, we derive from the final type anyway.
fn ConvertToBadBaseFinal(p: DeriveFromFinal*) -> Final* {
  return p;
}

fn AccessMemberWithInvalidBaseFinal_WithMember(p: DeriveFromFinal*) -> i32 {
  return (*p).a;
}

fn AccessMemberWithInvalidBaseFinal_NoMember(p: DeriveFromFinal*) -> i32 {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:10: ERROR: Name `b` not found.
  // CHECK:STDERR:   return (*p).b;
  // CHECK:STDERR:          ^~~~~~
  return (*p).b;
}

// CHECK:STDOUT: --- fail_base_bad_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %Final: type = class_type @Final [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type Final, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %DeriveFromError: type = class_type @DeriveFromError [template]
// CHECK:STDOUT:   %.4: type = ptr_type DeriveFromError [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseError: type = fn_type @AccessMemberWithInvalidBaseError [template]
// CHECK:STDOUT:   %.5: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: AccessMemberWithInvalidBaseError = struct_value () [template]
// CHECK:STDOUT:   %DeriveFromNonType: type = class_type @DeriveFromNonType [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 32 [template]
// CHECK:STDOUT:   %.7: type = ptr_type DeriveFromNonType [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBasNonType: type = fn_type @AccessMemberWithInvalidBasNonType [template]
// CHECK:STDOUT:   %struct.2: AccessMemberWithInvalidBasNonType = struct_value () [template]
// CHECK:STDOUT:   %DeriveFromi32: type = class_type @DeriveFromi32 [template]
// CHECK:STDOUT:   %.8: type = ptr_type DeriveFromi32 [template]
// CHECK:STDOUT:   %.9: type = ptr_type i32 [template]
// CHECK:STDOUT:   %ConvertToBadBasei32: type = fn_type @ConvertToBadBasei32 [template]
// CHECK:STDOUT:   %struct.3: ConvertToBadBasei32 = struct_value () [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBasei32: type = fn_type @AccessMemberWithInvalidBasei32 [template]
// CHECK:STDOUT:   %struct.4: AccessMemberWithInvalidBasei32 = struct_value () [template]
// CHECK:STDOUT:   %DeriveFromTuple: type = class_type @DeriveFromTuple [template]
// CHECK:STDOUT:   %.10: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.11: type = tuple_type (Base) [template]
// CHECK:STDOUT:   %.12: type = ptr_type {} [template]
// CHECK:STDOUT:   %.13: type = tuple_type ({}*) [template]
// CHECK:STDOUT:   %.14: type = ptr_type DeriveFromTuple [template]
// CHECK:STDOUT:   %.15: type = ptr_type (Base,) [template]
// CHECK:STDOUT:   %ConvertToBadBaseTuple: type = fn_type @ConvertToBadBaseTuple [template]
// CHECK:STDOUT:   %struct.5: ConvertToBadBaseTuple = struct_value () [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseTuple: type = fn_type @AccessMemberWithInvalidBaseTuple [template]
// CHECK:STDOUT:   %struct.6: AccessMemberWithInvalidBaseTuple = struct_value () [template]
// CHECK:STDOUT:   %DeriveFromStruct: type = class_type @DeriveFromStruct [template]
// CHECK:STDOUT:   %.16: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.17: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.18: type = ptr_type DeriveFromStruct [template]
// CHECK:STDOUT:   %ConvertToBadBaseStruct: type = fn_type @ConvertToBadBaseStruct [template]
// CHECK:STDOUT:   %struct.7: ConvertToBadBaseStruct = struct_value () [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseStruct: type = fn_type @AccessMemberWithInvalidBaseStruct [template]
// CHECK:STDOUT:   %struct.8: AccessMemberWithInvalidBaseStruct = struct_value () [template]
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
// CHECK:STDOUT:   %DeriveFromIncomplete: type = class_type @DeriveFromIncomplete [template]
// CHECK:STDOUT:   %.19: type = ptr_type DeriveFromIncomplete [template]
// CHECK:STDOUT:   %.20: type = ptr_type Incomplete [template]
// CHECK:STDOUT:   %ConvertToBadBaseIncomplete: type = fn_type @ConvertToBadBaseIncomplete [template]
// CHECK:STDOUT:   %struct.9: ConvertToBadBaseIncomplete = struct_value () [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseIncomplete: type = fn_type @AccessMemberWithInvalidBaseIncomplete [template]
// CHECK:STDOUT:   %struct.10: AccessMemberWithInvalidBaseIncomplete = struct_value () [template]
// CHECK:STDOUT:   %DeriveFromFinal: type = class_type @DeriveFromFinal [template]
// CHECK:STDOUT:   %.21: type = ptr_type {.a: i32} [template]
// CHECK:STDOUT:   %.22: type = unbound_element_type DeriveFromFinal, Final [template]
// CHECK:STDOUT:   %.23: type = struct_type {.base: Final} [template]
// CHECK:STDOUT:   %.24: type = ptr_type DeriveFromFinal [template]
// CHECK:STDOUT:   %.25: type = ptr_type Final [template]
// CHECK:STDOUT:   %ConvertToBadBaseFinal: type = fn_type @ConvertToBadBaseFinal [template]
// CHECK:STDOUT:   %struct.11: ConvertToBadBaseFinal = struct_value () [template]
// CHECK:STDOUT:   %.26: type = struct_type {.base: {.a: i32}*} [template]
// CHECK:STDOUT:   %.27: type = ptr_type {.base: Final} [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_WithMember: type = fn_type @AccessMemberWithInvalidBaseFinal_WithMember [template]
// CHECK:STDOUT:   %struct.12: AccessMemberWithInvalidBaseFinal_WithMember = struct_value () [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_NoMember: type = fn_type @AccessMemberWithInvalidBaseFinal_NoMember [template]
// CHECK:STDOUT:   %struct.13: AccessMemberWithInvalidBaseFinal_NoMember = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Final = %Final.decl
// CHECK:STDOUT:     .DeriveFromError = %DeriveFromError.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBaseError = %AccessMemberWithInvalidBaseError.decl
// CHECK:STDOUT:     .DeriveFromNonType = %DeriveFromNonType.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBasNonType = %AccessMemberWithInvalidBasNonType.decl
// CHECK:STDOUT:     .DeriveFromi32 = %DeriveFromi32.decl
// CHECK:STDOUT:     .ConvertToBadBasei32 = %ConvertToBadBasei32.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBasei32 = %AccessMemberWithInvalidBasei32.decl
// CHECK:STDOUT:     .DeriveFromTuple = %DeriveFromTuple.decl
// CHECK:STDOUT:     .ConvertToBadBaseTuple = %ConvertToBadBaseTuple.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBaseTuple = %AccessMemberWithInvalidBaseTuple.decl
// CHECK:STDOUT:     .DeriveFromStruct = %DeriveFromStruct.decl
// CHECK:STDOUT:     .ConvertToBadBaseStruct = %ConvertToBadBaseStruct.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBaseStruct = %AccessMemberWithInvalidBaseStruct.decl
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:     .DeriveFromIncomplete = %DeriveFromIncomplete.decl
// CHECK:STDOUT:     .ConvertToBadBaseIncomplete = %ConvertToBadBaseIncomplete.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBaseIncomplete = %AccessMemberWithInvalidBaseIncomplete.decl
// CHECK:STDOUT:     .DeriveFromFinal = %DeriveFromFinal.decl
// CHECK:STDOUT:     .ConvertToBadBaseFinal = %ConvertToBadBaseFinal.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBaseFinal_WithMember = %AccessMemberWithInvalidBaseFinal_WithMember.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBaseFinal_NoMember = %AccessMemberWithInvalidBaseFinal_NoMember.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %Final.decl: type = class_decl @Final [template = constants.%Final] {}
// CHECK:STDOUT:   %DeriveFromError.decl: type = class_decl @DeriveFromError [template = constants.%DeriveFromError] {}
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseError.decl: AccessMemberWithInvalidBaseError = fn_decl @AccessMemberWithInvalidBaseError [template = constants.%struct.1] {
// CHECK:STDOUT:     %DeriveFromError.ref: type = name_ref DeriveFromError, %DeriveFromError.decl [template = constants.%DeriveFromError]
// CHECK:STDOUT:     %.loc21: type = ptr_type DeriveFromError [template = constants.%.4]
// CHECK:STDOUT:     %p.loc21_37.1: DeriveFromError* = param p
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseError.%p: DeriveFromError* = bind_name p, %p.loc21_37.1
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseError.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeriveFromNonType.decl: type = class_decl @DeriveFromNonType [template = constants.%DeriveFromNonType] {}
// CHECK:STDOUT:   %AccessMemberWithInvalidBasNonType.decl: AccessMemberWithInvalidBasNonType = fn_decl @AccessMemberWithInvalidBasNonType [template = constants.%struct.2] {
// CHECK:STDOUT:     %DeriveFromNonType.ref: type = name_ref DeriveFromNonType, %DeriveFromNonType.decl [template = constants.%DeriveFromNonType]
// CHECK:STDOUT:     %.loc31: type = ptr_type DeriveFromNonType [template = constants.%.7]
// CHECK:STDOUT:     %p.loc31_38.1: DeriveFromNonType* = param p
// CHECK:STDOUT:     @AccessMemberWithInvalidBasNonType.%p: DeriveFromNonType* = bind_name p, %p.loc31_38.1
// CHECK:STDOUT:     @AccessMemberWithInvalidBasNonType.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeriveFromi32.decl: type = class_decl @DeriveFromi32 [template = constants.%DeriveFromi32] {}
// CHECK:STDOUT:   %ConvertToBadBasei32.decl: ConvertToBadBasei32 = fn_decl @ConvertToBadBasei32 [template = constants.%struct.3] {
// CHECK:STDOUT:     %DeriveFromi32.ref.loc47: type = name_ref DeriveFromi32, %DeriveFromi32.decl [template = constants.%DeriveFromi32]
// CHECK:STDOUT:     %.loc47_40: type = ptr_type DeriveFromi32 [template = constants.%.8]
// CHECK:STDOUT:     %p.loc47_24.1: DeriveFromi32* = param p
// CHECK:STDOUT:     @ConvertToBadBasei32.%p: DeriveFromi32* = bind_name p, %p.loc47_24.1
// CHECK:STDOUT:     %.loc47_49: type = ptr_type i32 [template = constants.%.9]
// CHECK:STDOUT:     @ConvertToBadBasei32.%return: ref i32* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBasei32.decl: AccessMemberWithInvalidBasei32 = fn_decl @AccessMemberWithInvalidBasei32 [template = constants.%struct.4] {
// CHECK:STDOUT:     %DeriveFromi32.ref.loc49: type = name_ref DeriveFromi32, %DeriveFromi32.decl [template = constants.%DeriveFromi32]
// CHECK:STDOUT:     %.loc49: type = ptr_type DeriveFromi32 [template = constants.%.8]
// CHECK:STDOUT:     %p.loc49_35.1: DeriveFromi32* = param p
// CHECK:STDOUT:     @AccessMemberWithInvalidBasei32.%p: DeriveFromi32* = bind_name p, %p.loc49_35.1
// CHECK:STDOUT:     @AccessMemberWithInvalidBasei32.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeriveFromTuple.decl: type = class_decl @DeriveFromTuple [template = constants.%DeriveFromTuple] {}
// CHECK:STDOUT:   %ConvertToBadBaseTuple.decl: ConvertToBadBaseTuple = fn_decl @ConvertToBadBaseTuple [template = constants.%struct.5] {
// CHECK:STDOUT:     %DeriveFromTuple.ref.loc63: type = name_ref DeriveFromTuple, %DeriveFromTuple.decl [template = constants.%DeriveFromTuple]
// CHECK:STDOUT:     %.loc63_44: type = ptr_type DeriveFromTuple [template = constants.%.14]
// CHECK:STDOUT:     %p.loc63_26.1: DeriveFromTuple* = param p
// CHECK:STDOUT:     @ConvertToBadBaseTuple.%p: DeriveFromTuple* = bind_name p, %p.loc63_26.1
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, %Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %.loc63_56: (type,) = tuple_literal (%Base.ref)
// CHECK:STDOUT:     %.loc63_57.1: type = converted %.loc63_56, constants.%.11 [template = constants.%.11]
// CHECK:STDOUT:     %.loc63_57.2: type = ptr_type (Base,) [template = constants.%.15]
// CHECK:STDOUT:     @ConvertToBadBaseTuple.%return: ref (Base,)* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseTuple.decl: AccessMemberWithInvalidBaseTuple = fn_decl @AccessMemberWithInvalidBaseTuple [template = constants.%struct.6] {
// CHECK:STDOUT:     %DeriveFromTuple.ref.loc65: type = name_ref DeriveFromTuple, %DeriveFromTuple.decl [template = constants.%DeriveFromTuple]
// CHECK:STDOUT:     %.loc65: type = ptr_type DeriveFromTuple [template = constants.%.14]
// CHECK:STDOUT:     %p.loc65_37.1: DeriveFromTuple* = param p
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseTuple.%p: DeriveFromTuple* = bind_name p, %p.loc65_37.1
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseTuple.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeriveFromStruct.decl: type = class_decl @DeriveFromStruct [template = constants.%DeriveFromStruct] {}
// CHECK:STDOUT:   %ConvertToBadBaseStruct.decl: ConvertToBadBaseStruct = fn_decl @ConvertToBadBaseStruct [template = constants.%struct.7] {
// CHECK:STDOUT:     %DeriveFromStruct.ref.loc81: type = name_ref DeriveFromStruct, %DeriveFromStruct.decl [template = constants.%DeriveFromStruct]
// CHECK:STDOUT:     %.loc81_46: type = ptr_type DeriveFromStruct [template = constants.%.18]
// CHECK:STDOUT:     %p.loc81_27.1: DeriveFromStruct* = param p
// CHECK:STDOUT:     @ConvertToBadBaseStruct.%p: DeriveFromStruct* = bind_name p, %p.loc81_27.1
// CHECK:STDOUT:     %.loc81_69: type = struct_type {.a: i32, .b: i32} [template = constants.%.16]
// CHECK:STDOUT:     %.loc81_70: type = ptr_type {.a: i32, .b: i32} [template = constants.%.17]
// CHECK:STDOUT:     @ConvertToBadBaseStruct.%return: ref {.a: i32, .b: i32}* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseStruct.decl: AccessMemberWithInvalidBaseStruct = fn_decl @AccessMemberWithInvalidBaseStruct [template = constants.%struct.8] {
// CHECK:STDOUT:     %DeriveFromStruct.ref.loc84: type = name_ref DeriveFromStruct, %DeriveFromStruct.decl [template = constants.%DeriveFromStruct]
// CHECK:STDOUT:     %.loc84: type = ptr_type DeriveFromStruct [template = constants.%.18]
// CHECK:STDOUT:     %p.loc84_38.1: DeriveFromStruct* = param p
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseStruct.%p: DeriveFromStruct* = bind_name p, %p.loc84_38.1
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseStruct.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Incomplete.decl: type = class_decl @Incomplete [template = constants.%Incomplete] {}
// CHECK:STDOUT:   %DeriveFromIncomplete.decl: type = class_decl @DeriveFromIncomplete [template = constants.%DeriveFromIncomplete] {}
// CHECK:STDOUT:   %ConvertToBadBaseIncomplete.decl: ConvertToBadBaseIncomplete = fn_decl @ConvertToBadBaseIncomplete [template = constants.%struct.9] {
// CHECK:STDOUT:     %DeriveFromIncomplete.ref.loc103: type = name_ref DeriveFromIncomplete, %DeriveFromIncomplete.decl [template = constants.%DeriveFromIncomplete]
// CHECK:STDOUT:     %.loc103_54: type = ptr_type DeriveFromIncomplete [template = constants.%.19]
// CHECK:STDOUT:     %p.loc103_31.1: DeriveFromIncomplete* = param p
// CHECK:STDOUT:     @ConvertToBadBaseIncomplete.%p: DeriveFromIncomplete* = bind_name p, %p.loc103_31.1
// CHECK:STDOUT:     %Incomplete.ref: type = name_ref Incomplete, %Incomplete.decl [template = constants.%Incomplete]
// CHECK:STDOUT:     %.loc103_70: type = ptr_type Incomplete [template = constants.%.20]
// CHECK:STDOUT:     @ConvertToBadBaseIncomplete.%return: ref Incomplete* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseIncomplete.decl: AccessMemberWithInvalidBaseIncomplete = fn_decl @AccessMemberWithInvalidBaseIncomplete [template = constants.%struct.10] {
// CHECK:STDOUT:     %DeriveFromIncomplete.ref.loc105: type = name_ref DeriveFromIncomplete, %DeriveFromIncomplete.decl [template = constants.%DeriveFromIncomplete]
// CHECK:STDOUT:     %.loc105: type = ptr_type DeriveFromIncomplete [template = constants.%.19]
// CHECK:STDOUT:     %p.loc105_42.1: DeriveFromIncomplete* = param p
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseIncomplete.%p: DeriveFromIncomplete* = bind_name p, %p.loc105_42.1
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseIncomplete.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeriveFromFinal.decl: type = class_decl @DeriveFromFinal [template = constants.%DeriveFromFinal] {}
// CHECK:STDOUT:   %ConvertToBadBaseFinal.decl: ConvertToBadBaseFinal = fn_decl @ConvertToBadBaseFinal [template = constants.%struct.11] {
// CHECK:STDOUT:     %DeriveFromFinal.ref.loc116: type = name_ref DeriveFromFinal, %DeriveFromFinal.decl [template = constants.%DeriveFromFinal]
// CHECK:STDOUT:     %.loc116_44: type = ptr_type DeriveFromFinal [template = constants.%.24]
// CHECK:STDOUT:     %p.loc116_26.1: DeriveFromFinal* = param p
// CHECK:STDOUT:     @ConvertToBadBaseFinal.%p: DeriveFromFinal* = bind_name p, %p.loc116_26.1
// CHECK:STDOUT:     %Final.ref: type = name_ref Final, %Final.decl [template = constants.%Final]
// CHECK:STDOUT:     %.loc116_55: type = ptr_type Final [template = constants.%.25]
// CHECK:STDOUT:     @ConvertToBadBaseFinal.%return: ref Final* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_WithMember.decl: AccessMemberWithInvalidBaseFinal_WithMember = fn_decl @AccessMemberWithInvalidBaseFinal_WithMember [template = constants.%struct.12] {
// CHECK:STDOUT:     %DeriveFromFinal.ref.loc120: type = name_ref DeriveFromFinal, %DeriveFromFinal.decl [template = constants.%DeriveFromFinal]
// CHECK:STDOUT:     %.loc120: type = ptr_type DeriveFromFinal [template = constants.%.24]
// CHECK:STDOUT:     %p.loc120_48.1: DeriveFromFinal* = param p
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseFinal_WithMember.%p: DeriveFromFinal* = bind_name p, %p.loc120_48.1
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseFinal_WithMember.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_NoMember.decl: AccessMemberWithInvalidBaseFinal_NoMember = fn_decl @AccessMemberWithInvalidBaseFinal_NoMember [template = constants.%struct.13] {
// CHECK:STDOUT:     %DeriveFromFinal.ref.loc124: type = name_ref DeriveFromFinal, %DeriveFromFinal.decl [template = constants.%DeriveFromFinal]
// CHECK:STDOUT:     %.loc124: type = ptr_type DeriveFromFinal [template = constants.%.24]
// CHECK:STDOUT:     %p.loc124_46.1: DeriveFromFinal* = param p
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseFinal_NoMember.%p: DeriveFromFinal* = bind_name p, %p.loc124_46.1
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseFinal_NoMember.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Final {
// CHECK:STDOUT:   %.loc9: <unbound element of class Final> = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Final
// CHECK:STDOUT:   .a = %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromError {
// CHECK:STDOUT:   %error.ref: <error> = name_ref error, <error> [template = <error>]
// CHECK:STDOUT:   %.loc17: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromError
// CHECK:STDOUT:   .base = %.loc17
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromNonType {
// CHECK:STDOUT:   %.loc28_16: i32 = int_literal 32 [template = constants.%.6]
// CHECK:STDOUT:   %.loc28_18: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromNonType
// CHECK:STDOUT:   .base = %.loc28_18
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromi32 {
// CHECK:STDOUT:   %.loc38: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromi32
// CHECK:STDOUT:   .base = %.loc38
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromTuple {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc56_22.1: (type,) = tuple_literal (%Base.ref)
// CHECK:STDOUT:   %.loc56_22.2: type = converted %.loc56_22.1, constants.%.11 [template = constants.%.11]
// CHECK:STDOUT:   %.loc56_23: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromTuple
// CHECK:STDOUT:   .base = %.loc56_23
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromStruct {
// CHECK:STDOUT:   %.loc74_33: type = struct_type {.a: i32, .b: i32} [template = constants.%.16]
// CHECK:STDOUT:   %.loc74_34: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromStruct
// CHECK:STDOUT:   .base = %.loc74_34
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromIncomplete {
// CHECK:STDOUT:   %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl [template = constants.%Incomplete]
// CHECK:STDOUT:   %.loc96: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromIncomplete
// CHECK:STDOUT:   .base = %.loc96
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromFinal {
// CHECK:STDOUT:   %Final.ref: type = name_ref Final, file.%Final.decl [template = constants.%Final]
// CHECK:STDOUT:   %.loc112: <unbound element of class DeriveFromFinal> = base_decl Final, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromFinal
// CHECK:STDOUT:   .base = %.loc112
// CHECK:STDOUT:   extend name_scope3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseError(%p: DeriveFromError*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromError* = name_ref p, %p
// CHECK:STDOUT:   %.loc21: ref DeriveFromError = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBasNonType(%p: DeriveFromNonType*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromNonType* = name_ref p, %p
// CHECK:STDOUT:   %.loc31: ref DeriveFromNonType = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBasei32(%p: DeriveFromi32*) -> i32* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromi32* = name_ref p, %p
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBasei32(%p: DeriveFromi32*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromi32* = name_ref p, %p
// CHECK:STDOUT:   %.loc49: ref DeriveFromi32 = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseTuple(%p: DeriveFromTuple*) -> (Base,)* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromTuple* = name_ref p, %p
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseTuple(%p: DeriveFromTuple*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromTuple* = name_ref p, %p
// CHECK:STDOUT:   %.loc65: ref DeriveFromTuple = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseStruct(%p: DeriveFromStruct*) -> {.a: i32, .b: i32}* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromStruct* = name_ref p, %p
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseStruct(%p: DeriveFromStruct*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromStruct* = name_ref p, %p
// CHECK:STDOUT:   %.loc84: ref DeriveFromStruct = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseIncomplete(%p: DeriveFromIncomplete*) -> Incomplete* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromIncomplete* = name_ref p, %p
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseIncomplete(%p: DeriveFromIncomplete*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromIncomplete* = name_ref p, %p
// CHECK:STDOUT:   %.loc105: ref DeriveFromIncomplete = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseFinal(%p: DeriveFromFinal*) -> Final* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromFinal* = name_ref p, %p
// CHECK:STDOUT:   %.loc117_11.1: ref DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %.loc117_11.2: ref Final = class_element_access %.loc117_11.1, element0
// CHECK:STDOUT:   %.loc117_11.3: Final* = addr_of %.loc117_11.2
// CHECK:STDOUT:   %.loc117_11.4: Final* = converted %p.ref, %.loc117_11.3
// CHECK:STDOUT:   return %.loc117_11.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseFinal_WithMember(%p: DeriveFromFinal*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromFinal* = name_ref p, %p
// CHECK:STDOUT:   %.loc121_11: ref DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %a.ref: <unbound element of class Final> = name_ref a, @Final.%.loc9 [template = @Final.%.loc9]
// CHECK:STDOUT:   %.loc121_14.1: ref Final = class_element_access %.loc121_11, element0
// CHECK:STDOUT:   %.loc121_14.2: ref Final = converted %.loc121_11, %.loc121_14.1
// CHECK:STDOUT:   %.loc121_14.3: ref i32 = class_element_access %.loc121_14.2, element0
// CHECK:STDOUT:   %.loc121_14.4: i32 = bind_value %.loc121_14.3
// CHECK:STDOUT:   return %.loc121_14.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseFinal_NoMember(%p: DeriveFromFinal*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromFinal* = name_ref p, %p
// CHECK:STDOUT:   %.loc128: ref DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %b.ref: <error> = name_ref b, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
