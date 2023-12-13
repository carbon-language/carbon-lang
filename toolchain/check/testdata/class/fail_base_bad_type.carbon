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
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:16: ERROR: Name `error` not found.
  // CHECK:STDERR:   extend base: error;
  // CHECK:STDERR:                ^~~~~
  extend base: error;
}

// This should not produce an error.
fn AccessMemberWithInvalidBaseError(p: DeriveFromError*) -> i32 { return (*p).n; }

class DeriveFromNonType {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:18: ERROR: Cannot implicitly convert from `i32` to `type`.
  // CHECK:STDERR:   extend base: 32;
  // CHECK:STDERR:                  ^
  extend base: 32;
}

fn AccessMemberWithInvalidBasNonType(p: DeriveFromNonType*) -> i32 { return (*p).n; }

class DeriveFromi32 {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:19: ERROR: Deriving from final type `i32`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   extend base: i32;
  // CHECK:STDERR:                   ^
  extend base: i32;
}

// It's not really important whether this conversion produces an error or not,
// but it shouldn't crash.
// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:61: ERROR: Cannot implicitly convert from `DeriveFromi32*` to `i32*`.
// CHECK:STDERR: fn ConvertToBadBasei32(p: DeriveFromi32*) -> i32* { return p; }
// CHECK:STDERR:                                                             ^
fn ConvertToBadBasei32(p: DeriveFromi32*) -> i32* { return p; }

fn AccessMemberWithInvalidBasei32(p: DeriveFromi32*) -> i32 { return (*p).n; }

class DeriveFromTuple {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:23: ERROR: Deriving from final type `(Base,)`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   extend base: (Base,);
  // CHECK:STDERR:                       ^
  extend base: (Base,);
}

// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:69: ERROR: Cannot implicitly convert from `DeriveFromTuple*` to `(Base,)*`.
// CHECK:STDERR: fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }
// CHECK:STDERR:                                                                     ^
fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }

fn AccessMemberWithInvalidBaseTuple(p: DeriveFromTuple*) -> i32 { return (*p).n; }

// TODO: Should we allow this?
// We do allow `{.base = {.a: i32, .b: i32}}`.
class DeriveFromStruct {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:34: ERROR: Deriving from final type `{.a: i32, .b: i32}`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   extend base: {.a: i32, .b: i32};
  // CHECK:STDERR:                                  ^
  extend base: {.a: i32, .b: i32};
}

// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:82: ERROR: Cannot implicitly convert from `DeriveFromStruct*` to `{.a: i32, .b: i32}*`.
// CHECK:STDERR: fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }
// CHECK:STDERR:                                                                                  ^
fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }

// It would be OK to reject this if we start actually looking in the struct type.
fn AccessMemberWithInvalidBaseStruct(p: DeriveFromStruct*) -> i32 { return (*p).n; }

base class Incomplete;

class DeriveFromIncomplete {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+6]]:26: ERROR: Base `Incomplete` is an incomplete type.
  // CHECK:STDERR:   extend base: Incomplete;
  // CHECK:STDERR:                          ^
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE-6]]:1: Class was forward declared here.
  // CHECK:STDERR: base class Incomplete;
  // CHECK:STDERR: ^~~~
  extend base: Incomplete;
}

// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:82: ERROR: Cannot implicitly convert from `DeriveFromIncomplete*` to `Incomplete*`.
// CHECK:STDERR: fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }
// CHECK:STDERR:                                                                                  ^
fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }

fn AccessMemberWithInvalidBaseIncomplete(p: DeriveFromIncomplete*) -> i32 { return (*p).n; }

class DeriveFromFinal {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:21: ERROR: Deriving from final type `Final`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   extend base: Final;
  // CHECK:STDERR:                     ^
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
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:14: ERROR: Name `b` not found.
  // CHECK:STDERR:   return (*p).b;
  // CHECK:STDERR:              ^
  return (*p).b;
}

// CHECK:STDOUT: --- fail_base_bad_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc7_18.1: type = struct_type {}
// CHECK:STDOUT:   %.loc10: type = struct_type {.a: i32}
// CHECK:STDOUT:   %.loc17: type = struct_type {.base: <error>}
// CHECK:STDOUT:   %.loc12: type = ptr_type {.base: <error>}
// CHECK:STDOUT:   %.loc51_22: type = tuple_type (type)
// CHECK:STDOUT:   %.loc51_23.1: type = tuple_type (Base)
// CHECK:STDOUT:   %.loc7_18.2: type = tuple_type ()
// CHECK:STDOUT:   %.loc7_1: type = ptr_type {}
// CHECK:STDOUT:   %.loc51_23.2: type = tuple_type ({}*)
// CHECK:STDOUT:   %.loc67: type = ptr_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc8: type = ptr_type {.a: i32}
// CHECK:STDOUT:   %.loc102_1.1: type = struct_type {.base: Final}
// CHECK:STDOUT:   %.loc102_1.2: type = struct_type {.base: {.a: i32}*}
// CHECK:STDOUT:   %.loc97: type = ptr_type {.base: Final}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Base = %Base.decl, .Final = %Final.decl, .DeriveFromError = %DeriveFromError.decl, .AccessMemberWithInvalidBaseError = %AccessMemberWithInvalidBaseError, .DeriveFromNonType = %DeriveFromNonType.decl, .AccessMemberWithInvalidBasNonType = %AccessMemberWithInvalidBasNonType, .DeriveFromi32 = %DeriveFromi32.decl, .ConvertToBadBasei32 = %ConvertToBadBasei32, .AccessMemberWithInvalidBasei32 = %AccessMemberWithInvalidBasei32, .DeriveFromTuple = %DeriveFromTuple.decl, .ConvertToBadBaseTuple = %ConvertToBadBaseTuple, .AccessMemberWithInvalidBaseTuple = %AccessMemberWithInvalidBaseTuple, .DeriveFromStruct = %DeriveFromStruct.decl, .ConvertToBadBaseStruct = %ConvertToBadBaseStruct, .AccessMemberWithInvalidBaseStruct = %AccessMemberWithInvalidBaseStruct, .Incomplete = %Incomplete.decl, .DeriveFromIncomplete = %DeriveFromIncomplete.decl, .ConvertToBadBaseIncomplete = %ConvertToBadBaseIncomplete, .AccessMemberWithInvalidBaseIncomplete = %AccessMemberWithInvalidBaseIncomplete, .DeriveFromFinal = %DeriveFromFinal.decl, .ConvertToBadBaseFinal = %ConvertToBadBaseFinal, .AccessMemberWithInvalidBaseFinal_WithMember = %AccessMemberWithInvalidBaseFinal_WithMember, .AccessMemberWithInvalidBaseFinal_NoMember = %AccessMemberWithInvalidBaseFinal_NoMember}
// CHECK:STDOUT:   %Base.decl = class_decl @Base, ()
// CHECK:STDOUT:   %Base: type = class_type @Base
// CHECK:STDOUT:   %Final.decl = class_decl @Final, ()
// CHECK:STDOUT:   %Final: type = class_type @Final
// CHECK:STDOUT:   %DeriveFromError.decl = class_decl @DeriveFromError, ()
// CHECK:STDOUT:   %DeriveFromError: type = class_type @DeriveFromError
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseError: <function> = fn_decl @AccessMemberWithInvalidBaseError
// CHECK:STDOUT:   %DeriveFromNonType.decl = class_decl @DeriveFromNonType, ()
// CHECK:STDOUT:   %DeriveFromNonType: type = class_type @DeriveFromNonType
// CHECK:STDOUT:   %AccessMemberWithInvalidBasNonType: <function> = fn_decl @AccessMemberWithInvalidBasNonType
// CHECK:STDOUT:   %DeriveFromi32.decl = class_decl @DeriveFromi32, ()
// CHECK:STDOUT:   %DeriveFromi32: type = class_type @DeriveFromi32
// CHECK:STDOUT:   %ConvertToBadBasei32: <function> = fn_decl @ConvertToBadBasei32
// CHECK:STDOUT:   %AccessMemberWithInvalidBasei32: <function> = fn_decl @AccessMemberWithInvalidBasei32
// CHECK:STDOUT:   %DeriveFromTuple.decl = class_decl @DeriveFromTuple, ()
// CHECK:STDOUT:   %DeriveFromTuple: type = class_type @DeriveFromTuple
// CHECK:STDOUT:   %ConvertToBadBaseTuple: <function> = fn_decl @ConvertToBadBaseTuple
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseTuple: <function> = fn_decl @AccessMemberWithInvalidBaseTuple
// CHECK:STDOUT:   %DeriveFromStruct.decl = class_decl @DeriveFromStruct, ()
// CHECK:STDOUT:   %DeriveFromStruct: type = class_type @DeriveFromStruct
// CHECK:STDOUT:   %ConvertToBadBaseStruct: <function> = fn_decl @ConvertToBadBaseStruct
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseStruct: <function> = fn_decl @AccessMemberWithInvalidBaseStruct
// CHECK:STDOUT:   %Incomplete.decl = class_decl @Incomplete, ()
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete
// CHECK:STDOUT:   %DeriveFromIncomplete.decl = class_decl @DeriveFromIncomplete, ()
// CHECK:STDOUT:   %DeriveFromIncomplete: type = class_type @DeriveFromIncomplete
// CHECK:STDOUT:   %ConvertToBadBaseIncomplete: <function> = fn_decl @ConvertToBadBaseIncomplete
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseIncomplete: <function> = fn_decl @AccessMemberWithInvalidBaseIncomplete
// CHECK:STDOUT:   %DeriveFromFinal.decl = class_decl @DeriveFromFinal, ()
// CHECK:STDOUT:   %DeriveFromFinal: type = class_type @DeriveFromFinal
// CHECK:STDOUT:   %ConvertToBadBaseFinal: <function> = fn_decl @ConvertToBadBaseFinal
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_WithMember: <function> = fn_decl @AccessMemberWithInvalidBaseFinal_WithMember
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_NoMember: <function> = fn_decl @AccessMemberWithInvalidBaseFinal_NoMember
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Final {
// CHECK:STDOUT:   %.loc9_8.1: type = unbound_element_type Final, i32
// CHECK:STDOUT:   %.loc9_8.2: <unbound element of class Final> = field_decl a, element0
// CHECK:STDOUT:   %a: <unbound element of class Final> = bind_name a, %.loc9_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromError {
// CHECK:STDOUT:   %error.ref: <error> = name_ref error, <error>
// CHECK:STDOUT:   %.loc16_21.1: type = unbound_element_type DeriveFromError, <error>
// CHECK:STDOUT:   %.loc16_21.2: <unbound element of class DeriveFromError> = base_decl <error>, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc16_21.2
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromNonType {
// CHECK:STDOUT:   %.loc26_16: i32 = int_literal 32
// CHECK:STDOUT:   %.loc26_18.1: type = unbound_element_type DeriveFromNonType, <error>
// CHECK:STDOUT:   %.loc26_18.2: <unbound element of class DeriveFromNonType> = base_decl <error>, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc26_18.2
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromi32 {
// CHECK:STDOUT:   %.loc35_19.1: type = unbound_element_type DeriveFromi32, <error>
// CHECK:STDOUT:   %.loc35_19.2: <unbound element of class DeriveFromi32> = base_decl <error>, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc35_19.2
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromTuple {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base
// CHECK:STDOUT:   %.loc51_22: (type,) = tuple_literal (%Base.ref)
// CHECK:STDOUT:   %.loc51_23.1: type = converted %.loc51_22, constants.%.loc51_23.1
// CHECK:STDOUT:   %.loc51_23.2: type = unbound_element_type DeriveFromTuple, <error>
// CHECK:STDOUT:   %.loc51_23.3: <unbound element of class DeriveFromTuple> = base_decl <error>, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc51_23.3
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromStruct {
// CHECK:STDOUT:   %.loc67_33: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc67_34.1: type = unbound_element_type DeriveFromStruct, <error>
// CHECK:STDOUT:   %.loc67_34.2: <unbound element of class DeriveFromStruct> = base_decl <error>, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc67_34.2
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromIncomplete {
// CHECK:STDOUT:   %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete
// CHECK:STDOUT:   %.loc87_26.1: type = unbound_element_type DeriveFromIncomplete, <error>
// CHECK:STDOUT:   %.loc87_26.2: <unbound element of class DeriveFromIncomplete> = base_decl <error>, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc87_26.2
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromFinal {
// CHECK:STDOUT:   %Final.ref: type = name_ref Final, file.%Final
// CHECK:STDOUT:   %.loc101_21.1: type = unbound_element_type DeriveFromFinal, Final
// CHECK:STDOUT:   %.loc101_21.2: <unbound element of class DeriveFromFinal> = base_decl Final, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc101_21.2
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseError(%p: DeriveFromError*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromError* = name_ref p, %p
// CHECK:STDOUT:   %.loc20: ref DeriveFromError = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error>
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBasNonType(%p: DeriveFromNonType*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromNonType* = name_ref p, %p
// CHECK:STDOUT:   %.loc29: ref DeriveFromNonType = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error>
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
// CHECK:STDOUT:   %.loc45: ref DeriveFromi32 = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error>
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
// CHECK:STDOUT:   %.loc59: ref DeriveFromTuple = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error>
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
// CHECK:STDOUT:   %.loc76: ref DeriveFromStruct = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error>
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
// CHECK:STDOUT:   %.loc95: ref DeriveFromIncomplete = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error>
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseFinal(%p: DeriveFromFinal*) -> Final* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromFinal* = name_ref p, %p
// CHECK:STDOUT:   %.loc106_11.1: ref DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %.loc106_11.2: ref Final = class_element_access %.loc106_11.1, element0
// CHECK:STDOUT:   %.loc106_11.3: Final* = address_of %.loc106_11.2
// CHECK:STDOUT:   %.loc106_11.4: Final* = converted %p.ref, %.loc106_11.3
// CHECK:STDOUT:   return %.loc106_11.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseFinal_WithMember(%p: DeriveFromFinal*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromFinal* = name_ref p, %p
// CHECK:STDOUT:   %.loc110_11: ref DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %.loc110_14.1: ref Final = class_element_access %.loc110_11, element0
// CHECK:STDOUT:   %.loc110_14.2: ref Final = converted %.loc110_11, %.loc110_14.1
// CHECK:STDOUT:   %.loc110_14.3: ref i32 = class_element_access %.loc110_14.2, element0
// CHECK:STDOUT:   %.loc110_14.4: i32 = bind_value %.loc110_14.3
// CHECK:STDOUT:   return %.loc110_14.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseFinal_NoMember(%p: DeriveFromFinal*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromFinal* = name_ref p, %p
// CHECK:STDOUT:   %.loc117: ref DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %b.ref: <error> = name_ref b, <error>
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
