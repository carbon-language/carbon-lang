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
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:16: ERROR: Cannot implicitly convert from `i32` to `type`.
  // CHECK:STDERR:   extend base: 32;
  // CHECK:STDERR:                ^~
  extend base: 32;
}

fn AccessMemberWithInvalidBasNonType(p: DeriveFromNonType*) -> i32 { return (*p).n; }

class DeriveFromi32 {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:16: ERROR: Deriving from final type `i32`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   extend base: i32;
  // CHECK:STDERR:                ^~~
  extend base: i32;
}

// It's not really important whether this conversion produces an error or not,
// but it shouldn't crash.
// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:53: ERROR: Cannot implicitly convert from `DeriveFromi32*` to `i32*`.
// CHECK:STDERR: fn ConvertToBadBasei32(p: DeriveFromi32*) -> i32* { return p; }
// CHECK:STDERR:                                                     ^~~~~~~~~
fn ConvertToBadBasei32(p: DeriveFromi32*) -> i32* { return p; }

fn AccessMemberWithInvalidBasei32(p: DeriveFromi32*) -> i32 { return (*p).n; }

class DeriveFromTuple {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:16: ERROR: Deriving from final type `(Base,)`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   extend base: (Base,);
  // CHECK:STDERR:                ^~~~~~~
  extend base: (Base,);
}

// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:61: ERROR: Cannot implicitly convert from `DeriveFromTuple*` to `(Base,)*`.
// CHECK:STDERR: fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }
// CHECK:STDERR:                                                             ^~~~~~~~~
fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }

fn AccessMemberWithInvalidBaseTuple(p: DeriveFromTuple*) -> i32 { return (*p).n; }

// TODO: Should we allow this?
// We do allow `{.base = {.a: i32, .b: i32}}`.
class DeriveFromStruct {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:16: ERROR: Deriving from final type `{.a: i32, .b: i32}`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   extend base: {.a: i32, .b: i32};
  // CHECK:STDERR:                ^~~~~~~~~~~~~~~~~~
  extend base: {.a: i32, .b: i32};
}

// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:74: ERROR: Cannot implicitly convert from `DeriveFromStruct*` to `{.a: i32, .b: i32}*`.
// CHECK:STDERR: fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }

// It would be OK to reject this if we start actually looking in the struct type.
fn AccessMemberWithInvalidBaseStruct(p: DeriveFromStruct*) -> i32 { return (*p).n; }

base class Incomplete;

class DeriveFromIncomplete {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+6]]:16: ERROR: Base `Incomplete` is an incomplete type.
  // CHECK:STDERR:   extend base: Incomplete;
  // CHECK:STDERR:                ^~~~~~~~~~
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE-6]]:1: Class was forward declared here.
  // CHECK:STDERR: base class Incomplete;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
  extend base: Incomplete;
}

// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:74: ERROR: Cannot implicitly convert from `DeriveFromIncomplete*` to `Incomplete*`.
// CHECK:STDERR: fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }

fn AccessMemberWithInvalidBaseIncomplete(p: DeriveFromIncomplete*) -> i32 { return (*p).n; }

class DeriveFromFinal {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:16: ERROR: Deriving from final type `Final`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   extend base: Final;
  // CHECK:STDERR:                ^~~~~
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
// CHECK:STDOUT:   %DeriveFromNonType: type = class_type @DeriveFromNonType [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 32 [template]
// CHECK:STDOUT:   %.6: type = ptr_type DeriveFromNonType [template]
// CHECK:STDOUT:   %DeriveFromi32: type = class_type @DeriveFromi32 [template]
// CHECK:STDOUT:   %.7: type = ptr_type DeriveFromi32 [template]
// CHECK:STDOUT:   %.8: type = ptr_type i32 [template]
// CHECK:STDOUT:   %DeriveFromTuple: type = class_type @DeriveFromTuple [template]
// CHECK:STDOUT:   %.9: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.10: type = tuple_type (Base) [template]
// CHECK:STDOUT:   %.11: type = tuple_type () [template]
// CHECK:STDOUT:   %.12: type = ptr_type {} [template]
// CHECK:STDOUT:   %.13: type = tuple_type ({}*) [template]
// CHECK:STDOUT:   %.14: type = ptr_type DeriveFromTuple [template]
// CHECK:STDOUT:   %.15: type = ptr_type (Base,) [template]
// CHECK:STDOUT:   %DeriveFromStruct: type = class_type @DeriveFromStruct [template]
// CHECK:STDOUT:   %.16: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.17: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.18: type = ptr_type DeriveFromStruct [template]
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
// CHECK:STDOUT:   %DeriveFromIncomplete: type = class_type @DeriveFromIncomplete [template]
// CHECK:STDOUT:   %.19: type = ptr_type DeriveFromIncomplete [template]
// CHECK:STDOUT:   %.20: type = ptr_type Incomplete [template]
// CHECK:STDOUT:   %DeriveFromFinal: type = class_type @DeriveFromFinal [template]
// CHECK:STDOUT:   %.21: type = ptr_type {.a: i32} [template]
// CHECK:STDOUT:   %.22: type = unbound_element_type DeriveFromFinal, Final [template]
// CHECK:STDOUT:   %.23: type = struct_type {.base: Final} [template]
// CHECK:STDOUT:   %.24: type = ptr_type DeriveFromFinal [template]
// CHECK:STDOUT:   %.25: type = ptr_type Final [template]
// CHECK:STDOUT:   %.26: type = struct_type {.base: {.a: i32}*} [template]
// CHECK:STDOUT:   %.27: type = ptr_type {.base: Final} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Base = %Base.decl, .Final = %Final.decl, .DeriveFromError = %DeriveFromError.decl, .AccessMemberWithInvalidBaseError = %AccessMemberWithInvalidBaseError, .DeriveFromNonType = %DeriveFromNonType.decl, .AccessMemberWithInvalidBasNonType = %AccessMemberWithInvalidBasNonType, .DeriveFromi32 = %DeriveFromi32.decl, .ConvertToBadBasei32 = %ConvertToBadBasei32, .AccessMemberWithInvalidBasei32 = %AccessMemberWithInvalidBasei32, .DeriveFromTuple = %DeriveFromTuple.decl, .ConvertToBadBaseTuple = %ConvertToBadBaseTuple, .AccessMemberWithInvalidBaseTuple = %AccessMemberWithInvalidBaseTuple, .DeriveFromStruct = %DeriveFromStruct.decl, .ConvertToBadBaseStruct = %ConvertToBadBaseStruct, .AccessMemberWithInvalidBaseStruct = %AccessMemberWithInvalidBaseStruct, .Incomplete = %Incomplete.decl, .DeriveFromIncomplete = %DeriveFromIncomplete.decl, .ConvertToBadBaseIncomplete = %ConvertToBadBaseIncomplete, .AccessMemberWithInvalidBaseIncomplete = %AccessMemberWithInvalidBaseIncomplete, .DeriveFromFinal = %DeriveFromFinal.decl, .ConvertToBadBaseFinal = %ConvertToBadBaseFinal, .AccessMemberWithInvalidBaseFinal_WithMember = %AccessMemberWithInvalidBaseFinal_WithMember, .AccessMemberWithInvalidBaseFinal_NoMember = %AccessMemberWithInvalidBaseFinal_NoMember} [template]
// CHECK:STDOUT:   %Base.decl = class_decl @Base, ()
// CHECK:STDOUT:   %Final.decl = class_decl @Final, ()
// CHECK:STDOUT:   %DeriveFromError.decl = class_decl @DeriveFromError, ()
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseError: <function> = fn_decl @AccessMemberWithInvalidBaseError [template]
// CHECK:STDOUT:   %DeriveFromNonType.decl = class_decl @DeriveFromNonType, ()
// CHECK:STDOUT:   %AccessMemberWithInvalidBasNonType: <function> = fn_decl @AccessMemberWithInvalidBasNonType [template]
// CHECK:STDOUT:   %DeriveFromi32.decl = class_decl @DeriveFromi32, ()
// CHECK:STDOUT:   %ConvertToBadBasei32: <function> = fn_decl @ConvertToBadBasei32 [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBasei32: <function> = fn_decl @AccessMemberWithInvalidBasei32 [template]
// CHECK:STDOUT:   %DeriveFromTuple.decl = class_decl @DeriveFromTuple, ()
// CHECK:STDOUT:   %ConvertToBadBaseTuple: <function> = fn_decl @ConvertToBadBaseTuple [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseTuple: <function> = fn_decl @AccessMemberWithInvalidBaseTuple [template]
// CHECK:STDOUT:   %DeriveFromStruct.decl = class_decl @DeriveFromStruct, ()
// CHECK:STDOUT:   %ConvertToBadBaseStruct: <function> = fn_decl @ConvertToBadBaseStruct [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseStruct: <function> = fn_decl @AccessMemberWithInvalidBaseStruct [template]
// CHECK:STDOUT:   %Incomplete.decl = class_decl @Incomplete, ()
// CHECK:STDOUT:   %DeriveFromIncomplete.decl = class_decl @DeriveFromIncomplete, ()
// CHECK:STDOUT:   %ConvertToBadBaseIncomplete: <function> = fn_decl @ConvertToBadBaseIncomplete [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseIncomplete: <function> = fn_decl @AccessMemberWithInvalidBaseIncomplete [template]
// CHECK:STDOUT:   %DeriveFromFinal.decl = class_decl @DeriveFromFinal, ()
// CHECK:STDOUT:   %ConvertToBadBaseFinal: <function> = fn_decl @ConvertToBadBaseFinal [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_WithMember: <function> = fn_decl @AccessMemberWithInvalidBaseFinal_WithMember [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_NoMember: <function> = fn_decl @AccessMemberWithInvalidBaseFinal_NoMember [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Final {
// CHECK:STDOUT:   %.loc9: <unbound element of class Final> = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromError {
// CHECK:STDOUT:   %error.ref: <error> = name_ref error, <error> [template = <error>]
// CHECK:STDOUT:   %.loc16: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromNonType {
// CHECK:STDOUT:   %.loc26_16: i32 = int_literal 32 [template = constants.%.5]
// CHECK:STDOUT:   %.loc26_18: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc26_18
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromi32 {
// CHECK:STDOUT:   %.loc35: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc35
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromTuple {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:   %.loc51_22.1: (type,) = tuple_literal (%Base.ref)
// CHECK:STDOUT:   %.loc51_22.2: type = converted %.loc51_22.1, constants.%.10 [template = constants.%.10]
// CHECK:STDOUT:   %.loc51_23: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc51_23
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromStruct {
// CHECK:STDOUT:   %.loc67_33: type = struct_type {.a: i32, .b: i32} [template = constants.%.16]
// CHECK:STDOUT:   %.loc67_34: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc67_34
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromIncomplete {
// CHECK:STDOUT:   %Incomplete.ref: type = name_ref Incomplete, constants.%Incomplete [template = constants.%Incomplete]
// CHECK:STDOUT:   %.loc87: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc87
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromFinal {
// CHECK:STDOUT:   %Final.ref: type = name_ref Final, constants.%Final [template = constants.%Final]
// CHECK:STDOUT:   %.loc101: <unbound element of class DeriveFromFinal> = base_decl Final, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc101
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseError(%p: DeriveFromError*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromError* = name_ref p, %p
// CHECK:STDOUT:   %.loc20: ref DeriveFromError = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBasNonType(%p: DeriveFromNonType*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromNonType* = name_ref p, %p
// CHECK:STDOUT:   %.loc29: ref DeriveFromNonType = deref %p.ref
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
// CHECK:STDOUT:   %.loc45: ref DeriveFromi32 = deref %p.ref
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
// CHECK:STDOUT:   %.loc59: ref DeriveFromTuple = deref %p.ref
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
// CHECK:STDOUT:   %.loc76: ref DeriveFromStruct = deref %p.ref
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
// CHECK:STDOUT:   %.loc95: ref DeriveFromIncomplete = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseFinal(%p: DeriveFromFinal*) -> Final* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromFinal* = name_ref p, %p
// CHECK:STDOUT:   %.loc106_11.1: ref DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %.loc106_11.2: ref Final = class_element_access %.loc106_11.1, element0
// CHECK:STDOUT:   %.loc106_11.3: Final* = addr_of %.loc106_11.2
// CHECK:STDOUT:   %.loc106_10: Final* = converted %p.ref, %.loc106_11.3
// CHECK:STDOUT:   return %.loc106_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseFinal_WithMember(%p: DeriveFromFinal*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromFinal* = name_ref p, %p
// CHECK:STDOUT:   %.loc110_11.1: ref DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %.loc110_14.1: ref Final = class_element_access %.loc110_11.1, element0
// CHECK:STDOUT:   %.loc110_11.2: ref Final = converted %.loc110_11.1, %.loc110_14.1
// CHECK:STDOUT:   %.loc110_14.2: ref i32 = class_element_access %.loc110_11.2, element0
// CHECK:STDOUT:   %.loc110_14.3: i32 = bind_value %.loc110_14.2
// CHECK:STDOUT:   return %.loc110_14.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseFinal_NoMember(%p: DeriveFromFinal*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: DeriveFromFinal* = name_ref p, %p
// CHECK:STDOUT:   %.loc117: ref DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %b.ref: <error> = name_ref b, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
