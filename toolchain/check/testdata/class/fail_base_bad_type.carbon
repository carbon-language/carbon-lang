// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_base_bad_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_base_bad_type.carbon

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
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+5]]:16: ERROR: Expected name `Core.ImplicitAs` implicitly referenced here to name an interface.
  // CHECK:STDERR:   extend base: 32;
  // CHECK:STDERR:                ^~
  // CHECK:STDERR: fail_base_bad_type.carbon: Name declared here.
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
// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+5]]:53: ERROR: Expected name `Core.ImplicitAs` implicitly referenced here to name an interface.
// CHECK:STDERR: fn ConvertToBadBasei32(p: DeriveFromi32*) -> i32* { return p; }
// CHECK:STDERR:                                                     ^~~~~~~~~
// CHECK:STDERR: fail_base_bad_type.carbon: Name declared here.
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

// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+5]]:61: ERROR: Expected name `Core.ImplicitAs` implicitly referenced here to name an interface.
// CHECK:STDERR: fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }
// CHECK:STDERR:                                                             ^~~~~~~~~
// CHECK:STDERR: fail_base_bad_type.carbon: Name declared here.
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

// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+5]]:74: ERROR: Expected name `Core.ImplicitAs` implicitly referenced here to name an interface.
// CHECK:STDERR: fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
// CHECK:STDERR: fail_base_bad_type.carbon: Name declared here.
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

// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+5]]:74: ERROR: Expected name `Core.ImplicitAs` implicitly referenced here to name an interface.
// CHECK:STDERR: fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
// CHECK:STDERR: fail_base_bad_type.carbon: Name declared here.
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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type %Final, i32 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %DeriveFromError: type = class_type @DeriveFromError [template]
// CHECK:STDOUT:   %.5: type = ptr_type %DeriveFromError [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseError.type: type = fn_type @AccessMemberWithInvalidBaseError [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseError: %AccessMemberWithInvalidBaseError.type = struct_value () [template]
// CHECK:STDOUT:   %DeriveFromNonType: type = class_type @DeriveFromNonType [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 32 [template]
// CHECK:STDOUT:   %ImplicitAs.type: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic]
// CHECK:STDOUT:   %.7: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%.1 (%.7) = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %.7 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %.8: type = ptr_type %DeriveFromNonType [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBasNonType.type: type = fn_type @AccessMemberWithInvalidBasNonType [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBasNonType: %AccessMemberWithInvalidBasNonType.type = struct_value () [template]
// CHECK:STDOUT:   %DeriveFromi32: type = class_type @DeriveFromi32 [template]
// CHECK:STDOUT:   %.9: type = ptr_type %DeriveFromi32 [template]
// CHECK:STDOUT:   %.10: type = ptr_type i32 [template]
// CHECK:STDOUT:   %ConvertToBadBasei32.type: type = fn_type @ConvertToBadBasei32 [template]
// CHECK:STDOUT:   %ConvertToBadBasei32: %ConvertToBadBasei32.type = struct_value () [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBasei32.type: type = fn_type @AccessMemberWithInvalidBasei32 [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBasei32: %AccessMemberWithInvalidBasei32.type = struct_value () [template]
// CHECK:STDOUT:   %DeriveFromTuple: type = class_type @DeriveFromTuple [template]
// CHECK:STDOUT:   %.11: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.12: type = tuple_type (%Base) [template]
// CHECK:STDOUT:   %.13: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.14: type = tuple_type (%.13) [template]
// CHECK:STDOUT:   %.15: type = ptr_type %DeriveFromTuple [template]
// CHECK:STDOUT:   %.16: type = ptr_type %.12 [template]
// CHECK:STDOUT:   %ConvertToBadBaseTuple.type: type = fn_type @ConvertToBadBaseTuple [template]
// CHECK:STDOUT:   %ConvertToBadBaseTuple: %ConvertToBadBaseTuple.type = struct_value () [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseTuple.type: type = fn_type @AccessMemberWithInvalidBaseTuple [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseTuple: %AccessMemberWithInvalidBaseTuple.type = struct_value () [template]
// CHECK:STDOUT:   %DeriveFromStruct: type = class_type @DeriveFromStruct [template]
// CHECK:STDOUT:   %.17: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.18: type = ptr_type %.17 [template]
// CHECK:STDOUT:   %.19: type = ptr_type %DeriveFromStruct [template]
// CHECK:STDOUT:   %ConvertToBadBaseStruct.type: type = fn_type @ConvertToBadBaseStruct [template]
// CHECK:STDOUT:   %ConvertToBadBaseStruct: %ConvertToBadBaseStruct.type = struct_value () [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseStruct.type: type = fn_type @AccessMemberWithInvalidBaseStruct [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseStruct: %AccessMemberWithInvalidBaseStruct.type = struct_value () [template]
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
// CHECK:STDOUT:   %DeriveFromIncomplete: type = class_type @DeriveFromIncomplete [template]
// CHECK:STDOUT:   %.20: type = ptr_type %DeriveFromIncomplete [template]
// CHECK:STDOUT:   %.21: type = ptr_type %Incomplete [template]
// CHECK:STDOUT:   %ConvertToBadBaseIncomplete.type: type = fn_type @ConvertToBadBaseIncomplete [template]
// CHECK:STDOUT:   %ConvertToBadBaseIncomplete: %ConvertToBadBaseIncomplete.type = struct_value () [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseIncomplete.type: type = fn_type @AccessMemberWithInvalidBaseIncomplete [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseIncomplete: %AccessMemberWithInvalidBaseIncomplete.type = struct_value () [template]
// CHECK:STDOUT:   %DeriveFromFinal: type = class_type @DeriveFromFinal [template]
// CHECK:STDOUT:   %.22: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %.23: type = unbound_element_type %DeriveFromFinal, %Final [template]
// CHECK:STDOUT:   %.24: type = struct_type {.base: %Final} [template]
// CHECK:STDOUT:   %.25: type = ptr_type %DeriveFromFinal [template]
// CHECK:STDOUT:   %.26: type = ptr_type %Final [template]
// CHECK:STDOUT:   %ConvertToBadBaseFinal.type: type = fn_type @ConvertToBadBaseFinal [template]
// CHECK:STDOUT:   %ConvertToBadBaseFinal: %ConvertToBadBaseFinal.type = struct_value () [template]
// CHECK:STDOUT:   %.27: type = struct_type {.base: %.22} [template]
// CHECK:STDOUT:   %.28: type = ptr_type %.24 [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_WithMember.type: type = fn_type @AccessMemberWithInvalidBaseFinal_WithMember [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_WithMember: %AccessMemberWithInvalidBaseFinal_WithMember.type = struct_value () [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_NoMember.type: type = fn_type @AccessMemberWithInvalidBaseFinal_NoMember [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_NoMember: %AccessMemberWithInvalidBaseFinal_NoMember.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type = import_ref Core//prelude/operators/as, inst+37, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+42, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
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
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %Final.decl: type = class_decl @Final [template = constants.%Final] {}
// CHECK:STDOUT:   %DeriveFromError.decl: type = class_decl @DeriveFromError [template = constants.%DeriveFromError] {}
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseError.decl: %AccessMemberWithInvalidBaseError.type = fn_decl @AccessMemberWithInvalidBaseError [template = constants.%AccessMemberWithInvalidBaseError] {
// CHECK:STDOUT:     %DeriveFromError.ref: type = name_ref DeriveFromError, %DeriveFromError.decl [template = constants.%DeriveFromError]
// CHECK:STDOUT:     %.loc25_55: type = ptr_type %DeriveFromError [template = constants.%.5]
// CHECK:STDOUT:     %p.loc25_37.1: %.5 = param p
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseError.%p: %.5 = bind_name p, %p.loc25_37.1
// CHECK:STDOUT:     %int.make_type_32.loc25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc25_61.1: type = value_of_initializer %int.make_type_32.loc25 [template = i32]
// CHECK:STDOUT:     %.loc25_61.2: type = converted %int.make_type_32.loc25, %.loc25_61.1 [template = i32]
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseError.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeriveFromNonType.decl: type = class_decl @DeriveFromNonType [template = constants.%DeriveFromNonType] {}
// CHECK:STDOUT:   %AccessMemberWithInvalidBasNonType.decl: %AccessMemberWithInvalidBasNonType.type = fn_decl @AccessMemberWithInvalidBasNonType [template = constants.%AccessMemberWithInvalidBasNonType] {
// CHECK:STDOUT:     %DeriveFromNonType.ref: type = name_ref DeriveFromNonType, %DeriveFromNonType.decl [template = constants.%DeriveFromNonType]
// CHECK:STDOUT:     %.loc36_58: type = ptr_type %DeriveFromNonType [template = constants.%.8]
// CHECK:STDOUT:     %p.loc36_38.1: %.8 = param p
// CHECK:STDOUT:     @AccessMemberWithInvalidBasNonType.%p: %.8 = bind_name p, %p.loc36_38.1
// CHECK:STDOUT:     %int.make_type_32.loc36: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc36_64.1: type = value_of_initializer %int.make_type_32.loc36 [template = i32]
// CHECK:STDOUT:     %.loc36_64.2: type = converted %int.make_type_32.loc36, %.loc36_64.1 [template = i32]
// CHECK:STDOUT:     @AccessMemberWithInvalidBasNonType.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeriveFromi32.decl: type = class_decl @DeriveFromi32 [template = constants.%DeriveFromi32] {}
// CHECK:STDOUT:   %ConvertToBadBasei32.decl: %ConvertToBadBasei32.type = fn_decl @ConvertToBadBasei32 [template = constants.%ConvertToBadBasei32] {
// CHECK:STDOUT:     %DeriveFromi32.ref.loc53: type = name_ref DeriveFromi32, %DeriveFromi32.decl [template = constants.%DeriveFromi32]
// CHECK:STDOUT:     %.loc53_40: type = ptr_type %DeriveFromi32 [template = constants.%.9]
// CHECK:STDOUT:     %p.loc53_24.1: %.9 = param p
// CHECK:STDOUT:     @ConvertToBadBasei32.%p: %.9 = bind_name p, %p.loc53_24.1
// CHECK:STDOUT:     %int.make_type_32.loc53: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc53_49.1: type = value_of_initializer %int.make_type_32.loc53 [template = i32]
// CHECK:STDOUT:     %.loc53_49.2: type = converted %int.make_type_32.loc53, %.loc53_49.1 [template = i32]
// CHECK:STDOUT:     %.loc53_49.3: type = ptr_type i32 [template = constants.%.10]
// CHECK:STDOUT:     @ConvertToBadBasei32.%return: ref %.10 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBasei32.decl: %AccessMemberWithInvalidBasei32.type = fn_decl @AccessMemberWithInvalidBasei32 [template = constants.%AccessMemberWithInvalidBasei32] {
// CHECK:STDOUT:     %DeriveFromi32.ref.loc55: type = name_ref DeriveFromi32, %DeriveFromi32.decl [template = constants.%DeriveFromi32]
// CHECK:STDOUT:     %.loc55_51: type = ptr_type %DeriveFromi32 [template = constants.%.9]
// CHECK:STDOUT:     %p.loc55_35.1: %.9 = param p
// CHECK:STDOUT:     @AccessMemberWithInvalidBasei32.%p: %.9 = bind_name p, %p.loc55_35.1
// CHECK:STDOUT:     %int.make_type_32.loc55: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc55_57.1: type = value_of_initializer %int.make_type_32.loc55 [template = i32]
// CHECK:STDOUT:     %.loc55_57.2: type = converted %int.make_type_32.loc55, %.loc55_57.1 [template = i32]
// CHECK:STDOUT:     @AccessMemberWithInvalidBasei32.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeriveFromTuple.decl: type = class_decl @DeriveFromTuple [template = constants.%DeriveFromTuple] {}
// CHECK:STDOUT:   %ConvertToBadBaseTuple.decl: %ConvertToBadBaseTuple.type = fn_decl @ConvertToBadBaseTuple [template = constants.%ConvertToBadBaseTuple] {
// CHECK:STDOUT:     %DeriveFromTuple.ref.loc70: type = name_ref DeriveFromTuple, %DeriveFromTuple.decl [template = constants.%DeriveFromTuple]
// CHECK:STDOUT:     %.loc70_44: type = ptr_type %DeriveFromTuple [template = constants.%.15]
// CHECK:STDOUT:     %p.loc70_26.1: %.15 = param p
// CHECK:STDOUT:     @ConvertToBadBaseTuple.%p: %.15 = bind_name p, %p.loc70_26.1
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, %Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %.loc70_56: %.11 = tuple_literal (%Base.ref)
// CHECK:STDOUT:     %.loc70_57.1: type = converted %.loc70_56, constants.%.12 [template = constants.%.12]
// CHECK:STDOUT:     %.loc70_57.2: type = ptr_type %.12 [template = constants.%.16]
// CHECK:STDOUT:     @ConvertToBadBaseTuple.%return: ref %.16 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseTuple.decl: %AccessMemberWithInvalidBaseTuple.type = fn_decl @AccessMemberWithInvalidBaseTuple [template = constants.%AccessMemberWithInvalidBaseTuple] {
// CHECK:STDOUT:     %DeriveFromTuple.ref.loc72: type = name_ref DeriveFromTuple, %DeriveFromTuple.decl [template = constants.%DeriveFromTuple]
// CHECK:STDOUT:     %.loc72_55: type = ptr_type %DeriveFromTuple [template = constants.%.15]
// CHECK:STDOUT:     %p.loc72_37.1: %.15 = param p
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseTuple.%p: %.15 = bind_name p, %p.loc72_37.1
// CHECK:STDOUT:     %int.make_type_32.loc72: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc72_61.1: type = value_of_initializer %int.make_type_32.loc72 [template = i32]
// CHECK:STDOUT:     %.loc72_61.2: type = converted %int.make_type_32.loc72, %.loc72_61.1 [template = i32]
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseTuple.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeriveFromStruct.decl: type = class_decl @DeriveFromStruct [template = constants.%DeriveFromStruct] {}
// CHECK:STDOUT:   %ConvertToBadBaseStruct.decl: %ConvertToBadBaseStruct.type = fn_decl @ConvertToBadBaseStruct [template = constants.%ConvertToBadBaseStruct] {
// CHECK:STDOUT:     %DeriveFromStruct.ref.loc89: type = name_ref DeriveFromStruct, %DeriveFromStruct.decl [template = constants.%DeriveFromStruct]
// CHECK:STDOUT:     %.loc89_46: type = ptr_type %DeriveFromStruct [template = constants.%.19]
// CHECK:STDOUT:     %p.loc89_27.1: %.19 = param p
// CHECK:STDOUT:     @ConvertToBadBaseStruct.%p: %.19 = bind_name p, %p.loc89_27.1
// CHECK:STDOUT:     %int.make_type_32.loc89_57: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc89_57.1: type = value_of_initializer %int.make_type_32.loc89_57 [template = i32]
// CHECK:STDOUT:     %.loc89_57.2: type = converted %int.make_type_32.loc89_57, %.loc89_57.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc89_66: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc89_66.1: type = value_of_initializer %int.make_type_32.loc89_66 [template = i32]
// CHECK:STDOUT:     %.loc89_66.2: type = converted %int.make_type_32.loc89_66, %.loc89_66.1 [template = i32]
// CHECK:STDOUT:     %.loc89_69: type = struct_type {.a: i32, .b: i32} [template = constants.%.17]
// CHECK:STDOUT:     %.loc89_70: type = ptr_type %.17 [template = constants.%.18]
// CHECK:STDOUT:     @ConvertToBadBaseStruct.%return: ref %.18 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseStruct.decl: %AccessMemberWithInvalidBaseStruct.type = fn_decl @AccessMemberWithInvalidBaseStruct [template = constants.%AccessMemberWithInvalidBaseStruct] {
// CHECK:STDOUT:     %DeriveFromStruct.ref.loc92: type = name_ref DeriveFromStruct, %DeriveFromStruct.decl [template = constants.%DeriveFromStruct]
// CHECK:STDOUT:     %.loc92_57: type = ptr_type %DeriveFromStruct [template = constants.%.19]
// CHECK:STDOUT:     %p.loc92_38.1: %.19 = param p
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseStruct.%p: %.19 = bind_name p, %p.loc92_38.1
// CHECK:STDOUT:     %int.make_type_32.loc92: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc92_63.1: type = value_of_initializer %int.make_type_32.loc92 [template = i32]
// CHECK:STDOUT:     %.loc92_63.2: type = converted %int.make_type_32.loc92, %.loc92_63.1 [template = i32]
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseStruct.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Incomplete.decl: type = class_decl @Incomplete [template = constants.%Incomplete] {}
// CHECK:STDOUT:   %DeriveFromIncomplete.decl: type = class_decl @DeriveFromIncomplete [template = constants.%DeriveFromIncomplete] {}
// CHECK:STDOUT:   %ConvertToBadBaseIncomplete.decl: %ConvertToBadBaseIncomplete.type = fn_decl @ConvertToBadBaseIncomplete [template = constants.%ConvertToBadBaseIncomplete] {
// CHECK:STDOUT:     %DeriveFromIncomplete.ref.loc112: type = name_ref DeriveFromIncomplete, %DeriveFromIncomplete.decl [template = constants.%DeriveFromIncomplete]
// CHECK:STDOUT:     %.loc112_54: type = ptr_type %DeriveFromIncomplete [template = constants.%.20]
// CHECK:STDOUT:     %p.loc112_31.1: %.20 = param p
// CHECK:STDOUT:     @ConvertToBadBaseIncomplete.%p: %.20 = bind_name p, %p.loc112_31.1
// CHECK:STDOUT:     %Incomplete.ref: type = name_ref Incomplete, %Incomplete.decl [template = constants.%Incomplete]
// CHECK:STDOUT:     %.loc112_70: type = ptr_type %Incomplete [template = constants.%.21]
// CHECK:STDOUT:     @ConvertToBadBaseIncomplete.%return: ref %.21 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseIncomplete.decl: %AccessMemberWithInvalidBaseIncomplete.type = fn_decl @AccessMemberWithInvalidBaseIncomplete [template = constants.%AccessMemberWithInvalidBaseIncomplete] {
// CHECK:STDOUT:     %DeriveFromIncomplete.ref.loc114: type = name_ref DeriveFromIncomplete, %DeriveFromIncomplete.decl [template = constants.%DeriveFromIncomplete]
// CHECK:STDOUT:     %.loc114_65: type = ptr_type %DeriveFromIncomplete [template = constants.%.20]
// CHECK:STDOUT:     %p.loc114_42.1: %.20 = param p
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseIncomplete.%p: %.20 = bind_name p, %p.loc114_42.1
// CHECK:STDOUT:     %int.make_type_32.loc114: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc114_71.1: type = value_of_initializer %int.make_type_32.loc114 [template = i32]
// CHECK:STDOUT:     %.loc114_71.2: type = converted %int.make_type_32.loc114, %.loc114_71.1 [template = i32]
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseIncomplete.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeriveFromFinal.decl: type = class_decl @DeriveFromFinal [template = constants.%DeriveFromFinal] {}
// CHECK:STDOUT:   %ConvertToBadBaseFinal.decl: %ConvertToBadBaseFinal.type = fn_decl @ConvertToBadBaseFinal [template = constants.%ConvertToBadBaseFinal] {
// CHECK:STDOUT:     %DeriveFromFinal.ref.loc125: type = name_ref DeriveFromFinal, %DeriveFromFinal.decl [template = constants.%DeriveFromFinal]
// CHECK:STDOUT:     %.loc125_44: type = ptr_type %DeriveFromFinal [template = constants.%.25]
// CHECK:STDOUT:     %p.loc125_26.1: %.25 = param p
// CHECK:STDOUT:     @ConvertToBadBaseFinal.%p: %.25 = bind_name p, %p.loc125_26.1
// CHECK:STDOUT:     %Final.ref: type = name_ref Final, %Final.decl [template = constants.%Final]
// CHECK:STDOUT:     %.loc125_55: type = ptr_type %Final [template = constants.%.26]
// CHECK:STDOUT:     @ConvertToBadBaseFinal.%return: ref %.26 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_WithMember.decl: %AccessMemberWithInvalidBaseFinal_WithMember.type = fn_decl @AccessMemberWithInvalidBaseFinal_WithMember [template = constants.%AccessMemberWithInvalidBaseFinal_WithMember] {
// CHECK:STDOUT:     %DeriveFromFinal.ref.loc129: type = name_ref DeriveFromFinal, %DeriveFromFinal.decl [template = constants.%DeriveFromFinal]
// CHECK:STDOUT:     %.loc129_66: type = ptr_type %DeriveFromFinal [template = constants.%.25]
// CHECK:STDOUT:     %p.loc129_48.1: %.25 = param p
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseFinal_WithMember.%p: %.25 = bind_name p, %p.loc129_48.1
// CHECK:STDOUT:     %int.make_type_32.loc129: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc129_72.1: type = value_of_initializer %int.make_type_32.loc129 [template = i32]
// CHECK:STDOUT:     %.loc129_72.2: type = converted %int.make_type_32.loc129, %.loc129_72.1 [template = i32]
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseFinal_WithMember.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_NoMember.decl: %AccessMemberWithInvalidBaseFinal_NoMember.type = fn_decl @AccessMemberWithInvalidBaseFinal_NoMember [template = constants.%AccessMemberWithInvalidBaseFinal_NoMember] {
// CHECK:STDOUT:     %DeriveFromFinal.ref.loc133: type = name_ref DeriveFromFinal, %DeriveFromFinal.decl [template = constants.%DeriveFromFinal]
// CHECK:STDOUT:     %.loc133_64: type = ptr_type %DeriveFromFinal [template = constants.%.25]
// CHECK:STDOUT:     %p.loc133_46.1: %.25 = param p
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseFinal_NoMember.%p: %.25 = bind_name p, %p.loc133_46.1
// CHECK:STDOUT:     %int.make_type_32.loc133: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc133_70.1: type = value_of_initializer %int.make_type_32.loc133 [template = i32]
// CHECK:STDOUT:     %.loc133_70.2: type = converted %int.make_type_32.loc133, %.loc133_70.1 [template = i32]
// CHECK:STDOUT:     @AccessMemberWithInvalidBaseFinal_NoMember.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:   %Self: %.7 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Final {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc13_10.2: type = converted %int.make_type_32, %.loc13_10.1 [template = i32]
// CHECK:STDOUT:   %.loc13_8: %.3 = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Final
// CHECK:STDOUT:   .a = %.loc13_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromError {
// CHECK:STDOUT:   %error.ref: <error> = name_ref error, <error> [template = <error>]
// CHECK:STDOUT:   %.loc21: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromError
// CHECK:STDOUT:   .base = %.loc21
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromNonType {
// CHECK:STDOUT:   %.loc33_16.1: i32 = int_literal 32 [template = constants.%.6]
// CHECK:STDOUT:   %.loc33_16.2: type = converted %.loc33_16.1, <error> [template = <error>]
// CHECK:STDOUT:   %.loc33_18: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromNonType
// CHECK:STDOUT:   .base = %.loc33_18
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromi32 {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc43_16.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc43_16.2: type = converted %int.make_type_32, %.loc43_16.1 [template = i32]
// CHECK:STDOUT:   %.loc43_19: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromi32
// CHECK:STDOUT:   .base = %.loc43_19
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromTuple {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc62_22.1: %.11 = tuple_literal (%Base.ref)
// CHECK:STDOUT:   %.loc62_22.2: type = converted %.loc62_22.1, constants.%.12 [template = constants.%.12]
// CHECK:STDOUT:   %.loc62_23: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromTuple
// CHECK:STDOUT:   .base = %.loc62_23
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromStruct {
// CHECK:STDOUT:   %int.make_type_32.loc81_21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc81_21.1: type = value_of_initializer %int.make_type_32.loc81_21 [template = i32]
// CHECK:STDOUT:   %.loc81_21.2: type = converted %int.make_type_32.loc81_21, %.loc81_21.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc81_30: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc81_30.1: type = value_of_initializer %int.make_type_32.loc81_30 [template = i32]
// CHECK:STDOUT:   %.loc81_30.2: type = converted %int.make_type_32.loc81_30, %.loc81_30.1 [template = i32]
// CHECK:STDOUT:   %.loc81_33: type = struct_type {.a: i32, .b: i32} [template = constants.%.17]
// CHECK:STDOUT:   %.loc81_34: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromStruct
// CHECK:STDOUT:   .base = %.loc81_34
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromIncomplete {
// CHECK:STDOUT:   %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl [template = constants.%Incomplete]
// CHECK:STDOUT:   %.loc104: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromIncomplete
// CHECK:STDOUT:   .base = %.loc104
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromFinal {
// CHECK:STDOUT:   %Final.ref: type = name_ref Final, file.%Final.decl [template = constants.%Final]
// CHECK:STDOUT:   %.loc121: %.23 = base_decl %Final, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromFinal
// CHECK:STDOUT:   .base = %.loc121
// CHECK:STDOUT:   extend name_scope3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseError(%p: %.5) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.5 = name_ref p, %p
// CHECK:STDOUT:   %.loc25: ref %DeriveFromError = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBasNonType(%p: %.8) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.8 = name_ref p, %p
// CHECK:STDOUT:   %.loc36: ref %DeriveFromNonType = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBasei32(%p: %.9) -> %.10 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.9 = name_ref p, %p
// CHECK:STDOUT:   %.loc53: %.10 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBasei32(%p: %.9) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.9 = name_ref p, %p
// CHECK:STDOUT:   %.loc55: ref %DeriveFromi32 = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseTuple(%p: %.15) -> %.16 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.15 = name_ref p, %p
// CHECK:STDOUT:   %.loc70: %.16 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseTuple(%p: %.15) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.15 = name_ref p, %p
// CHECK:STDOUT:   %.loc72: ref %DeriveFromTuple = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseStruct(%p: %.19) -> %.18 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.19 = name_ref p, %p
// CHECK:STDOUT:   %.loc89: %.18 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseStruct(%p: %.19) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.19 = name_ref p, %p
// CHECK:STDOUT:   %.loc92: ref %DeriveFromStruct = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseIncomplete(%p: %.20) -> %.21 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.20 = name_ref p, %p
// CHECK:STDOUT:   %.loc112: %.21 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseIncomplete(%p: %.20) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.20 = name_ref p, %p
// CHECK:STDOUT:   %.loc114: ref %DeriveFromIncomplete = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseFinal(%p: %.25) -> %.26 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.25 = name_ref p, %p
// CHECK:STDOUT:   %.loc126_11.1: ref %DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %.loc126_11.2: ref %Final = class_element_access %.loc126_11.1, element0
// CHECK:STDOUT:   %.loc126_11.3: %.26 = addr_of %.loc126_11.2
// CHECK:STDOUT:   %.loc126_11.4: %.26 = converted %p.ref, %.loc126_11.3
// CHECK:STDOUT:   return %.loc126_11.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseFinal_WithMember(%p: %.25) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.25 = name_ref p, %p
// CHECK:STDOUT:   %.loc130_11: ref %DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %a.ref: %.3 = name_ref a, @Final.%.loc13_8 [template = @Final.%.loc13_8]
// CHECK:STDOUT:   %.loc130_14.1: ref %Final = class_element_access %.loc130_11, element0
// CHECK:STDOUT:   %.loc130_14.2: ref %Final = converted %.loc130_11, %.loc130_14.1
// CHECK:STDOUT:   %.loc130_14.3: ref i32 = class_element_access %.loc130_14.2, element0
// CHECK:STDOUT:   %.loc130_14.4: i32 = bind_value %.loc130_14.3
// CHECK:STDOUT:   return %.loc130_14.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseFinal_NoMember(%p: %.25) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.25 = name_ref p, %p
// CHECK:STDOUT:   %.loc137: ref %DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %b.ref: <error> = name_ref b, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
