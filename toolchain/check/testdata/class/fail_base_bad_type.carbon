// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_base_bad_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_base_bad_type.carbon

// --- fail_derive_from_error.carbon

class DeriveFromError {
  // CHECK:STDERR: fail_derive_from_error.carbon:[[@LINE+6]]:16: error: name `error` not found [NameNotFound]
  // CHECK:STDERR:   extend base: error;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_derive_from_non_type.carbon: error: `Main//default` previously provided by `fail_derive_from_error.carbon` [DuplicateMainApi]
  // CHECK:STDERR:
  extend base: error;
}

// This should not produce an error.
fn AccessMemberWithInvalidBaseError(p: DeriveFromError*) -> i32 { return (*p).n; }

// --- fail_derive_from_non_type.carbon

class DeriveFromNonType {
  // CHECK:STDERR: fail_derive_from_non_type.carbon:[[@LINE+9]]:16: error: cannot implicitly convert from `Core.IntLiteral` to `type` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   extend base: 32;
  // CHECK:STDERR:                ^~
  // CHECK:STDERR: fail_derive_from_non_type.carbon:[[@LINE+6]]:16: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   extend base: 32;
  // CHECK:STDERR:                ^~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_derive_from_i32.carbon: error: `Main//default` previously provided by `fail_derive_from_error.carbon` [DuplicateMainApi]
  // CHECK:STDERR:
  extend base: 32;
}

fn AccessMemberWithInvalidBasNonType(p: DeriveFromNonType*) -> i32 { return (*p).n; }

// --- fail_derive_from_i32.carbon

class DeriveFromi32 {
  // CHECK:STDERR: fail_derive_from_i32.carbon:[[@LINE+4]]:16: error: deriving from final type `i32`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:   extend base: i32;
  // CHECK:STDERR:                ^~~
  // CHECK:STDERR:
  extend base: i32;
}

// It's not really important whether this conversion produces an error or not,
// but it shouldn't crash.
// CHECK:STDERR: fail_derive_from_i32.carbon:[[@LINE+9]]:53: error: cannot implicitly convert from `DeriveFromi32*` to `i32*` [ImplicitAsConversionFailure]
// CHECK:STDERR: fn ConvertToBadBasei32(p: DeriveFromi32*) -> i32* { return p; }
// CHECK:STDERR:                                                     ^~~~~~~~~
// CHECK:STDERR: fail_derive_from_i32.carbon:[[@LINE+6]]:53: note: type `DeriveFromi32*` does not implement interface `ImplicitAs(i32*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertToBadBasei32(p: DeriveFromi32*) -> i32* { return p; }
// CHECK:STDERR:                                                     ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_derive_from_tuple.carbon: error: `Main//default` previously provided by `fail_derive_from_error.carbon` [DuplicateMainApi]
// CHECK:STDERR:
fn ConvertToBadBasei32(p: DeriveFromi32*) -> i32* { return p; }

fn AccessMemberWithInvalidBasei32(p: DeriveFromi32*) -> i32 { return (*p).n; }

// --- fail_derive_from_tuple.carbon

base class Base {}

class DeriveFromTuple {
  // CHECK:STDERR: fail_derive_from_tuple.carbon:[[@LINE+4]]:16: error: deriving from final type `(Base,)`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:   extend base: (Base,);
  // CHECK:STDERR:                ^~~~~~~
  // CHECK:STDERR:
  extend base: (Base,);
}

// CHECK:STDERR: fail_derive_from_tuple.carbon:[[@LINE+9]]:61: error: cannot implicitly convert from `DeriveFromTuple*` to `(Base,)*` [ImplicitAsConversionFailure]
// CHECK:STDERR: fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }
// CHECK:STDERR:                                                             ^~~~~~~~~
// CHECK:STDERR: fail_derive_from_tuple.carbon:[[@LINE+6]]:61: note: type `DeriveFromTuple*` does not implement interface `ImplicitAs((Base,)*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }
// CHECK:STDERR:                                                             ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_derive_from_struct.carbon: error: `Main//default` previously provided by `fail_derive_from_error.carbon` [DuplicateMainApi]
// CHECK:STDERR:
fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }

fn AccessMemberWithInvalidBaseTuple(p: DeriveFromTuple*) -> i32 { return (*p).n; }

// --- fail_derive_from_struct.carbon

// TODO: Should we allow this?
// We do allow `{.base = {.a: i32, .b: i32}}`.
class DeriveFromStruct {
  // CHECK:STDERR: fail_derive_from_struct.carbon:[[@LINE+4]]:16: error: deriving from final type `{.a: i32, .b: i32}`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:   extend base: {.a: i32, .b: i32};
  // CHECK:STDERR:                ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: {.a: i32, .b: i32};
}

// CHECK:STDERR: fail_derive_from_struct.carbon:[[@LINE+9]]:74: error: cannot implicitly convert from `DeriveFromStruct*` to `{.a: i32, .b: i32}*` [ImplicitAsConversionFailure]
// CHECK:STDERR: fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
// CHECK:STDERR: fail_derive_from_struct.carbon:[[@LINE+6]]:74: note: type `DeriveFromStruct*` does not implement interface `ImplicitAs({.a: i32, .b: i32}*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_derive_from_incomplete.carbon: error: `Main//default` previously provided by `fail_derive_from_error.carbon` [DuplicateMainApi]
// CHECK:STDERR:
fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }

// It would be OK to reject this if we start actually looking in the struct type.
fn AccessMemberWithInvalidBaseStruct(p: DeriveFromStruct*) -> i32 { return (*p).n; }

// --- fail_derive_from_incomplete.carbon

// CHECK:STDERR: fail_derive_from_incomplete.carbon:[[@LINE+4]]:1: error: `base` not allowed on `class` forward declaration, only definition [ModifierOnlyAllowedOnDefinition]
// CHECK:STDERR: base class Incomplete;
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
base class Incomplete;

class DeriveFromIncomplete {
  // CHECK:STDERR: fail_derive_from_incomplete.carbon:[[@LINE+7]]:16: error: base `Incomplete` is an incomplete type [IncompleteTypeInBaseDecl]
  // CHECK:STDERR:   extend base: Incomplete;
  // CHECK:STDERR:                ^~~~~~~~~~
  // CHECK:STDERR: fail_derive_from_incomplete.carbon:[[@LINE-6]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: base class Incomplete;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: Incomplete;
}

// CHECK:STDERR: fail_derive_from_incomplete.carbon:[[@LINE+9]]:74: error: cannot implicitly convert from `DeriveFromIncomplete*` to `Incomplete*` [ImplicitAsConversionFailure]
// CHECK:STDERR: fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
// CHECK:STDERR: fail_derive_from_incomplete.carbon:[[@LINE+6]]:74: note: type `DeriveFromIncomplete*` does not implement interface `ImplicitAs(Incomplete*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_derive_from_final.carbon: error: `Main//default` previously provided by `fail_derive_from_error.carbon` [DuplicateMainApi]
// CHECK:STDERR:
fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }

fn AccessMemberWithInvalidBaseIncomplete(p: DeriveFromIncomplete*) -> i32 { return (*p).n; }

// --- fail_derive_from_final.carbon

class Final {
  var a: i32;
}

class DeriveFromFinal {
  // CHECK:STDERR: fail_derive_from_final.carbon:[[@LINE+4]]:16: error: deriving from final type `Final`; base type must be an `abstract` or `base` class [BaseIsFinal]
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
  // CHECK:STDERR: fail_derive_from_final.carbon:[[@LINE+3]]:10: error: name `b` not found [NameNotFound]
  // CHECK:STDERR:   return (*p).b;
  // CHECK:STDERR:          ^~~~~~
  return (*p).b;
}

// CHECK:STDOUT: --- fail_derive_from_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DeriveFromError: type = class_type @DeriveFromError [template]
// CHECK:STDOUT:   %ptr: type = ptr_type %DeriveFromError [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseError.type: type = fn_type @AccessMemberWithInvalidBaseError [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseError: %AccessMemberWithInvalidBaseError.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .DeriveFromError = %DeriveFromError.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBaseError = %AccessMemberWithInvalidBaseError.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %DeriveFromError.decl: type = class_decl @DeriveFromError [template = constants.%DeriveFromError] {} {}
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseError.decl: %AccessMemberWithInvalidBaseError.type = fn_decl @AccessMemberWithInvalidBaseError [template = constants.%AccessMemberWithInvalidBaseError] {
// CHECK:STDOUT:     %p.patt: %ptr = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromError.ref: type = name_ref DeriveFromError, file.%DeriveFromError.decl [template = constants.%DeriveFromError]
// CHECK:STDOUT:     %ptr: type = ptr_type %DeriveFromError [template = constants.%ptr]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_61.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_61.2: type = converted %int.make_type_signed, %.loc13_61.1 [template = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromError {
// CHECK:STDOUT:   %error.ref: <error> = name_ref error, <error> [template = <error>]
// CHECK:STDOUT:   %.loc9: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromError
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   has_error
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseError(%p.param_patt: %ptr) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   %.loc13_75: ref %DeriveFromError = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_derive_from_non_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DeriveFromNonType: type = class_type @DeriveFromNonType [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %ptr: type = ptr_type %DeriveFromNonType [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBasNonType.type: type = fn_type @AccessMemberWithInvalidBasNonType [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBasNonType: %AccessMemberWithInvalidBasNonType.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.37
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .DeriveFromNonType = %DeriveFromNonType.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBasNonType = %AccessMemberWithInvalidBasNonType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %DeriveFromNonType.decl: type = class_decl @DeriveFromNonType [template = constants.%DeriveFromNonType] {} {}
// CHECK:STDOUT:   %AccessMemberWithInvalidBasNonType.decl: %AccessMemberWithInvalidBasNonType.type = fn_decl @AccessMemberWithInvalidBasNonType [template = constants.%AccessMemberWithInvalidBasNonType] {
// CHECK:STDOUT:     %p.patt: %ptr = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromNonType.ref: type = name_ref DeriveFromNonType, file.%DeriveFromNonType.decl [template = constants.%DeriveFromNonType]
// CHECK:STDOUT:     %ptr: type = ptr_type %DeriveFromNonType [template = constants.%ptr]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_64.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_64.2: type = converted %int.make_type_signed, %.loc15_64.1 [template = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromNonType {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %.loc12_16: type = converted %int_32, <error> [template = <error>]
// CHECK:STDOUT:   %.loc12_18: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromNonType
// CHECK:STDOUT:   .base = %.loc12_18
// CHECK:STDOUT:   has_error
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBasNonType(%p.param_patt: %ptr) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   %.loc15_78: ref %DeriveFromNonType = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_derive_from_i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DeriveFromi32: type = class_type @DeriveFromi32 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %DeriveFromi32 [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %ConvertToBadBasei32.type: type = fn_type @ConvertToBadBasei32 [template]
// CHECK:STDOUT:   %ConvertToBadBasei32: %ConvertToBadBasei32.type = struct_value () [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBasei32.type: type = fn_type @AccessMemberWithInvalidBasei32 [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBasei32: %AccessMemberWithInvalidBasei32.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .DeriveFromi32 = %DeriveFromi32.decl
// CHECK:STDOUT:     .ConvertToBadBasei32 = %ConvertToBadBasei32.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBasei32 = %AccessMemberWithInvalidBasei32.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %DeriveFromi32.decl: type = class_decl @DeriveFromi32 [template = constants.%DeriveFromi32] {} {}
// CHECK:STDOUT:   %ConvertToBadBasei32.decl: %ConvertToBadBasei32.type = fn_decl @ConvertToBadBasei32 [template = constants.%ConvertToBadBasei32] {
// CHECK:STDOUT:     %p.patt: %ptr.1 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.1 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %ptr.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromi32.ref: type = name_ref DeriveFromi32, file.%DeriveFromi32.decl [template = constants.%DeriveFromi32]
// CHECK:STDOUT:     %ptr.loc21_40: type = ptr_type %DeriveFromi32 [template = constants.%ptr.1]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_49.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_49.2: type = converted %int.make_type_signed, %.loc21_49.1 [template = constants.%i32]
// CHECK:STDOUT:     %ptr.loc21_49: type = ptr_type %i32 [template = constants.%ptr.2]
// CHECK:STDOUT:     %p.param: %ptr.1 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.1 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %ptr.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBasei32.decl: %AccessMemberWithInvalidBasei32.type = fn_decl @AccessMemberWithInvalidBasei32 [template = constants.%AccessMemberWithInvalidBasei32] {
// CHECK:STDOUT:     %p.patt: %ptr.1 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.1 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromi32.ref: type = name_ref DeriveFromi32, file.%DeriveFromi32.decl [template = constants.%DeriveFromi32]
// CHECK:STDOUT:     %ptr: type = ptr_type %DeriveFromi32 [template = constants.%ptr.1]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc23_57.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc23_57.2: type = converted %int.make_type_signed, %.loc23_57.1 [template = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.1 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.1 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromi32 {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_16.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_16.2: type = converted %int.make_type_signed, %.loc7_16.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_19: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromi32
// CHECK:STDOUT:   .base = %.loc7_19
// CHECK:STDOUT:   has_error
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBasei32(%p.param_patt: %ptr.1) -> %ptr.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.1 = name_ref p, %p
// CHECK:STDOUT:   %.loc21_61: %ptr.2 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBasei32(%p.param_patt: %ptr.1) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.1 = name_ref p, %p
// CHECK:STDOUT:   %.loc23_71: ref %DeriveFromi32 = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_derive_from_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %DeriveFromTuple: type = class_type @DeriveFromTuple [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%Base) [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %DeriveFromTuple [template]
// CHECK:STDOUT:   %ptr.3: type = ptr_type %tuple.type.2 [template]
// CHECK:STDOUT:   %ConvertToBadBaseTuple.type: type = fn_type @ConvertToBadBaseTuple [template]
// CHECK:STDOUT:   %ConvertToBadBaseTuple: %ConvertToBadBaseTuple.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseTuple.type: type = fn_type @AccessMemberWithInvalidBaseTuple [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseTuple: %AccessMemberWithInvalidBaseTuple.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.37
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .DeriveFromTuple = %DeriveFromTuple.decl
// CHECK:STDOUT:     .ConvertToBadBaseTuple = %ConvertToBadBaseTuple.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBaseTuple = %AccessMemberWithInvalidBaseTuple.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {} {}
// CHECK:STDOUT:   %DeriveFromTuple.decl: type = class_decl @DeriveFromTuple [template = constants.%DeriveFromTuple] {} {}
// CHECK:STDOUT:   %ConvertToBadBaseTuple.decl: %ConvertToBadBaseTuple.type = fn_decl @ConvertToBadBaseTuple [template = constants.%ConvertToBadBaseTuple] {
// CHECK:STDOUT:     %p.patt: %ptr.2 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.2 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %ptr.3 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.3 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromTuple.ref: type = name_ref DeriveFromTuple, file.%DeriveFromTuple.decl [template = constants.%DeriveFromTuple]
// CHECK:STDOUT:     %ptr.loc21_44: type = ptr_type %DeriveFromTuple [template = constants.%ptr.2]
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %.loc21_56: %tuple.type.1 = tuple_literal (%Base.ref)
// CHECK:STDOUT:     %.loc21_57: type = converted %.loc21_56, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %ptr.loc21_57: type = ptr_type %tuple.type.2 [template = constants.%ptr.3]
// CHECK:STDOUT:     %p.param: %ptr.2 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.2 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.3 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %ptr.3 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseTuple.decl: %AccessMemberWithInvalidBaseTuple.type = fn_decl @AccessMemberWithInvalidBaseTuple [template = constants.%AccessMemberWithInvalidBaseTuple] {
// CHECK:STDOUT:     %p.patt: %ptr.2 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.2 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromTuple.ref: type = name_ref DeriveFromTuple, file.%DeriveFromTuple.decl [template = constants.%DeriveFromTuple]
// CHECK:STDOUT:     %ptr: type = ptr_type %DeriveFromTuple [template = constants.%ptr.2]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc23_61.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc23_61.2: type = converted %int.make_type_signed, %.loc23_61.1 [template = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.2 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.2 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromTuple {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc9_22.1: %tuple.type.1 = tuple_literal (%Base.ref)
// CHECK:STDOUT:   %.loc9_22.2: type = converted %.loc9_22.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc9_23: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromTuple
// CHECK:STDOUT:   .base = %.loc9_23
// CHECK:STDOUT:   has_error
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseTuple(%p.param_patt: %ptr.2) -> %ptr.3 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.2 = name_ref p, %p
// CHECK:STDOUT:   %.loc21_69: %ptr.3 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseTuple(%p.param_patt: %ptr.2) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.2 = name_ref p, %p
// CHECK:STDOUT:   %.loc23_75: ref %DeriveFromTuple = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_derive_from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DeriveFromStruct: type = class_type @DeriveFromStruct [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %struct_type.a.b [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %DeriveFromStruct [template]
// CHECK:STDOUT:   %ConvertToBadBaseStruct.type: type = fn_type @ConvertToBadBaseStruct [template]
// CHECK:STDOUT:   %ConvertToBadBaseStruct: %ConvertToBadBaseStruct.type = struct_value () [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseStruct.type: type = fn_type @AccessMemberWithInvalidBaseStruct [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseStruct: %AccessMemberWithInvalidBaseStruct.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .DeriveFromStruct = %DeriveFromStruct.decl
// CHECK:STDOUT:     .ConvertToBadBaseStruct = %ConvertToBadBaseStruct.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBaseStruct = %AccessMemberWithInvalidBaseStruct.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %DeriveFromStruct.decl: type = class_decl @DeriveFromStruct [template = constants.%DeriveFromStruct] {} {}
// CHECK:STDOUT:   %ConvertToBadBaseStruct.decl: %ConvertToBadBaseStruct.type = fn_decl @ConvertToBadBaseStruct [template = constants.%ConvertToBadBaseStruct] {
// CHECK:STDOUT:     %p.patt: %ptr.2 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.2 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %ptr.1 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.1 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromStruct.ref: type = name_ref DeriveFromStruct, file.%DeriveFromStruct.decl [template = constants.%DeriveFromStruct]
// CHECK:STDOUT:     %ptr.loc21_46: type = ptr_type %DeriveFromStruct [template = constants.%ptr.2]
// CHECK:STDOUT:     %int_32.loc21_57: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc21_57: init type = call constants.%Int(%int_32.loc21_57) [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_57.1: type = value_of_initializer %int.make_type_signed.loc21_57 [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_57.2: type = converted %int.make_type_signed.loc21_57, %.loc21_57.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc21_66: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc21_66: init type = call constants.%Int(%int_32.loc21_66) [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_66.1: type = value_of_initializer %int.make_type_signed.loc21_66 [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_66.2: type = converted %int.make_type_signed.loc21_66, %.loc21_66.1 [template = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [template = constants.%struct_type.a.b]
// CHECK:STDOUT:     %ptr.loc21_70: type = ptr_type %struct_type.a.b [template = constants.%ptr.1]
// CHECK:STDOUT:     %p.param: %ptr.2 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.2 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.1 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %ptr.1 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseStruct.decl: %AccessMemberWithInvalidBaseStruct.type = fn_decl @AccessMemberWithInvalidBaseStruct [template = constants.%AccessMemberWithInvalidBaseStruct] {
// CHECK:STDOUT:     %p.patt: %ptr.2 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.2 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromStruct.ref: type = name_ref DeriveFromStruct, file.%DeriveFromStruct.decl [template = constants.%DeriveFromStruct]
// CHECK:STDOUT:     %ptr: type = ptr_type %DeriveFromStruct [template = constants.%ptr.2]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc24_63.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc24_63.2: type = converted %int.make_type_signed, %.loc24_63.1 [template = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.2 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.2 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromStruct {
// CHECK:STDOUT:   %int_32.loc9_21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc9_21: init type = call constants.%Int(%int_32.loc9_21) [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_21.1: type = value_of_initializer %int.make_type_signed.loc9_21 [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_21.2: type = converted %int.make_type_signed.loc9_21, %.loc9_21.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc9_30: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc9_30: init type = call constants.%Int(%int_32.loc9_30) [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_30.1: type = value_of_initializer %int.make_type_signed.loc9_30 [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_30.2: type = converted %int.make_type_signed.loc9_30, %.loc9_30.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [template = constants.%struct_type.a.b]
// CHECK:STDOUT:   %.loc9_34: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromStruct
// CHECK:STDOUT:   .base = %.loc9_34
// CHECK:STDOUT:   has_error
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseStruct(%p.param_patt: %ptr.2) -> %ptr.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.2 = name_ref p, %p
// CHECK:STDOUT:   %.loc21_82: %ptr.1 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseStruct(%p.param_patt: %ptr.2) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.2 = name_ref p, %p
// CHECK:STDOUT:   %.loc24_77: ref %DeriveFromStruct = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_derive_from_incomplete.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
// CHECK:STDOUT:   %DeriveFromIncomplete: type = class_type @DeriveFromIncomplete [template]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %DeriveFromIncomplete [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %Incomplete [template]
// CHECK:STDOUT:   %ConvertToBadBaseIncomplete.type: type = fn_type @ConvertToBadBaseIncomplete [template]
// CHECK:STDOUT:   %ConvertToBadBaseIncomplete: %ConvertToBadBaseIncomplete.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseIncomplete.type: type = fn_type @AccessMemberWithInvalidBaseIncomplete [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseIncomplete: %AccessMemberWithInvalidBaseIncomplete.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.37
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:     .DeriveFromIncomplete = %DeriveFromIncomplete.decl
// CHECK:STDOUT:     .ConvertToBadBaseIncomplete = %ConvertToBadBaseIncomplete.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBaseIncomplete = %AccessMemberWithInvalidBaseIncomplete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Incomplete.decl: type = class_decl @Incomplete [template = constants.%Incomplete] {} {}
// CHECK:STDOUT:   %DeriveFromIncomplete.decl: type = class_decl @DeriveFromIncomplete [template = constants.%DeriveFromIncomplete] {} {}
// CHECK:STDOUT:   %ConvertToBadBaseIncomplete.decl: %ConvertToBadBaseIncomplete.type = fn_decl @ConvertToBadBaseIncomplete [template = constants.%ConvertToBadBaseIncomplete] {
// CHECK:STDOUT:     %p.patt: %ptr.1 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.1 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %ptr.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromIncomplete.ref: type = name_ref DeriveFromIncomplete, file.%DeriveFromIncomplete.decl [template = constants.%DeriveFromIncomplete]
// CHECK:STDOUT:     %ptr.loc28_54: type = ptr_type %DeriveFromIncomplete [template = constants.%ptr.1]
// CHECK:STDOUT:     %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl [template = constants.%Incomplete]
// CHECK:STDOUT:     %ptr.loc28_70: type = ptr_type %Incomplete [template = constants.%ptr.2]
// CHECK:STDOUT:     %p.param: %ptr.1 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.1 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %ptr.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseIncomplete.decl: %AccessMemberWithInvalidBaseIncomplete.type = fn_decl @AccessMemberWithInvalidBaseIncomplete [template = constants.%AccessMemberWithInvalidBaseIncomplete] {
// CHECK:STDOUT:     %p.patt: %ptr.1 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.1 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromIncomplete.ref: type = name_ref DeriveFromIncomplete, file.%DeriveFromIncomplete.decl [template = constants.%DeriveFromIncomplete]
// CHECK:STDOUT:     %ptr: type = ptr_type %DeriveFromIncomplete [template = constants.%ptr.1]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc30_71.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc30_71.2: type = converted %int.make_type_signed, %.loc30_71.1 [template = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.1 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.1 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromIncomplete {
// CHECK:STDOUT:   %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl [template = constants.%Incomplete]
// CHECK:STDOUT:   %.loc16: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromIncomplete
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   has_error
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseIncomplete(%p.param_patt: %ptr.1) -> %ptr.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.1 = name_ref p, %p
// CHECK:STDOUT:   %.loc28: %ptr.2 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseIncomplete(%p.param_patt: %ptr.1) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.1 = name_ref p, %p
// CHECK:STDOUT:   %.loc30_85: ref %DeriveFromIncomplete = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_derive_from_final.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Final: type = class_type @Final [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Final.elem: type = unbound_element_type %Final, %i32 [template]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.a [template]
// CHECK:STDOUT:   %DeriveFromFinal: type = class_type @DeriveFromFinal [template]
// CHECK:STDOUT:   %DeriveFromFinal.elem: type = unbound_element_type %DeriveFromFinal, %Final [template]
// CHECK:STDOUT:   %struct_type.base.1: type = struct_type {.base: %Final} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.base.1 [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %DeriveFromFinal [template]
// CHECK:STDOUT:   %ptr.3: type = ptr_type %Final [template]
// CHECK:STDOUT:   %ConvertToBadBaseFinal.type: type = fn_type @ConvertToBadBaseFinal [template]
// CHECK:STDOUT:   %ConvertToBadBaseFinal: %ConvertToBadBaseFinal.type = struct_value () [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_WithMember.type: type = fn_type @AccessMemberWithInvalidBaseFinal_WithMember [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_WithMember: %AccessMemberWithInvalidBaseFinal_WithMember.type = struct_value () [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_NoMember.type: type = fn_type @AccessMemberWithInvalidBaseFinal_NoMember [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_NoMember: %AccessMemberWithInvalidBaseFinal_NoMember.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Final = %Final.decl
// CHECK:STDOUT:     .DeriveFromFinal = %DeriveFromFinal.decl
// CHECK:STDOUT:     .ConvertToBadBaseFinal = %ConvertToBadBaseFinal.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBaseFinal_WithMember = %AccessMemberWithInvalidBaseFinal_WithMember.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBaseFinal_NoMember = %AccessMemberWithInvalidBaseFinal_NoMember.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Final.decl: type = class_decl @Final [template = constants.%Final] {} {}
// CHECK:STDOUT:   %DeriveFromFinal.decl: type = class_decl @DeriveFromFinal [template = constants.%DeriveFromFinal] {} {}
// CHECK:STDOUT:   %ConvertToBadBaseFinal.decl: %ConvertToBadBaseFinal.type = fn_decl @ConvertToBadBaseFinal [template = constants.%ConvertToBadBaseFinal] {
// CHECK:STDOUT:     %p.patt: %ptr.2 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.2 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %ptr.3 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.3 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromFinal.ref: type = name_ref DeriveFromFinal, file.%DeriveFromFinal.decl [template = constants.%DeriveFromFinal]
// CHECK:STDOUT:     %ptr.loc15_44: type = ptr_type %DeriveFromFinal [template = constants.%ptr.2]
// CHECK:STDOUT:     %Final.ref: type = name_ref Final, file.%Final.decl [template = constants.%Final]
// CHECK:STDOUT:     %ptr.loc15_55: type = ptr_type %Final [template = constants.%ptr.3]
// CHECK:STDOUT:     %p.param: %ptr.2 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.2 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.3 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %ptr.3 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_WithMember.decl: %AccessMemberWithInvalidBaseFinal_WithMember.type = fn_decl @AccessMemberWithInvalidBaseFinal_WithMember [template = constants.%AccessMemberWithInvalidBaseFinal_WithMember] {
// CHECK:STDOUT:     %p.patt: %ptr.2 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.2 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromFinal.ref: type = name_ref DeriveFromFinal, file.%DeriveFromFinal.decl [template = constants.%DeriveFromFinal]
// CHECK:STDOUT:     %ptr: type = ptr_type %DeriveFromFinal [template = constants.%ptr.2]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_72.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_72.2: type = converted %int.make_type_signed, %.loc19_72.1 [template = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.2 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.2 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_NoMember.decl: %AccessMemberWithInvalidBaseFinal_NoMember.type = fn_decl @AccessMemberWithInvalidBaseFinal_NoMember [template = constants.%AccessMemberWithInvalidBaseFinal_NoMember] {
// CHECK:STDOUT:     %p.patt: %ptr.2 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.2 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromFinal.ref: type = name_ref DeriveFromFinal, file.%DeriveFromFinal.decl [template = constants.%DeriveFromFinal]
// CHECK:STDOUT:     %ptr: type = ptr_type %DeriveFromFinal [template = constants.%ptr.2]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc23_70.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc23_70.2: type = converted %int.make_type_signed, %.loc23_70.1 [template = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.2 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.2 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Final {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc3_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc3_10.2: type = converted %int.make_type_signed, %.loc3_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc3_8: %Final.elem = field_decl a, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Final
// CHECK:STDOUT:   .a = %.loc3_8
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromFinal {
// CHECK:STDOUT:   %Final.ref: type = name_ref Final, file.%Final.decl [template = constants.%Final]
// CHECK:STDOUT:   %.loc11: %DeriveFromFinal.elem = base_decl %Final.ref, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.1 [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromFinal
// CHECK:STDOUT:   .base = %.loc11
// CHECK:STDOUT:   extend %Final.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseFinal(%p.param_patt: %ptr.2) -> %ptr.3 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.2 = name_ref p, %p
// CHECK:STDOUT:   %.loc16_11.1: ref %DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %.loc16_11.2: ref %Final = class_element_access %.loc16_11.1, element0
// CHECK:STDOUT:   %addr: %ptr.3 = addr_of %.loc16_11.2
// CHECK:STDOUT:   %.loc16_11.3: %ptr.3 = converted %p.ref, %addr
// CHECK:STDOUT:   return %.loc16_11.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseFinal_WithMember(%p.param_patt: %ptr.2) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.2 = name_ref p, %p
// CHECK:STDOUT:   %.loc20_11: ref %DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %a.ref: %Final.elem = name_ref a, @Final.%.loc3_8 [template = @Final.%.loc3_8]
// CHECK:STDOUT:   %.loc20_14.1: ref %Final = class_element_access %.loc20_11, element0
// CHECK:STDOUT:   %.loc20_14.2: ref %Final = converted %.loc20_11, %.loc20_14.1
// CHECK:STDOUT:   %.loc20_14.3: ref %i32 = class_element_access %.loc20_14.2, element0
// CHECK:STDOUT:   %.loc20_14.4: %i32 = bind_value %.loc20_14.3
// CHECK:STDOUT:   return %.loc20_14.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseFinal_NoMember(%p.param_patt: %ptr.2) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.2 = name_ref p, %p
// CHECK:STDOUT:   %.loc27: ref %DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %b.ref: <error> = name_ref b, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
