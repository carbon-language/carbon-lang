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

library "[[@TEST_NAME]]";

class DeriveFromError {
  // CHECK:STDERR: fail_derive_from_error.carbon:[[@LINE+4]]:16: error: name `error` not found [NameNotFound]
  // CHECK:STDERR:   extend base: error;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  extend base: error;
}

// This should not produce an error.
fn AccessMemberWithInvalidBaseError(p: DeriveFromError*) -> i32 { return (*p).n; }

// --- fail_derive_from_non_type.carbon

library "[[@TEST_NAME]]";

class DeriveFromNonType {
  // CHECK:STDERR: fail_derive_from_non_type.carbon:[[@LINE+7]]:16: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   extend base: 32;
  // CHECK:STDERR:                ^~
  // CHECK:STDERR: fail_derive_from_non_type.carbon:[[@LINE+4]]:16: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   extend base: 32;
  // CHECK:STDERR:                ^~
  // CHECK:STDERR:
  extend base: 32;
}

fn AccessMemberWithInvalidBasNonType(p: DeriveFromNonType*) -> i32 { return (*p).n; }

// --- fail_derive_from_i32.carbon

library "[[@TEST_NAME]]";

class DeriveFromi32 {
  // CHECK:STDERR: fail_derive_from_i32.carbon:[[@LINE+4]]:16: error: deriving from final type `i32`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:   extend base: i32;
  // CHECK:STDERR:                ^~~
  // CHECK:STDERR:
  extend base: i32;
}

// It's not really important whether this conversion produces an error or not,
// but it shouldn't crash.
fn ConvertToBadBasei32(p: DeriveFromi32*) -> i32* { return p; }

// CHECK:STDERR: fail_derive_from_i32.carbon:[[@LINE+4]]:70: error: member name `n` not found in `DeriveFromi32` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: fn AccessMemberWithInvalidBasei32(p: DeriveFromi32*) -> i32 { return (*p).n; }
// CHECK:STDERR:                                                                      ^~~~~~
// CHECK:STDERR:
fn AccessMemberWithInvalidBasei32(p: DeriveFromi32*) -> i32 { return (*p).n; }

// --- fail_derive_from_tuple.carbon

library "[[@TEST_NAME]]";

base class Base {}

class DeriveFromTuple {
  // CHECK:STDERR: fail_derive_from_tuple.carbon:[[@LINE+4]]:16: error: deriving from final type `(Base,)`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:   extend base: (Base,);
  // CHECK:STDERR:                ^~~~~~~
  // CHECK:STDERR:
  extend base: (Base,);
}

// CHECK:STDERR: fail_derive_from_tuple.carbon:[[@LINE+7]]:61: error: cannot implicitly convert expression of type `DeriveFromTuple*` to `(Base,)*` [ConversionFailure]
// CHECK:STDERR: fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }
// CHECK:STDERR:                                                             ^~~~~~~~~
// CHECK:STDERR: fail_derive_from_tuple.carbon:[[@LINE+4]]:61: note: type `DeriveFromTuple*` does not implement interface `Core.ImplicitAs((Base,)*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }
// CHECK:STDERR:                                                             ^~~~~~~~~
// CHECK:STDERR:
fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }

fn AccessMemberWithInvalidBaseTuple(p: DeriveFromTuple*) -> i32 { return (*p).n; }

// --- fail_derive_from_struct.carbon

library "[[@TEST_NAME]]";

// TODO: Should we allow this?
// We do allow `{.base = {.a: i32, .b: i32}}`.
class DeriveFromStruct {
  // CHECK:STDERR: fail_derive_from_struct.carbon:[[@LINE+4]]:16: error: deriving from final type `{.a: i32, .b: i32}`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:   extend base: {.a: i32, .b: i32};
  // CHECK:STDERR:                ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: {.a: i32, .b: i32};
}

// CHECK:STDERR: fail_derive_from_struct.carbon:[[@LINE+7]]:74: error: cannot implicitly convert expression of type `DeriveFromStruct*` to `{.a: i32, .b: i32}*` [ConversionFailure]
// CHECK:STDERR: fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
// CHECK:STDERR: fail_derive_from_struct.carbon:[[@LINE+4]]:74: note: type `DeriveFromStruct*` does not implement interface `Core.ImplicitAs({.a: i32, .b: i32}*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
// CHECK:STDERR:
fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }

// It would be OK to reject this if we start actually looking in the struct type.
fn AccessMemberWithInvalidBaseStruct(p: DeriveFromStruct*) -> i32 { return (*p).n; }

// --- fail_derive_from_incomplete.carbon

library "[[@TEST_NAME]]";

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

// CHECK:STDERR: fail_derive_from_incomplete.carbon:[[@LINE+7]]:74: error: cannot implicitly convert expression of type `DeriveFromIncomplete*` to `Incomplete*` [ConversionFailure]
// CHECK:STDERR: fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
// CHECK:STDERR: fail_derive_from_incomplete.carbon:[[@LINE+4]]:74: note: type `DeriveFromIncomplete*` does not implement interface `Core.ImplicitAs(Incomplete*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
// CHECK:STDERR:
fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }

fn AccessMemberWithInvalidBaseIncomplete(p: DeriveFromIncomplete*) -> i32 { return (*p).n; }

// --- fail_derive_from_final.carbon

library "[[@TEST_NAME]]";

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
  // CHECK:STDERR: fail_derive_from_final.carbon:[[@LINE+4]]:10: error: member name `b` not found in `DeriveFromFinal` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   return (*p).b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return (*p).b;
}

// CHECK:STDOUT: --- fail_derive_from_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DeriveFromError: type = class_type @DeriveFromError [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %DeriveFromError [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseError.type: type = fn_type @AccessMemberWithInvalidBaseError [concrete]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseError: %AccessMemberWithInvalidBaseError.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .DeriveFromError = %DeriveFromError.decl
// CHECK:STDOUT:     .error = <poisoned>
// CHECK:STDOUT:     .AccessMemberWithInvalidBaseError = %AccessMemberWithInvalidBaseError.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %DeriveFromError.decl: type = class_decl @DeriveFromError [concrete = constants.%DeriveFromError] {} {}
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseError.decl: %AccessMemberWithInvalidBaseError.type = fn_decl @AccessMemberWithInvalidBaseError [concrete = constants.%AccessMemberWithInvalidBaseError] {
// CHECK:STDOUT:     %p.patt: %ptr = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc13_55: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %DeriveFromError.ref: type = name_ref DeriveFromError, file.%DeriveFromError.decl [concrete = constants.%DeriveFromError]
// CHECK:STDOUT:       %ptr: type = ptr_type %DeriveFromError [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromError {
// CHECK:STDOUT:   %error.ref: <error> = name_ref error, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc9: <error> = base_decl <error>, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness <error> [concrete = <error>]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromError
// CHECK:STDOUT:   .error = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .n = <poisoned>
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseError(%p.param_patt: %ptr) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   %.loc13_75: ref %DeriveFromError = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_derive_from_non_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DeriveFromNonType: type = class_type @DeriveFromNonType [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %DeriveFromNonType [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %AccessMemberWithInvalidBasNonType.type: type = fn_type @AccessMemberWithInvalidBasNonType [concrete]
// CHECK:STDOUT:   %AccessMemberWithInvalidBasNonType: %AccessMemberWithInvalidBasNonType.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .DeriveFromNonType = %DeriveFromNonType.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBasNonType = %AccessMemberWithInvalidBasNonType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %DeriveFromNonType.decl: type = class_decl @DeriveFromNonType [concrete = constants.%DeriveFromNonType] {} {}
// CHECK:STDOUT:   %AccessMemberWithInvalidBasNonType.decl: %AccessMemberWithInvalidBasNonType.type = fn_decl @AccessMemberWithInvalidBasNonType [concrete = constants.%AccessMemberWithInvalidBasNonType] {
// CHECK:STDOUT:     %p.patt: %ptr = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc15_58: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %DeriveFromNonType.ref: type = name_ref DeriveFromNonType, file.%DeriveFromNonType.decl [concrete = constants.%DeriveFromNonType]
// CHECK:STDOUT:       %ptr: type = ptr_type %DeriveFromNonType [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromNonType {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %.loc12_16: type = converted %int_32, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc12_18: <error> = base_decl <error>, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness <error> [concrete = <error>]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromNonType
// CHECK:STDOUT:   .base = %.loc12_18
// CHECK:STDOUT:   .n = <poisoned>
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBasNonType(%p.param_patt: %ptr) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   %.loc15_78: ref %DeriveFromNonType = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_derive_from_i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DeriveFromi32: type = class_type @DeriveFromi32 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %DeriveFromi32.elem: type = unbound_element_type %DeriveFromi32, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.386: <witness> = complete_type_witness %struct_type.base [concrete]
// CHECK:STDOUT:   %ptr.45c: type = ptr_type %DeriveFromi32 [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %ConvertToBadBasei32.type: type = fn_type @ConvertToBadBasei32 [concrete]
// CHECK:STDOUT:   %ConvertToBadBasei32: %ConvertToBadBasei32.type = struct_value () [concrete]
// CHECK:STDOUT:   %AccessMemberWithInvalidBasei32.type: type = fn_type @AccessMemberWithInvalidBasei32 [concrete]
// CHECK:STDOUT:   %AccessMemberWithInvalidBasei32: %AccessMemberWithInvalidBasei32.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .DeriveFromi32 = %DeriveFromi32.decl
// CHECK:STDOUT:     .ConvertToBadBasei32 = %ConvertToBadBasei32.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBasei32 = %AccessMemberWithInvalidBasei32.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %DeriveFromi32.decl: type = class_decl @DeriveFromi32 [concrete = constants.%DeriveFromi32] {} {}
// CHECK:STDOUT:   %ConvertToBadBasei32.decl: %ConvertToBadBasei32.type = fn_decl @ConvertToBadBasei32 [concrete = constants.%ConvertToBadBasei32] {
// CHECK:STDOUT:     %p.patt: %ptr.45c = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.45c = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %ptr.235 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.235 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc14_49: type = ptr_type %i32 [concrete = constants.%ptr.235]
// CHECK:STDOUT:     %p.param: %ptr.45c = value_param call_param0
// CHECK:STDOUT:     %.loc14_40: type = splice_block %ptr.loc14_40 [concrete = constants.%ptr.45c] {
// CHECK:STDOUT:       %DeriveFromi32.ref: type = name_ref DeriveFromi32, file.%DeriveFromi32.decl [concrete = constants.%DeriveFromi32]
// CHECK:STDOUT:       %ptr.loc14_40: type = ptr_type %DeriveFromi32 [concrete = constants.%ptr.45c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.45c = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.235 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.235 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBasei32.decl: %AccessMemberWithInvalidBasei32.type = fn_decl @AccessMemberWithInvalidBasei32 [concrete = constants.%AccessMemberWithInvalidBasei32] {
// CHECK:STDOUT:     %p.patt: %ptr.45c = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.45c = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.45c = value_param call_param0
// CHECK:STDOUT:     %.loc20_51: type = splice_block %ptr [concrete = constants.%ptr.45c] {
// CHECK:STDOUT:       %DeriveFromi32.ref: type = name_ref DeriveFromi32, file.%DeriveFromi32.decl [concrete = constants.%DeriveFromi32]
// CHECK:STDOUT:       %ptr: type = ptr_type %DeriveFromi32 [concrete = constants.%ptr.45c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.45c = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromi32 {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc9: %DeriveFromi32.elem = base_decl %i32, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.386]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromi32
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .n = <poisoned>
// CHECK:STDOUT:   extend %i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBasei32(%p.param_patt: %ptr.45c) -> %ptr.235 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.45c = name_ref p, %p
// CHECK:STDOUT:   %.loc14_61.1: ref %DeriveFromi32 = deref %p.ref
// CHECK:STDOUT:   %.loc14_61.2: ref %i32 = class_element_access %.loc14_61.1, element0
// CHECK:STDOUT:   %addr: %ptr.235 = addr_of %.loc14_61.2
// CHECK:STDOUT:   %.loc14_61.3: %ptr.235 = converted %p.ref, %addr
// CHECK:STDOUT:   return %.loc14_61.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBasei32(%p.param_patt: %ptr.45c) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.45c = name_ref p, %p
// CHECK:STDOUT:   %.loc20_71: ref %DeriveFromi32 = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_derive_from_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %DeriveFromTuple: type = class_type @DeriveFromTuple [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.type.469: type = tuple_type (%Base) [concrete]
// CHECK:STDOUT:   %ptr.340: type = ptr_type %DeriveFromTuple [concrete]
// CHECK:STDOUT:   %ptr.1ab: type = ptr_type %tuple.type.469 [concrete]
// CHECK:STDOUT:   %ConvertToBadBaseTuple.type: type = fn_type @ConvertToBadBaseTuple [concrete]
// CHECK:STDOUT:   %ConvertToBadBaseTuple: %ConvertToBadBaseTuple.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseTuple.type: type = fn_type @AccessMemberWithInvalidBaseTuple [concrete]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseTuple: %AccessMemberWithInvalidBaseTuple.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
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
// CHECK:STDOUT:     .DeriveFromTuple = %DeriveFromTuple.decl
// CHECK:STDOUT:     .ConvertToBadBaseTuple = %ConvertToBadBaseTuple.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBaseTuple = %AccessMemberWithInvalidBaseTuple.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %DeriveFromTuple.decl: type = class_decl @DeriveFromTuple [concrete = constants.%DeriveFromTuple] {} {}
// CHECK:STDOUT:   %ConvertToBadBaseTuple.decl: %ConvertToBadBaseTuple.type = fn_decl @ConvertToBadBaseTuple [concrete = constants.%ConvertToBadBaseTuple] {
// CHECK:STDOUT:     %p.patt: %ptr.340 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.340 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %ptr.1ab = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.1ab = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:     %.loc21_56: %tuple.type.85c = tuple_literal (%Base.ref)
// CHECK:STDOUT:     %.loc21_57: type = converted %.loc21_56, constants.%tuple.type.469 [concrete = constants.%tuple.type.469]
// CHECK:STDOUT:     %ptr.loc21_57: type = ptr_type %tuple.type.469 [concrete = constants.%ptr.1ab]
// CHECK:STDOUT:     %p.param: %ptr.340 = value_param call_param0
// CHECK:STDOUT:     %.loc21_44: type = splice_block %ptr.loc21_44 [concrete = constants.%ptr.340] {
// CHECK:STDOUT:       %DeriveFromTuple.ref: type = name_ref DeriveFromTuple, file.%DeriveFromTuple.decl [concrete = constants.%DeriveFromTuple]
// CHECK:STDOUT:       %ptr.loc21_44: type = ptr_type %DeriveFromTuple [concrete = constants.%ptr.340]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.340 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.1ab = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.1ab = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseTuple.decl: %AccessMemberWithInvalidBaseTuple.type = fn_decl @AccessMemberWithInvalidBaseTuple [concrete = constants.%AccessMemberWithInvalidBaseTuple] {
// CHECK:STDOUT:     %p.patt: %ptr.340 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.340 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.340 = value_param call_param0
// CHECK:STDOUT:     %.loc23_55: type = splice_block %ptr [concrete = constants.%ptr.340] {
// CHECK:STDOUT:       %DeriveFromTuple.ref: type = name_ref DeriveFromTuple, file.%DeriveFromTuple.decl [concrete = constants.%DeriveFromTuple]
// CHECK:STDOUT:       %ptr: type = ptr_type %DeriveFromTuple [concrete = constants.%ptr.340]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.340 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: class @DeriveFromTuple {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc11_22.1: %tuple.type.85c = tuple_literal (%Base.ref)
// CHECK:STDOUT:   %.loc11_22.2: type = converted %.loc11_22.1, constants.%tuple.type.469 [concrete = constants.%tuple.type.469]
// CHECK:STDOUT:   %.loc11_23: <error> = base_decl <error>, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness <error> [concrete = <error>]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromTuple
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc11_23
// CHECK:STDOUT:   .n = <poisoned>
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseTuple(%p.param_patt: %ptr.340) -> %ptr.1ab {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.340 = name_ref p, %p
// CHECK:STDOUT:   %.loc21_69: %ptr.1ab = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseTuple(%p.param_patt: %ptr.340) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.340 = name_ref p, %p
// CHECK:STDOUT:   %.loc23_75: ref %DeriveFromTuple = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_derive_from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DeriveFromStruct: type = class_type @DeriveFromStruct [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %ptr.3ee: type = ptr_type %struct_type.a.b [concrete]
// CHECK:STDOUT:   %ptr.df0: type = ptr_type %DeriveFromStruct [concrete]
// CHECK:STDOUT:   %ConvertToBadBaseStruct.type: type = fn_type @ConvertToBadBaseStruct [concrete]
// CHECK:STDOUT:   %ConvertToBadBaseStruct: %ConvertToBadBaseStruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseStruct.type: type = fn_type @AccessMemberWithInvalidBaseStruct [concrete]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseStruct: %AccessMemberWithInvalidBaseStruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .DeriveFromStruct = %DeriveFromStruct.decl
// CHECK:STDOUT:     .ConvertToBadBaseStruct = %ConvertToBadBaseStruct.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBaseStruct = %AccessMemberWithInvalidBaseStruct.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %DeriveFromStruct.decl: type = class_decl @DeriveFromStruct [concrete = constants.%DeriveFromStruct] {} {}
// CHECK:STDOUT:   %ConvertToBadBaseStruct.decl: %ConvertToBadBaseStruct.type = fn_decl @ConvertToBadBaseStruct [concrete = constants.%ConvertToBadBaseStruct] {
// CHECK:STDOUT:     %p.patt: %ptr.df0 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.df0 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %ptr.3ee = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.3ee = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc21_57: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc21_57: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc21_66: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc21_66: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b]
// CHECK:STDOUT:     %ptr.loc21_70: type = ptr_type %struct_type.a.b [concrete = constants.%ptr.3ee]
// CHECK:STDOUT:     %p.param: %ptr.df0 = value_param call_param0
// CHECK:STDOUT:     %.loc21_46: type = splice_block %ptr.loc21_46 [concrete = constants.%ptr.df0] {
// CHECK:STDOUT:       %DeriveFromStruct.ref: type = name_ref DeriveFromStruct, file.%DeriveFromStruct.decl [concrete = constants.%DeriveFromStruct]
// CHECK:STDOUT:       %ptr.loc21_46: type = ptr_type %DeriveFromStruct [concrete = constants.%ptr.df0]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.df0 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.3ee = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.3ee = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseStruct.decl: %AccessMemberWithInvalidBaseStruct.type = fn_decl @AccessMemberWithInvalidBaseStruct [concrete = constants.%AccessMemberWithInvalidBaseStruct] {
// CHECK:STDOUT:     %p.patt: %ptr.df0 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.df0 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.df0 = value_param call_param0
// CHECK:STDOUT:     %.loc24_57: type = splice_block %ptr [concrete = constants.%ptr.df0] {
// CHECK:STDOUT:       %DeriveFromStruct.ref: type = name_ref DeriveFromStruct, file.%DeriveFromStruct.decl [concrete = constants.%DeriveFromStruct]
// CHECK:STDOUT:       %ptr: type = ptr_type %DeriveFromStruct [concrete = constants.%ptr.df0]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.df0 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromStruct {
// CHECK:STDOUT:   %int_32.loc11_21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc11_21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc11_30: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc11_30: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b]
// CHECK:STDOUT:   %.loc11: <error> = base_decl <error>, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness <error> [concrete = <error>]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromStruct
// CHECK:STDOUT:   .base = %.loc11
// CHECK:STDOUT:   .n = <poisoned>
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseStruct(%p.param_patt: %ptr.df0) -> %ptr.3ee {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.df0 = name_ref p, %p
// CHECK:STDOUT:   %.loc21_82: %ptr.3ee = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseStruct(%p.param_patt: %ptr.df0) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.df0 = name_ref p, %p
// CHECK:STDOUT:   %.loc24_77: ref %DeriveFromStruct = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_derive_from_incomplete.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [concrete]
// CHECK:STDOUT:   %DeriveFromIncomplete: type = class_type @DeriveFromIncomplete [concrete]
// CHECK:STDOUT:   %ptr.089: type = ptr_type %DeriveFromIncomplete [concrete]
// CHECK:STDOUT:   %ptr.c62: type = ptr_type %Incomplete [concrete]
// CHECK:STDOUT:   %ConvertToBadBaseIncomplete.type: type = fn_type @ConvertToBadBaseIncomplete [concrete]
// CHECK:STDOUT:   %ConvertToBadBaseIncomplete: %ConvertToBadBaseIncomplete.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseIncomplete.type: type = fn_type @AccessMemberWithInvalidBaseIncomplete [concrete]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseIncomplete: %AccessMemberWithInvalidBaseIncomplete.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:     .DeriveFromIncomplete = %DeriveFromIncomplete.decl
// CHECK:STDOUT:     .ConvertToBadBaseIncomplete = %ConvertToBadBaseIncomplete.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBaseIncomplete = %AccessMemberWithInvalidBaseIncomplete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Incomplete.decl: type = class_decl @Incomplete [concrete = constants.%Incomplete] {} {}
// CHECK:STDOUT:   %DeriveFromIncomplete.decl: type = class_decl @DeriveFromIncomplete [concrete = constants.%DeriveFromIncomplete] {} {}
// CHECK:STDOUT:   %ConvertToBadBaseIncomplete.decl: %ConvertToBadBaseIncomplete.type = fn_decl @ConvertToBadBaseIncomplete [concrete = constants.%ConvertToBadBaseIncomplete] {
// CHECK:STDOUT:     %p.patt: %ptr.089 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.089 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %ptr.c62 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.c62 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl [concrete = constants.%Incomplete]
// CHECK:STDOUT:     %ptr.loc28_70: type = ptr_type %Incomplete [concrete = constants.%ptr.c62]
// CHECK:STDOUT:     %p.param: %ptr.089 = value_param call_param0
// CHECK:STDOUT:     %.loc28_54: type = splice_block %ptr.loc28_54 [concrete = constants.%ptr.089] {
// CHECK:STDOUT:       %DeriveFromIncomplete.ref: type = name_ref DeriveFromIncomplete, file.%DeriveFromIncomplete.decl [concrete = constants.%DeriveFromIncomplete]
// CHECK:STDOUT:       %ptr.loc28_54: type = ptr_type %DeriveFromIncomplete [concrete = constants.%ptr.089]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.089 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.c62 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.c62 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseIncomplete.decl: %AccessMemberWithInvalidBaseIncomplete.type = fn_decl @AccessMemberWithInvalidBaseIncomplete [concrete = constants.%AccessMemberWithInvalidBaseIncomplete] {
// CHECK:STDOUT:     %p.patt: %ptr.089 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.089 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.089 = value_param call_param0
// CHECK:STDOUT:     %.loc30_65: type = splice_block %ptr [concrete = constants.%ptr.089] {
// CHECK:STDOUT:       %DeriveFromIncomplete.ref: type = name_ref DeriveFromIncomplete, file.%DeriveFromIncomplete.decl [concrete = constants.%DeriveFromIncomplete]
// CHECK:STDOUT:       %ptr: type = ptr_type %DeriveFromIncomplete [concrete = constants.%ptr.089]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.089 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromIncomplete {
// CHECK:STDOUT:   %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl [concrete = constants.%Incomplete]
// CHECK:STDOUT:   %.loc18: <error> = base_decl <error>, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness <error> [concrete = <error>]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromIncomplete
// CHECK:STDOUT:   .Incomplete = <poisoned>
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   .n = <poisoned>
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseIncomplete(%p.param_patt: %ptr.089) -> %ptr.c62 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.089 = name_ref p, %p
// CHECK:STDOUT:   %.loc28_82: %ptr.c62 = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseIncomplete(%p.param_patt: %ptr.089) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.089 = name_ref p, %p
// CHECK:STDOUT:   %.loc30_85: ref %DeriveFromIncomplete = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_derive_from_final.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Final: type = class_type @Final [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Final.elem: type = unbound_element_type %Final, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.fd7: <witness> = complete_type_witness %struct_type.a [concrete]
// CHECK:STDOUT:   %DeriveFromFinal: type = class_type @DeriveFromFinal [concrete]
// CHECK:STDOUT:   %DeriveFromFinal.elem: type = unbound_element_type %DeriveFromFinal, %Final [concrete]
// CHECK:STDOUT:   %struct_type.base.dae: type = struct_type {.base: %Final} [concrete]
// CHECK:STDOUT:   %complete_type.970: <witness> = complete_type_witness %struct_type.base.dae [concrete]
// CHECK:STDOUT:   %ptr.160: type = ptr_type %DeriveFromFinal [concrete]
// CHECK:STDOUT:   %ptr.5f6: type = ptr_type %Final [concrete]
// CHECK:STDOUT:   %ConvertToBadBaseFinal.type: type = fn_type @ConvertToBadBaseFinal [concrete]
// CHECK:STDOUT:   %ConvertToBadBaseFinal: %ConvertToBadBaseFinal.type = struct_value () [concrete]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_WithMember.type: type = fn_type @AccessMemberWithInvalidBaseFinal_WithMember [concrete]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_WithMember: %AccessMemberWithInvalidBaseFinal_WithMember.type = struct_value () [concrete]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_NoMember.type: type = fn_type @AccessMemberWithInvalidBaseFinal_NoMember [concrete]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_NoMember: %AccessMemberWithInvalidBaseFinal_NoMember.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .Final = %Final.decl
// CHECK:STDOUT:     .DeriveFromFinal = %DeriveFromFinal.decl
// CHECK:STDOUT:     .ConvertToBadBaseFinal = %ConvertToBadBaseFinal.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBaseFinal_WithMember = %AccessMemberWithInvalidBaseFinal_WithMember.decl
// CHECK:STDOUT:     .AccessMemberWithInvalidBaseFinal_NoMember = %AccessMemberWithInvalidBaseFinal_NoMember.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Final.decl: type = class_decl @Final [concrete = constants.%Final] {} {}
// CHECK:STDOUT:   %DeriveFromFinal.decl: type = class_decl @DeriveFromFinal [concrete = constants.%DeriveFromFinal] {} {}
// CHECK:STDOUT:   %ConvertToBadBaseFinal.decl: %ConvertToBadBaseFinal.type = fn_decl @ConvertToBadBaseFinal [concrete = constants.%ConvertToBadBaseFinal] {
// CHECK:STDOUT:     %p.patt: %ptr.160 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.160 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %ptr.5f6 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.5f6 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Final.ref: type = name_ref Final, file.%Final.decl [concrete = constants.%Final]
// CHECK:STDOUT:     %ptr.loc17_55: type = ptr_type %Final [concrete = constants.%ptr.5f6]
// CHECK:STDOUT:     %p.param: %ptr.160 = value_param call_param0
// CHECK:STDOUT:     %.loc17: type = splice_block %ptr.loc17_44 [concrete = constants.%ptr.160] {
// CHECK:STDOUT:       %DeriveFromFinal.ref: type = name_ref DeriveFromFinal, file.%DeriveFromFinal.decl [concrete = constants.%DeriveFromFinal]
// CHECK:STDOUT:       %ptr.loc17_44: type = ptr_type %DeriveFromFinal [concrete = constants.%ptr.160]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.160 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.5f6 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.5f6 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_WithMember.decl: %AccessMemberWithInvalidBaseFinal_WithMember.type = fn_decl @AccessMemberWithInvalidBaseFinal_WithMember [concrete = constants.%AccessMemberWithInvalidBaseFinal_WithMember] {
// CHECK:STDOUT:     %p.patt: %ptr.160 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.160 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.160 = value_param call_param0
// CHECK:STDOUT:     %.loc21: type = splice_block %ptr [concrete = constants.%ptr.160] {
// CHECK:STDOUT:       %DeriveFromFinal.ref: type = name_ref DeriveFromFinal, file.%DeriveFromFinal.decl [concrete = constants.%DeriveFromFinal]
// CHECK:STDOUT:       %ptr: type = ptr_type %DeriveFromFinal [concrete = constants.%ptr.160]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.160 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_NoMember.decl: %AccessMemberWithInvalidBaseFinal_NoMember.type = fn_decl @AccessMemberWithInvalidBaseFinal_NoMember [concrete = constants.%AccessMemberWithInvalidBaseFinal_NoMember] {
// CHECK:STDOUT:     %p.patt: %ptr.160 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.160 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.160 = value_param call_param0
// CHECK:STDOUT:     %.loc25: type = splice_block %ptr [concrete = constants.%ptr.160] {
// CHECK:STDOUT:       %DeriveFromFinal.ref: type = name_ref DeriveFromFinal, file.%DeriveFromFinal.decl [concrete = constants.%DeriveFromFinal]
// CHECK:STDOUT:       %ptr: type = ptr_type %DeriveFromFinal [concrete = constants.%ptr.160]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.160 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Final {
// CHECK:STDOUT:   %.loc5_8: %Final.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc5_3: %Final.elem = var_pattern %.loc5_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %Final.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [concrete = constants.%complete_type.fd7]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Final
// CHECK:STDOUT:   .a = %.loc5_8
// CHECK:STDOUT:   .b = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromFinal {
// CHECK:STDOUT:   %Final.ref: type = name_ref Final, file.%Final.decl [concrete = constants.%Final]
// CHECK:STDOUT:   %.loc13: %DeriveFromFinal.elem = base_decl %Final.ref, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.dae [concrete = constants.%complete_type.970]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromFinal
// CHECK:STDOUT:   .Final = <poisoned>
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   .a = <poisoned>
// CHECK:STDOUT:   .b = <poisoned>
// CHECK:STDOUT:   extend %Final.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseFinal(%p.param_patt: %ptr.160) -> %ptr.5f6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.160 = name_ref p, %p
// CHECK:STDOUT:   %.loc18_11.1: ref %DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %.loc18_11.2: ref %Final = class_element_access %.loc18_11.1, element0
// CHECK:STDOUT:   %addr: %ptr.5f6 = addr_of %.loc18_11.2
// CHECK:STDOUT:   %.loc18_11.3: %ptr.5f6 = converted %p.ref, %addr
// CHECK:STDOUT:   return %.loc18_11.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseFinal_WithMember(%p.param_patt: %ptr.160) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.160 = name_ref p, %p
// CHECK:STDOUT:   %.loc22_11: ref %DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %a.ref: %Final.elem = name_ref a, @Final.%.loc5_8 [concrete = @Final.%.loc5_8]
// CHECK:STDOUT:   %.loc22_14.1: ref %Final = class_element_access %.loc22_11, element0
// CHECK:STDOUT:   %.loc22_14.2: ref %Final = converted %.loc22_11, %.loc22_14.1
// CHECK:STDOUT:   %.loc22_14.3: ref %i32 = class_element_access %.loc22_14.2, element0
// CHECK:STDOUT:   %.loc22_14.4: %i32 = bind_value %.loc22_14.3
// CHECK:STDOUT:   return %.loc22_14.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseFinal_NoMember(%p.param_patt: %ptr.160) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.160 = name_ref p, %p
// CHECK:STDOUT:   %.loc30: ref %DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %b.ref: <error> = name_ref b, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
