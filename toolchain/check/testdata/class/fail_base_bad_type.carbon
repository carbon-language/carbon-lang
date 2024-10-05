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
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:16: error: name `error` not found
  // CHECK:STDERR:   extend base: error;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  extend base: error;
}

// This should not produce an error.
fn AccessMemberWithInvalidBaseError(p: DeriveFromError*) -> i32 { return (*p).n; }

class DeriveFromNonType {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+7]]:16: error: cannot implicitly convert from `i32` to `type`
  // CHECK:STDERR:   extend base: 32;
  // CHECK:STDERR:                ^~
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:16: note: type `i32` does not implement interface `ImplicitAs`
  // CHECK:STDERR:   extend base: 32;
  // CHECK:STDERR:                ^~
  // CHECK:STDERR:
  extend base: 32;
}

fn AccessMemberWithInvalidBasNonType(p: DeriveFromNonType*) -> i32 { return (*p).n; }

class DeriveFromi32 {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:16: error: deriving from final type `i32`; base type must be an `abstract` or `base` class
  // CHECK:STDERR:   extend base: i32;
  // CHECK:STDERR:                ^~~
  // CHECK:STDERR:
  extend base: i32;
}

// It's not really important whether this conversion produces an error or not,
// but it shouldn't crash.
// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+7]]:53: error: cannot implicitly convert from `DeriveFromi32*` to `i32*`
// CHECK:STDERR: fn ConvertToBadBasei32(p: DeriveFromi32*) -> i32* { return p; }
// CHECK:STDERR:                                                     ^~~~~~~~~
// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:53: note: type `DeriveFromi32*` does not implement interface `ImplicitAs`
// CHECK:STDERR: fn ConvertToBadBasei32(p: DeriveFromi32*) -> i32* { return p; }
// CHECK:STDERR:                                                     ^~~~~~~~~
// CHECK:STDERR:
fn ConvertToBadBasei32(p: DeriveFromi32*) -> i32* { return p; }

fn AccessMemberWithInvalidBasei32(p: DeriveFromi32*) -> i32 { return (*p).n; }

class DeriveFromTuple {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:16: error: deriving from final type `(Base,)`; base type must be an `abstract` or `base` class
  // CHECK:STDERR:   extend base: (Base,);
  // CHECK:STDERR:                ^~~~~~~
  // CHECK:STDERR:
  extend base: (Base,);
}

// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+7]]:61: error: cannot implicitly convert from `DeriveFromTuple*` to `(Base,)*`
// CHECK:STDERR: fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }
// CHECK:STDERR:                                                             ^~~~~~~~~
// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:61: note: type `DeriveFromTuple*` does not implement interface `ImplicitAs`
// CHECK:STDERR: fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }
// CHECK:STDERR:                                                             ^~~~~~~~~
// CHECK:STDERR:
fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }

fn AccessMemberWithInvalidBaseTuple(p: DeriveFromTuple*) -> i32 { return (*p).n; }

// TODO: Should we allow this?
// We do allow `{.base = {.a: i32, .b: i32}}`.
class DeriveFromStruct {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:16: error: deriving from final type `{.a: i32, .b: i32}`; base type must be an `abstract` or `base` class
  // CHECK:STDERR:   extend base: {.a: i32, .b: i32};
  // CHECK:STDERR:                ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: {.a: i32, .b: i32};
}

// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+7]]:74: error: cannot implicitly convert from `DeriveFromStruct*` to `{.a: i32, .b: i32}*`
// CHECK:STDERR: fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:74: note: type `DeriveFromStruct*` does not implement interface `ImplicitAs`
// CHECK:STDERR: fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
// CHECK:STDERR:
fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }

// It would be OK to reject this if we start actually looking in the struct type.
fn AccessMemberWithInvalidBaseStruct(p: DeriveFromStruct*) -> i32 { return (*p).n; }

base class Incomplete;

class DeriveFromIncomplete {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+7]]:16: error: base `Incomplete` is an incomplete type
  // CHECK:STDERR:   extend base: Incomplete;
  // CHECK:STDERR:                ^~~~~~~~~~
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE-6]]:1: note: class was forward declared here
  // CHECK:STDERR: base class Incomplete;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: Incomplete;
}

// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+7]]:74: error: cannot implicitly convert from `DeriveFromIncomplete*` to `Incomplete*`
// CHECK:STDERR: fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
// CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:74: note: type `DeriveFromIncomplete*` does not implement interface `ImplicitAs`
// CHECK:STDERR: fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
// CHECK:STDERR:
fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }

fn AccessMemberWithInvalidBaseIncomplete(p: DeriveFromIncomplete*) -> i32 { return (*p).n; }

class DeriveFromFinal {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+4]]:16: error: deriving from final type `Final`; base type must be an `abstract` or `base` class
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
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:10: error: name `b` not found
  // CHECK:STDERR:   return (*p).b;
  // CHECK:STDERR:          ^~~~~~
  return (*p).b;
}

// CHECK:STDOUT: --- fail_base_bad_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Final: type = class_type @Final [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type %Final, i32 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %.5 [template]
// CHECK:STDOUT:   %DeriveFromError: type = class_type @DeriveFromError [template]
// CHECK:STDOUT:   %.7: type = ptr_type %DeriveFromError [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseError.type: type = fn_type @AccessMemberWithInvalidBaseError [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseError: %AccessMemberWithInvalidBaseError.type = struct_value () [template]
// CHECK:STDOUT:   %DeriveFromNonType: type = class_type @DeriveFromNonType [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 32 [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.9: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.10: %.9 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = interface_type @ImplicitAs, @ImplicitAs(type) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(type) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.13: %.9 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %.14: type = ptr_type %DeriveFromNonType [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBasNonType.type: type = fn_type @AccessMemberWithInvalidBasNonType [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBasNonType: %AccessMemberWithInvalidBasNonType.type = struct_value () [template]
// CHECK:STDOUT:   %DeriveFromi32: type = class_type @DeriveFromi32 [template]
// CHECK:STDOUT:   %.15: type = ptr_type %DeriveFromi32 [template]
// CHECK:STDOUT:   %.16: type = ptr_type i32 [template]
// CHECK:STDOUT:   %ConvertToBadBasei32.type: type = fn_type @ConvertToBadBasei32 [template]
// CHECK:STDOUT:   %ConvertToBadBasei32: %ConvertToBadBasei32.type = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = interface_type @ImplicitAs, @ImplicitAs(%.16) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert, @ImplicitAs(%.16) [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.17: type = assoc_entity_type %ImplicitAs.type.4, %Convert.type.3 [template]
// CHECK:STDOUT:   %.18: %.17 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBasei32.type: type = fn_type @AccessMemberWithInvalidBasei32 [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBasei32: %AccessMemberWithInvalidBasei32.type = struct_value () [template]
// CHECK:STDOUT:   %DeriveFromTuple: type = class_type @DeriveFromTuple [template]
// CHECK:STDOUT:   %.19: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.20: type = tuple_type (%Base) [template]
// CHECK:STDOUT:   %.21: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.22: type = tuple_type (%.21) [template]
// CHECK:STDOUT:   %.23: type = ptr_type %DeriveFromTuple [template]
// CHECK:STDOUT:   %.24: type = ptr_type %.20 [template]
// CHECK:STDOUT:   %ConvertToBadBaseTuple.type: type = fn_type @ConvertToBadBaseTuple [template]
// CHECK:STDOUT:   %ConvertToBadBaseTuple: %ConvertToBadBaseTuple.type = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = interface_type @ImplicitAs, @ImplicitAs(%.24) [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert, @ImplicitAs(%.24) [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [template]
// CHECK:STDOUT:   %.25: type = assoc_entity_type %ImplicitAs.type.5, %Convert.type.4 [template]
// CHECK:STDOUT:   %.26: %.25 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseTuple.type: type = fn_type @AccessMemberWithInvalidBaseTuple [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseTuple: %AccessMemberWithInvalidBaseTuple.type = struct_value () [template]
// CHECK:STDOUT:   %DeriveFromStruct: type = class_type @DeriveFromStruct [template]
// CHECK:STDOUT:   %.27: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.28: type = ptr_type %.27 [template]
// CHECK:STDOUT:   %.29: type = ptr_type %DeriveFromStruct [template]
// CHECK:STDOUT:   %ConvertToBadBaseStruct.type: type = fn_type @ConvertToBadBaseStruct [template]
// CHECK:STDOUT:   %ConvertToBadBaseStruct: %ConvertToBadBaseStruct.type = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = interface_type @ImplicitAs, @ImplicitAs(%.28) [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert, @ImplicitAs(%.28) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.30: type = assoc_entity_type %ImplicitAs.type.6, %Convert.type.5 [template]
// CHECK:STDOUT:   %.31: %.30 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseStruct.type: type = fn_type @AccessMemberWithInvalidBaseStruct [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseStruct: %AccessMemberWithInvalidBaseStruct.type = struct_value () [template]
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
// CHECK:STDOUT:   %DeriveFromIncomplete: type = class_type @DeriveFromIncomplete [template]
// CHECK:STDOUT:   %.32: type = ptr_type %DeriveFromIncomplete [template]
// CHECK:STDOUT:   %.33: type = ptr_type %Incomplete [template]
// CHECK:STDOUT:   %ConvertToBadBaseIncomplete.type: type = fn_type @ConvertToBadBaseIncomplete [template]
// CHECK:STDOUT:   %ConvertToBadBaseIncomplete: %ConvertToBadBaseIncomplete.type = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.7: type = interface_type @ImplicitAs, @ImplicitAs(%.33) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert, @ImplicitAs(%.33) [template]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [template]
// CHECK:STDOUT:   %.34: type = assoc_entity_type %ImplicitAs.type.7, %Convert.type.6 [template]
// CHECK:STDOUT:   %.35: %.34 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseIncomplete.type: type = fn_type @AccessMemberWithInvalidBaseIncomplete [template]
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseIncomplete: %AccessMemberWithInvalidBaseIncomplete.type = struct_value () [template]
// CHECK:STDOUT:   %DeriveFromFinal: type = class_type @DeriveFromFinal [template]
// CHECK:STDOUT:   %.36: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %.37: type = unbound_element_type %DeriveFromFinal, %Final [template]
// CHECK:STDOUT:   %.38: type = struct_type {.base: %Final} [template]
// CHECK:STDOUT:   %.39: <witness> = complete_type_witness %.38 [template]
// CHECK:STDOUT:   %.40: type = ptr_type %DeriveFromFinal [template]
// CHECK:STDOUT:   %.41: type = ptr_type %Final [template]
// CHECK:STDOUT:   %ConvertToBadBaseFinal.type: type = fn_type @ConvertToBadBaseFinal [template]
// CHECK:STDOUT:   %ConvertToBadBaseFinal: %ConvertToBadBaseFinal.type = struct_value () [template]
// CHECK:STDOUT:   %.42: type = struct_type {.base: %.36} [template]
// CHECK:STDOUT:   %.43: type = ptr_type %.38 [template]
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
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+40, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.4: @ImplicitAs.%.1 (%.9) = import_ref Core//prelude/operators/as, inst+63, loaded [symbolic = @ImplicitAs.%.2 (constants.%.13)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+56, unloaded
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
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {} {}
// CHECK:STDOUT:   %Final.decl: type = class_decl @Final [template = constants.%Final] {} {}
// CHECK:STDOUT:   %DeriveFromError.decl: type = class_decl @DeriveFromError [template = constants.%DeriveFromError] {} {}
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseError.decl: %AccessMemberWithInvalidBaseError.type = fn_decl @AccessMemberWithInvalidBaseError [template = constants.%AccessMemberWithInvalidBaseError] {
// CHECK:STDOUT:     %p.patt: %.7 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromError.ref: type = name_ref DeriveFromError, file.%DeriveFromError.decl [template = constants.%DeriveFromError]
// CHECK:STDOUT:     %.loc25_55: type = ptr_type %DeriveFromError [template = constants.%.7]
// CHECK:STDOUT:     %p.param: %.7 = param p, runtime_param0
// CHECK:STDOUT:     %p: %.7 = bind_name p, %p.param
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc25_61.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc25_61.2: type = converted %int.make_type_32, %.loc25_61.1 [template = i32]
// CHECK:STDOUT:     %return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeriveFromNonType.decl: type = class_decl @DeriveFromNonType [template = constants.%DeriveFromNonType] {} {}
// CHECK:STDOUT:   %AccessMemberWithInvalidBasNonType.decl: %AccessMemberWithInvalidBasNonType.type = fn_decl @AccessMemberWithInvalidBasNonType [template = constants.%AccessMemberWithInvalidBasNonType] {
// CHECK:STDOUT:     %p.patt: %.14 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromNonType.ref: type = name_ref DeriveFromNonType, file.%DeriveFromNonType.decl [template = constants.%DeriveFromNonType]
// CHECK:STDOUT:     %.loc38_58: type = ptr_type %DeriveFromNonType [template = constants.%.14]
// CHECK:STDOUT:     %p.param: %.14 = param p, runtime_param0
// CHECK:STDOUT:     %p: %.14 = bind_name p, %p.param
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc38_64.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc38_64.2: type = converted %int.make_type_32, %.loc38_64.1 [template = i32]
// CHECK:STDOUT:     %return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeriveFromi32.decl: type = class_decl @DeriveFromi32 [template = constants.%DeriveFromi32] {} {}
// CHECK:STDOUT:   %ConvertToBadBasei32.decl: %ConvertToBadBasei32.type = fn_decl @ConvertToBadBasei32 [template = constants.%ConvertToBadBasei32] {
// CHECK:STDOUT:     %p.patt: %.15 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromi32.ref: type = name_ref DeriveFromi32, file.%DeriveFromi32.decl [template = constants.%DeriveFromi32]
// CHECK:STDOUT:     %.loc57_40: type = ptr_type %DeriveFromi32 [template = constants.%.15]
// CHECK:STDOUT:     %p.param: %.15 = param p, runtime_param0
// CHECK:STDOUT:     %p: %.15 = bind_name p, %p.param
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc57_49.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc57_49.2: type = converted %int.make_type_32, %.loc57_49.1 [template = i32]
// CHECK:STDOUT:     %.loc57_49.3: type = ptr_type i32 [template = constants.%.16]
// CHECK:STDOUT:     %return: ref %.16 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBasei32.decl: %AccessMemberWithInvalidBasei32.type = fn_decl @AccessMemberWithInvalidBasei32 [template = constants.%AccessMemberWithInvalidBasei32] {
// CHECK:STDOUT:     %p.patt: %.15 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromi32.ref: type = name_ref DeriveFromi32, file.%DeriveFromi32.decl [template = constants.%DeriveFromi32]
// CHECK:STDOUT:     %.loc59_51: type = ptr_type %DeriveFromi32 [template = constants.%.15]
// CHECK:STDOUT:     %p.param: %.15 = param p, runtime_param0
// CHECK:STDOUT:     %p: %.15 = bind_name p, %p.param
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc59_57.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc59_57.2: type = converted %int.make_type_32, %.loc59_57.1 [template = i32]
// CHECK:STDOUT:     %return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeriveFromTuple.decl: type = class_decl @DeriveFromTuple [template = constants.%DeriveFromTuple] {} {}
// CHECK:STDOUT:   %ConvertToBadBaseTuple.decl: %ConvertToBadBaseTuple.type = fn_decl @ConvertToBadBaseTuple [template = constants.%ConvertToBadBaseTuple] {
// CHECK:STDOUT:     %p.patt: %.23 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromTuple.ref: type = name_ref DeriveFromTuple, file.%DeriveFromTuple.decl [template = constants.%DeriveFromTuple]
// CHECK:STDOUT:     %.loc76_44: type = ptr_type %DeriveFromTuple [template = constants.%.23]
// CHECK:STDOUT:     %p.param: %.23 = param p, runtime_param0
// CHECK:STDOUT:     %p: %.23 = bind_name p, %p.param
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %.loc76_56: %.19 = tuple_literal (%Base.ref)
// CHECK:STDOUT:     %.loc76_57.1: type = converted %.loc76_56, constants.%.20 [template = constants.%.20]
// CHECK:STDOUT:     %.loc76_57.2: type = ptr_type %.20 [template = constants.%.24]
// CHECK:STDOUT:     %return: ref %.24 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseTuple.decl: %AccessMemberWithInvalidBaseTuple.type = fn_decl @AccessMemberWithInvalidBaseTuple [template = constants.%AccessMemberWithInvalidBaseTuple] {
// CHECK:STDOUT:     %p.patt: %.23 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromTuple.ref: type = name_ref DeriveFromTuple, file.%DeriveFromTuple.decl [template = constants.%DeriveFromTuple]
// CHECK:STDOUT:     %.loc78_55: type = ptr_type %DeriveFromTuple [template = constants.%.23]
// CHECK:STDOUT:     %p.param: %.23 = param p, runtime_param0
// CHECK:STDOUT:     %p: %.23 = bind_name p, %p.param
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc78_61.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc78_61.2: type = converted %int.make_type_32, %.loc78_61.1 [template = i32]
// CHECK:STDOUT:     %return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeriveFromStruct.decl: type = class_decl @DeriveFromStruct [template = constants.%DeriveFromStruct] {} {}
// CHECK:STDOUT:   %ConvertToBadBaseStruct.decl: %ConvertToBadBaseStruct.type = fn_decl @ConvertToBadBaseStruct [template = constants.%ConvertToBadBaseStruct] {
// CHECK:STDOUT:     %p.patt: %.29 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromStruct.ref: type = name_ref DeriveFromStruct, file.%DeriveFromStruct.decl [template = constants.%DeriveFromStruct]
// CHECK:STDOUT:     %.loc97_46: type = ptr_type %DeriveFromStruct [template = constants.%.29]
// CHECK:STDOUT:     %p.param: %.29 = param p, runtime_param0
// CHECK:STDOUT:     %p: %.29 = bind_name p, %p.param
// CHECK:STDOUT:     %int.make_type_32.loc97_57: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc97_57.1: type = value_of_initializer %int.make_type_32.loc97_57 [template = i32]
// CHECK:STDOUT:     %.loc97_57.2: type = converted %int.make_type_32.loc97_57, %.loc97_57.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc97_66: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc97_66.1: type = value_of_initializer %int.make_type_32.loc97_66 [template = i32]
// CHECK:STDOUT:     %.loc97_66.2: type = converted %int.make_type_32.loc97_66, %.loc97_66.1 [template = i32]
// CHECK:STDOUT:     %.loc97_69: type = struct_type {.a: i32, .b: i32} [template = constants.%.27]
// CHECK:STDOUT:     %.loc97_70: type = ptr_type %.27 [template = constants.%.28]
// CHECK:STDOUT:     %return: ref %.28 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseStruct.decl: %AccessMemberWithInvalidBaseStruct.type = fn_decl @AccessMemberWithInvalidBaseStruct [template = constants.%AccessMemberWithInvalidBaseStruct] {
// CHECK:STDOUT:     %p.patt: %.29 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromStruct.ref: type = name_ref DeriveFromStruct, file.%DeriveFromStruct.decl [template = constants.%DeriveFromStruct]
// CHECK:STDOUT:     %.loc100_57: type = ptr_type %DeriveFromStruct [template = constants.%.29]
// CHECK:STDOUT:     %p.param: %.29 = param p, runtime_param0
// CHECK:STDOUT:     %p: %.29 = bind_name p, %p.param
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc100_63.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc100_63.2: type = converted %int.make_type_32, %.loc100_63.1 [template = i32]
// CHECK:STDOUT:     %return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Incomplete.decl: type = class_decl @Incomplete [template = constants.%Incomplete] {} {}
// CHECK:STDOUT:   %DeriveFromIncomplete.decl: type = class_decl @DeriveFromIncomplete [template = constants.%DeriveFromIncomplete] {} {}
// CHECK:STDOUT:   %ConvertToBadBaseIncomplete.decl: %ConvertToBadBaseIncomplete.type = fn_decl @ConvertToBadBaseIncomplete [template = constants.%ConvertToBadBaseIncomplete] {
// CHECK:STDOUT:     %p.patt: %.32 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromIncomplete.ref: type = name_ref DeriveFromIncomplete, file.%DeriveFromIncomplete.decl [template = constants.%DeriveFromIncomplete]
// CHECK:STDOUT:     %.loc122_54: type = ptr_type %DeriveFromIncomplete [template = constants.%.32]
// CHECK:STDOUT:     %p.param: %.32 = param p, runtime_param0
// CHECK:STDOUT:     %p: %.32 = bind_name p, %p.param
// CHECK:STDOUT:     %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl [template = constants.%Incomplete]
// CHECK:STDOUT:     %.loc122_70: type = ptr_type %Incomplete [template = constants.%.33]
// CHECK:STDOUT:     %return: ref %.33 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseIncomplete.decl: %AccessMemberWithInvalidBaseIncomplete.type = fn_decl @AccessMemberWithInvalidBaseIncomplete [template = constants.%AccessMemberWithInvalidBaseIncomplete] {
// CHECK:STDOUT:     %p.patt: %.32 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromIncomplete.ref: type = name_ref DeriveFromIncomplete, file.%DeriveFromIncomplete.decl [template = constants.%DeriveFromIncomplete]
// CHECK:STDOUT:     %.loc124_65: type = ptr_type %DeriveFromIncomplete [template = constants.%.32]
// CHECK:STDOUT:     %p.param: %.32 = param p, runtime_param0
// CHECK:STDOUT:     %p: %.32 = bind_name p, %p.param
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc124_71.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc124_71.2: type = converted %int.make_type_32, %.loc124_71.1 [template = i32]
// CHECK:STDOUT:     %return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeriveFromFinal.decl: type = class_decl @DeriveFromFinal [template = constants.%DeriveFromFinal] {} {}
// CHECK:STDOUT:   %ConvertToBadBaseFinal.decl: %ConvertToBadBaseFinal.type = fn_decl @ConvertToBadBaseFinal [template = constants.%ConvertToBadBaseFinal] {
// CHECK:STDOUT:     %p.patt: %.40 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromFinal.ref: type = name_ref DeriveFromFinal, file.%DeriveFromFinal.decl [template = constants.%DeriveFromFinal]
// CHECK:STDOUT:     %.loc135_44: type = ptr_type %DeriveFromFinal [template = constants.%.40]
// CHECK:STDOUT:     %p.param: %.40 = param p, runtime_param0
// CHECK:STDOUT:     %p: %.40 = bind_name p, %p.param
// CHECK:STDOUT:     %Final.ref: type = name_ref Final, file.%Final.decl [template = constants.%Final]
// CHECK:STDOUT:     %.loc135_55: type = ptr_type %Final [template = constants.%.41]
// CHECK:STDOUT:     %return: ref %.41 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_WithMember.decl: %AccessMemberWithInvalidBaseFinal_WithMember.type = fn_decl @AccessMemberWithInvalidBaseFinal_WithMember [template = constants.%AccessMemberWithInvalidBaseFinal_WithMember] {
// CHECK:STDOUT:     %p.patt: %.40 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromFinal.ref: type = name_ref DeriveFromFinal, file.%DeriveFromFinal.decl [template = constants.%DeriveFromFinal]
// CHECK:STDOUT:     %.loc139_66: type = ptr_type %DeriveFromFinal [template = constants.%.40]
// CHECK:STDOUT:     %p.param: %.40 = param p, runtime_param0
// CHECK:STDOUT:     %p: %.40 = bind_name p, %p.param
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc139_72.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc139_72.2: type = converted %int.make_type_32, %.loc139_72.1 [template = i32]
// CHECK:STDOUT:     %return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMemberWithInvalidBaseFinal_NoMember.decl: %AccessMemberWithInvalidBaseFinal_NoMember.type = fn_decl @AccessMemberWithInvalidBaseFinal_NoMember [template = constants.%AccessMemberWithInvalidBaseFinal_NoMember] {
// CHECK:STDOUT:     %p.patt: %.40 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %DeriveFromFinal.ref: type = name_ref DeriveFromFinal, file.%DeriveFromFinal.decl [template = constants.%DeriveFromFinal]
// CHECK:STDOUT:     %.loc143_64: type = ptr_type %DeriveFromFinal [template = constants.%.40]
// CHECK:STDOUT:     %p.param: %.40 = param p, runtime_param0
// CHECK:STDOUT:     %p: %.40 = bind_name p, %p.param
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc143_70.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc143_70.2: type = converted %int.make_type_32, %.loc143_70.1 [template = i32]
// CHECK:STDOUT:     %return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.9) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .Convert = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc11: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Final {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc13_10.2: type = converted %int.make_type_32, %.loc13_10.1 [template = i32]
// CHECK:STDOUT:   %.loc13_8: %.4 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc14: <witness> = complete_type_witness %.5 [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Final
// CHECK:STDOUT:   .a = %.loc13_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromError {
// CHECK:STDOUT:   %error.ref: <error> = name_ref error, <error> [template = <error>]
// CHECK:STDOUT:   %.loc21: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:   %.loc22: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromError
// CHECK:STDOUT:   .base = %.loc21
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromNonType {
// CHECK:STDOUT:   %.loc35_16.1: i32 = int_literal 32 [template = constants.%.8]
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(type) [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %.loc35_16.2: %.11 = specific_constant imports.%import_ref.4, @ImplicitAs(type) [template = constants.%.12]
// CHECK:STDOUT:   %Convert.ref: %.11 = name_ref Convert, %.loc35_16.2 [template = constants.%.12]
// CHECK:STDOUT:   %.loc35_16.3: type = converted %.loc35_16.1, <error> [template = <error>]
// CHECK:STDOUT:   %.loc35_18: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:   %.loc36: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromNonType
// CHECK:STDOUT:   .base = %.loc35_18
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromi32 {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc45_16.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc45_16.2: type = converted %int.make_type_32, %.loc45_16.1 [template = i32]
// CHECK:STDOUT:   %.loc45_19: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:   %.loc46: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromi32
// CHECK:STDOUT:   .base = %.loc45_19
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromTuple {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc66_22.1: %.19 = tuple_literal (%Base.ref)
// CHECK:STDOUT:   %.loc66_22.2: type = converted %.loc66_22.1, constants.%.20 [template = constants.%.20]
// CHECK:STDOUT:   %.loc66_23: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:   %.loc67: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromTuple
// CHECK:STDOUT:   .base = %.loc66_23
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromStruct {
// CHECK:STDOUT:   %int.make_type_32.loc87_21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc87_21.1: type = value_of_initializer %int.make_type_32.loc87_21 [template = i32]
// CHECK:STDOUT:   %.loc87_21.2: type = converted %int.make_type_32.loc87_21, %.loc87_21.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc87_30: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc87_30.1: type = value_of_initializer %int.make_type_32.loc87_30 [template = i32]
// CHECK:STDOUT:   %.loc87_30.2: type = converted %int.make_type_32.loc87_30, %.loc87_30.1 [template = i32]
// CHECK:STDOUT:   %.loc87_33: type = struct_type {.a: i32, .b: i32} [template = constants.%.27]
// CHECK:STDOUT:   %.loc87_34: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:   %.loc88: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromStruct
// CHECK:STDOUT:   .base = %.loc87_34
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromIncomplete {
// CHECK:STDOUT:   %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl [template = constants.%Incomplete]
// CHECK:STDOUT:   %.loc112: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:   %.loc113: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromIncomplete
// CHECK:STDOUT:   .base = %.loc112
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromFinal {
// CHECK:STDOUT:   %Final.ref: type = name_ref Final, file.%Final.decl [template = constants.%Final]
// CHECK:STDOUT:   %.loc131: %.37 = base_decl %Final, element0 [template]
// CHECK:STDOUT:   %.loc132: <witness> = complete_type_witness %.38 [template = constants.%.39]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%DeriveFromFinal
// CHECK:STDOUT:   .base = %.loc131
// CHECK:STDOUT:   extend name_scope3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseError(%p: %.7) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.7 = name_ref p, %p
// CHECK:STDOUT:   %.loc25_75: ref %DeriveFromError = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.2)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBasNonType(%p: %.14) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.14 = name_ref p, %p
// CHECK:STDOUT:   %.loc38_78: ref %DeriveFromNonType = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBasei32(%p: %.15) -> %.16 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.15 = name_ref p, %p
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(constants.%.16) [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %.loc57_61.1: %.17 = specific_constant imports.%import_ref.4, @ImplicitAs(constants.%.16) [template = constants.%.18]
// CHECK:STDOUT:   %Convert.ref: %.17 = name_ref Convert, %.loc57_61.1 [template = constants.%.18]
// CHECK:STDOUT:   %.loc57_61.2: %.16 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBasei32(%p: %.15) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.15 = name_ref p, %p
// CHECK:STDOUT:   %.loc59_71: ref %DeriveFromi32 = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseTuple(%p: %.23) -> %.24 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.23 = name_ref p, %p
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(constants.%.24) [template = constants.%ImplicitAs.type.5]
// CHECK:STDOUT:   %.loc76_69.1: %.25 = specific_constant imports.%import_ref.4, @ImplicitAs(constants.%.24) [template = constants.%.26]
// CHECK:STDOUT:   %Convert.ref: %.25 = name_ref Convert, %.loc76_69.1 [template = constants.%.26]
// CHECK:STDOUT:   %.loc76_69.2: %.24 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseTuple(%p: %.23) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.23 = name_ref p, %p
// CHECK:STDOUT:   %.loc78_75: ref %DeriveFromTuple = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseStruct(%p: %.29) -> %.28 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.29 = name_ref p, %p
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(constants.%.28) [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %.loc97_82.1: %.30 = specific_constant imports.%import_ref.4, @ImplicitAs(constants.%.28) [template = constants.%.31]
// CHECK:STDOUT:   %Convert.ref: %.30 = name_ref Convert, %.loc97_82.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc97_82.2: %.28 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseStruct(%p: %.29) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.29 = name_ref p, %p
// CHECK:STDOUT:   %.loc100_77: ref %DeriveFromStruct = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseIncomplete(%p: %.32) -> %.33 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.32 = name_ref p, %p
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(constants.%.33) [template = constants.%ImplicitAs.type.7]
// CHECK:STDOUT:   %.loc122_82.1: %.34 = specific_constant imports.%import_ref.4, @ImplicitAs(constants.%.33) [template = constants.%.35]
// CHECK:STDOUT:   %Convert.ref: %.34 = name_ref Convert, %.loc122_82.1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc122_82.2: %.33 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseIncomplete(%p: %.32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.32 = name_ref p, %p
// CHECK:STDOUT:   %.loc124_85: ref %DeriveFromIncomplete = deref %p.ref
// CHECK:STDOUT:   %n.ref: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToBadBaseFinal(%p: %.40) -> %.41 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.40 = name_ref p, %p
// CHECK:STDOUT:   %.loc136_11.1: ref %DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %.loc136_11.2: ref %Final = class_element_access %.loc136_11.1, element0
// CHECK:STDOUT:   %.loc136_11.3: %.41 = addr_of %.loc136_11.2
// CHECK:STDOUT:   %.loc136_11.4: %.41 = converted %p.ref, %.loc136_11.3
// CHECK:STDOUT:   return %.loc136_11.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseFinal_WithMember(%p: %.40) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.40 = name_ref p, %p
// CHECK:STDOUT:   %.loc140_11: ref %DeriveFromFinal = deref %p.ref
// CHECK:STDOUT:   %a.ref: %.4 = name_ref a, @Final.%.loc13_8 [template = @Final.%.loc13_8]
// CHECK:STDOUT:   %.loc140_14.1: ref %Final = class_element_access %.loc140_11, element0
// CHECK:STDOUT:   %.loc140_14.2: ref %Final = converted %.loc140_11, %.loc140_14.1
// CHECK:STDOUT:   %.loc140_14.3: ref i32 = class_element_access %.loc140_14.2, element0
// CHECK:STDOUT:   %.loc140_14.4: i32 = bind_value %.loc140_14.3
// CHECK:STDOUT:   return %.loc140_14.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMemberWithInvalidBaseFinal_NoMember(%p: %.40) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.40 = name_ref p, %p
// CHECK:STDOUT:   %.loc147: ref %DeriveFromFinal = deref %p.ref
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
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(type) {
// CHECK:STDOUT:   %Dest => type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT:   %.2 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.16) {
// CHECK:STDOUT:   %Dest => constants.%.16
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.3
// CHECK:STDOUT:   %Convert => constants.%Convert.3
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT:   %.2 => constants.%.18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.24) {
// CHECK:STDOUT:   %Dest => constants.%.24
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.4
// CHECK:STDOUT:   %Convert => constants.%Convert.4
// CHECK:STDOUT:   %.1 => constants.%.25
// CHECK:STDOUT:   %.2 => constants.%.26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.28) {
// CHECK:STDOUT:   %Dest => constants.%.28
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.6
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.5
// CHECK:STDOUT:   %Convert => constants.%Convert.5
// CHECK:STDOUT:   %.1 => constants.%.30
// CHECK:STDOUT:   %.2 => constants.%.31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.33) {
// CHECK:STDOUT:   %Dest => constants.%.33
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.7
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.6
// CHECK:STDOUT:   %Convert => constants.%Convert.6
// CHECK:STDOUT:   %.1 => constants.%.34
// CHECK:STDOUT:   %.2 => constants.%.35
// CHECK:STDOUT: }
// CHECK:STDOUT:
