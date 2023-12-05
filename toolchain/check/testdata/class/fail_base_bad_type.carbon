// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class Base {}
class Final {}

class DeriveFromNonType {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:11: ERROR: Cannot implicitly convert from `i32` to `type`.
  // CHECK:STDERR:   base: 32;
  // CHECK:STDERR:           ^
  base: 32;
}

class DeriveFromi32 {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:12: ERROR: Deriving from final type `i32`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   base: i32;
  // CHECK:STDERR:            ^
  base: i32;
}

class DeriveFromTuple {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:16: ERROR: Deriving from final type `(Base,)`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   base: (Base,);
  // CHECK:STDERR:                ^
  base: (Base,);
}

// TODO: Should we allow this?
// We do allow `{.base = {.a: i32, .b: i32}}`.
class DeriveFromStruct {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:27: ERROR: Deriving from final type `{.a: i32, .b: i32}`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   base: {.a: i32, .b: i32};
  // CHECK:STDERR:                           ^
  base: {.a: i32, .b: i32};
}

base class Incomplete;

class DeriveFromIncomplete {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+6]]:19: ERROR: Base `Incomplete` is an incomplete type.
  // CHECK:STDERR:   base: Incomplete;
  // CHECK:STDERR:                   ^
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE-6]]:1: Class was forward declared here.
  // CHECK:STDERR: base class Incomplete;
  // CHECK:STDERR: ^~~~
  base: Incomplete;
}

class DeriveFromFinal {
  // CHECK:STDERR: fail_base_bad_type.carbon:[[@LINE+3]]:14: ERROR: Deriving from final type `Final`. Base type must be an `abstract` or `base` class.
  // CHECK:STDERR:   base: Final;
  // CHECK:STDERR:              ^
  base: Final;
}

// CHECK:STDOUT: --- fail_base_bad_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc7_18.1: type = struct_type {}
// CHECK:STDOUT:   %.loc15: type = struct_type {.base: <error>}
// CHECK:STDOUT:   %.loc22: type = struct_type {.base: i32}
// CHECK:STDOUT:   %.loc28_15: type = tuple_type (type)
// CHECK:STDOUT:   %.loc28_16.1: type = tuple_type (Base)
// CHECK:STDOUT:   %.loc7_18.2: type = tuple_type ()
// CHECK:STDOUT:   %.loc7_1: type = ptr_type {}
// CHECK:STDOUT:   %.loc28_16.2: type = tuple_type ({}*)
// CHECK:STDOUT:   %.loc29: type = struct_type {.base: (Base,)}
// CHECK:STDOUT:   %.loc37: type = ptr_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc38: type = struct_type {.base: {.a: i32, .b: i32}}
// CHECK:STDOUT:   %.loc57: type = struct_type {.base: Final}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Base = %Base.decl, .Final = %Final.decl, .DeriveFromNonType = %DeriveFromNonType.decl, .DeriveFromi32 = %DeriveFromi32.decl, .DeriveFromTuple = %DeriveFromTuple.decl, .DeriveFromStruct = %DeriveFromStruct.decl, .Incomplete = %Incomplete.decl, .DeriveFromIncomplete = %DeriveFromIncomplete.decl, .DeriveFromFinal = %DeriveFromFinal.decl}
// CHECK:STDOUT:   %Base.decl = class_decl @Base, ()
// CHECK:STDOUT:   %Base: type = class_type @Base
// CHECK:STDOUT:   %Final.decl = class_decl @Final, ()
// CHECK:STDOUT:   %Final: type = class_type @Final
// CHECK:STDOUT:   %DeriveFromNonType.decl = class_decl @DeriveFromNonType, ()
// CHECK:STDOUT:   %DeriveFromNonType: type = class_type @DeriveFromNonType
// CHECK:STDOUT:   %DeriveFromi32.decl = class_decl @DeriveFromi32, ()
// CHECK:STDOUT:   %DeriveFromi32: type = class_type @DeriveFromi32
// CHECK:STDOUT:   %DeriveFromTuple.decl = class_decl @DeriveFromTuple, ()
// CHECK:STDOUT:   %DeriveFromTuple: type = class_type @DeriveFromTuple
// CHECK:STDOUT:   %DeriveFromStruct.decl = class_decl @DeriveFromStruct, ()
// CHECK:STDOUT:   %DeriveFromStruct: type = class_type @DeriveFromStruct
// CHECK:STDOUT:   %Incomplete.decl = class_decl @Incomplete, ()
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete
// CHECK:STDOUT:   %DeriveFromIncomplete.decl = class_decl @DeriveFromIncomplete, ()
// CHECK:STDOUT:   %DeriveFromIncomplete: type = class_type @DeriveFromIncomplete
// CHECK:STDOUT:   %DeriveFromFinal.decl = class_decl @DeriveFromFinal, ()
// CHECK:STDOUT:   %DeriveFromFinal: type = class_type @DeriveFromFinal
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
// CHECK:STDOUT:   %.loc14_9: i32 = int_literal 32
// CHECK:STDOUT:   %.loc14_11.1: type = unbound_element_type DeriveFromNonType, <error>
// CHECK:STDOUT:   %.loc14_11.2: <unbound element of class DeriveFromNonType> = base_decl <error>, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc14_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromi32 {
// CHECK:STDOUT:   %.loc21_12.1: type = unbound_element_type DeriveFromi32, i32
// CHECK:STDOUT:   %.loc21_12.2: <unbound element of class DeriveFromi32> = base_decl i32, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc21_12.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromTuple {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base
// CHECK:STDOUT:   %.loc28_15: (type,) = tuple_literal (%Base.ref)
// CHECK:STDOUT:   %.loc28_16.1: type = converted %.loc28_15, constants.%.loc28_16.1
// CHECK:STDOUT:   %.loc28_16.2: type = unbound_element_type DeriveFromTuple, (Base,)
// CHECK:STDOUT:   %.loc28_16.3: <unbound element of class DeriveFromTuple> = base_decl (Base,), element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc28_16.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromStruct {
// CHECK:STDOUT:   %.loc37_26: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc37_27.1: type = unbound_element_type DeriveFromStruct, {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc37_27.2: <unbound element of class DeriveFromStruct> = base_decl {.a: i32, .b: i32}, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc37_27.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromIncomplete {
// CHECK:STDOUT:   %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete
// CHECK:STDOUT:   %.loc49_19.1: type = unbound_element_type DeriveFromIncomplete, <error>
// CHECK:STDOUT:   %.loc49_19.2: <unbound element of class DeriveFromIncomplete> = base_decl <error>, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc49_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DeriveFromFinal {
// CHECK:STDOUT:   %Final.ref: type = name_ref Final, file.%Final
// CHECK:STDOUT:   %.loc56_14.1: type = unbound_element_type DeriveFromFinal, Final
// CHECK:STDOUT:   %.loc56_14.2: <unbound element of class DeriveFromFinal> = base_decl Final, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc56_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
