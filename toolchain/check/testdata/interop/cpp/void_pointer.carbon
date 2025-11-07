// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/void_pointer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/void_pointer.carbon

// --- fail_todo_implicit_as_from_cpp_class_pointer_to_void_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct S {};
''';

fn F(input: Cpp.S*) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_implicit_as_from_cpp_class_pointer_to_void_pointer.carbon:[[@LINE+7]]:22: error: cannot implicitly convert expression of type `Cpp.S*` to `Cpp.void*` [ConversionFailure]
  // CHECK:STDERR:   let p: Cpp.void* = input;
  // CHECK:STDERR:                      ^~~~~
  // CHECK:STDERR: fail_todo_implicit_as_from_cpp_class_pointer_to_void_pointer.carbon:[[@LINE+4]]:22: note: type `Cpp.S*` does not implement interface `Core.ImplicitAs(Cpp.void*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let p: Cpp.void* = input;
  // CHECK:STDERR:                      ^~~~~
  // CHECK:STDERR:
  let p: Cpp.void* = input;
  //@dump-sem-ir-end
}

// --- fail_todo_implicit_as_from_carbon_class_pointer_to_void_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp;

class C {}

fn F(input: C*) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_implicit_as_from_carbon_class_pointer_to_void_pointer.carbon:[[@LINE+7]]:22: error: cannot implicitly convert expression of type `C*` to `Cpp.void*` [ConversionFailure]
  // CHECK:STDERR:   let p: Cpp.void* = input;
  // CHECK:STDERR:                      ^~~~~
  // CHECK:STDERR: fail_todo_implicit_as_from_carbon_class_pointer_to_void_pointer.carbon:[[@LINE+4]]:22: note: type `C*` does not implement interface `Core.ImplicitAs(Cpp.void*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let p: Cpp.void* = input;
  // CHECK:STDERR:                      ^~~~~
  // CHECK:STDERR:
  let p: Cpp.void* = input;
  //@dump-sem-ir-end
}

// --- fail_todo_explicit_as_from_void_pointer_to_cpp_class_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct S {};
''';

fn F(input: Cpp.void*) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_explicit_as_from_void_pointer_to_cpp_class_pointer.carbon:[[@LINE+7]]:19: error: cannot convert expression of type `Cpp.void*` to `Cpp.S*` with `as` [ConversionFailure]
  // CHECK:STDERR:   let p: Cpp.S* = input as Cpp.S*;
  // CHECK:STDERR:                   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_explicit_as_from_void_pointer_to_cpp_class_pointer.carbon:[[@LINE+4]]:19: note: type `Cpp.void*` does not implement interface `Core.As(Cpp.S*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let p: Cpp.S* = input as Cpp.S*;
  // CHECK:STDERR:                   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let p: Cpp.S* = input as Cpp.S*;
  //@dump-sem-ir-end
}

// --- fail_todo_explicit_as_from_void_pointer_to_carbon_class_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp;

class C {}

fn F(input: Cpp.void*) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_explicit_as_from_void_pointer_to_carbon_class_pointer.carbon:[[@LINE+7]]:15: error: cannot convert expression of type `Cpp.void*` to `C*` with `as` [ConversionFailure]
  // CHECK:STDERR:   let p: C* = input as C*;
  // CHECK:STDERR:               ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_explicit_as_from_void_pointer_to_carbon_class_pointer.carbon:[[@LINE+4]]:15: note: type `Cpp.void*` does not implement interface `Core.As(C*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let p: C* = input as C*;
  // CHECK:STDERR:               ^~~~~~~~~~~
  // CHECK:STDERR:
  let p: C* = input as C*;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- fail_todo_implicit_as_from_cpp_class_pointer_to_void_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %ptr.03c: type = ptr_type Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.1f6: type = pattern_type %ptr.03c [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .void = Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.5c7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.1f6 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.5c7 = name_ref input, %input
// CHECK:STDOUT:   %.loc17_18: type = splice_block %ptr.loc17 [concrete = constants.%ptr.03c] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, Cpp.void [concrete = Cpp.void]
// CHECK:STDOUT:     %ptr.loc17: type = ptr_type %void.ref [concrete = constants.%ptr.03c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17_22: %ptr.03c = converted %input.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %p: %ptr.03c = value_binding p, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_implicit_as_from_carbon_class_pointer_to_void_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %ptr.03c: type = ptr_type Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.1f6: type = pattern_type %ptr.03c [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.019) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.1f6 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.019 = name_ref input, %input
// CHECK:STDOUT:   %.loc17_18: type = splice_block %ptr.loc17 [concrete = constants.%ptr.03c] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, Cpp.void [concrete = Cpp.void]
// CHECK:STDOUT:     %ptr.loc17: type = ptr_type %void.ref [concrete = constants.%ptr.03c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17_22: %ptr.03c = converted %input.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %p: %ptr.03c = value_binding p, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_explicit_as_from_void_pointer_to_cpp_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ptr.03c: type = ptr_type Cpp.void [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.259: type = pattern_type %ptr.5c7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = Cpp.void
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.03c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.259 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.03c = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc17_28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc17_31: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %ptr.loc17_33: type = ptr_type %S.ref.loc17_31 [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:   %.loc17_25: %ptr.5c7 = converted %input.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc17_15: type = splice_block %ptr.loc17_15 [concrete = constants.%ptr.5c7] {
// CHECK:STDOUT:     %Cpp.ref.loc17_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc17_13: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr.loc17_15: type = ptr_type %S.ref.loc17_13 [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.5c7 = value_binding p, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_explicit_as_from_void_pointer_to_carbon_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.03c: type = ptr_type Cpp.void [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.03c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.44a = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.03c = name_ref input, %input
// CHECK:STDOUT:   %C.ref.loc17_24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %ptr.loc17_25: type = ptr_type %C.ref.loc17_24 [concrete = constants.%ptr.019]
// CHECK:STDOUT:   %.loc17_21: %ptr.019 = converted %input.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc17_11: type = splice_block %ptr.loc17_11 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:     %C.ref.loc17_10: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc17_11: type = ptr_type %C.ref.loc17_10 [concrete = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.019 = value_binding p, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
