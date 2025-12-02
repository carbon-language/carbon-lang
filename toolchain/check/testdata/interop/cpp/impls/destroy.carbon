// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// EXTRA-ARGS: --clang-arg=--std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/impls/destroy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/impls/destroy.carbon

// --- types.h

class PublicDestructor {
 public:
  ~PublicDestructor();
};

class TrivialDestructor {
 public:
  ~TrivialDestructor() = default;
};

class DeletedDestructor {
 public:
  ~DeletedDestructor() = delete;
};

template<int N = 0>
class AmbiguousDestructorT {
 public:
  ~AmbiguousDestructorT() requires (N != 1);
  ~AmbiguousDestructorT() requires (N != 2);
};

using AmbiguousDestructor = AmbiguousDestructorT<>;

class PrivateDestructor {
 private:
  ~PrivateDestructor();
};

class ProtectedDestructor {
 protected:
  ~ProtectedDestructor();
};

// --- destroy_destroyable.carbon

library "[[@TEST_NAME]]";

import Cpp library "types.h";

fn PublicDestroy() {
  //@dump-sem-ir-begin
  var a: Cpp.PublicDestructor = {};
  //@dump-sem-ir-end
}

fn TrivialDestroy() {
  //@dump-sem-ir-begin
  var a: Cpp.TrivialDestructor = {};
  //@dump-sem-ir-end
}

// --- fail_destroy_nondestroyable.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+16]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./types.h:18:7: error: destructor of class 'AmbiguousDestructorT<>' is ambiguous [CppInteropParseError]
// CHECK:STDERR:    18 | class AmbiguousDestructorT {
// CHECK:STDERR:       |       ^
// CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+12]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./types.h:18:7: note: in instantiation of template class 'AmbiguousDestructorT<>' requested here [CppInteropParseNote]
// CHECK:STDERR:    18 | class AmbiguousDestructorT {
// CHECK:STDERR:       |       ^
// CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./types.h:20:3: note: candidate function [CppInteropParseNote]
// CHECK:STDERR:    20 |   ~AmbiguousDestructorT() requires (N != 1);
// CHECK:STDERR:       |   ^
// CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./types.h:21:3: note: candidate function [CppInteropParseNote]
// CHECK:STDERR:    21 |   ~AmbiguousDestructorT() requires (N != 2);
// CHECK:STDERR:       |   ^
import Cpp library "types.h";

fn DeletedDestroy() {
  var a: Cpp.DeletedDestructor = {};
}

fn AmbiguousDestroy() {
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+12]]:10: note: while completing C++ type `Cpp.AmbiguousDestructorT` [InCppTypeCompletion]
  // CHECK:STDERR:   var a: Cpp.AmbiguousDestructor = {};
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE-8]]:3: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    23 |   var a: Cpp.DeletedDestructor = {};
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE-14]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:14:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:    14 |   ~DeletedDestructor() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR:
  var a: Cpp.AmbiguousDestructor = {};
}

fn PrivateDestroy() {
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+5]]:3: error: cannot access private member `<C++ destructor>` of type `Cpp.PrivateDestructor` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   var a: Cpp.PrivateDestructor = {};
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  var a: Cpp.PrivateDestructor = {};
}

fn ProtectedDestroy() {
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+5]]:3: error: cannot access protected member `<C++ destructor>` of type `Cpp.ProtectedDestructor` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   var a: Cpp.ProtectedDestructor = {};
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  var a: Cpp.ProtectedDestructor = {};
}

// --- fail_todo_destroy_generically.carbon

library "[[@TEST_NAME]]";

import Cpp library "types.h";

fn Destroy[T:! Core.Copy & Core.Destroy](c: T) {
  var d: T = c;
}

fn DoDestroy() {
  // TODO: We perform multiple impl lookups for `Cpp.PublicDestructor as
  // Core.Destroy` here, and each one produces a different impl witness,
  // resulting in a deduction failure.
  // CHECK:STDERR: fail_todo_destroy_generically.carbon:[[@LINE+7]]:3: error: inconsistent deductions for value of generic parameter `T` [DeductionInconsistent]
  // CHECK:STDERR:   Destroy({} as Cpp.PublicDestructor);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_destroy_generically.carbon:[[@LINE-11]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Destroy[T:! Core.Copy & Core.Destroy](c: T) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Destroy({} as Cpp.PublicDestructor);
}

class Wrap(T:! Core.Destroy) {}

// TODO: This should type-check: each conversion of `Cpp.PublicDestructor` to
// `Core.Destroy` should produce the same value.
fn EqualWitnesses(p: Wrap(Cpp.PublicDestructor)*) -> Wrap(Cpp.PublicDestructor)* {
  // CHECK:STDERR: fail_todo_destroy_generically.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Wrap(Cpp.PublicDestructor as Core.Destroy)*` to `Wrap(Cpp.PublicDestructor as Core.Destroy)*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_destroy_generically.carbon:[[@LINE+4]]:3: note: type `Wrap(Cpp.PublicDestructor as Core.Destroy)*` does not implement interface `Core.ImplicitAs(Wrap(Cpp.PublicDestructor as Core.Destroy)*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

// CHECK:STDOUT: --- destroy_destroyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %PublicDestructor: type = class_type @PublicDestructor [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.27d: type = pattern_type %PublicDestructor [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %PublicDestructor.val: %PublicDestructor = struct_value () [concrete]
// CHECK:STDOUT:   %PublicDestructor.cpp_destructor.type: type = fn_type @PublicDestructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %PublicDestructor.cpp_destructor: %PublicDestructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %impl_witness.b8f: <witness> = impl_witness @PublicDestroy.%impl_witness_table [concrete]
// CHECK:STDOUT:   %TrivialDestructor: type = class_type @TrivialDestructor [concrete]
// CHECK:STDOUT:   %pattern_type.1b8: type = pattern_type %TrivialDestructor [concrete]
// CHECK:STDOUT:   %TrivialDestructor.val: %TrivialDestructor = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestructor.cpp_destructor.type: type = fn_type @TrivialDestructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %TrivialDestructor.cpp_destructor: %TrivialDestructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %impl_witness.2ea: <witness> = impl_witness @TrivialDestroy.%impl_witness_table [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .PublicDestructor = %PublicDestructor.decl
// CHECK:STDOUT:     .TrivialDestructor = %TrivialDestructor.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PublicDestructor.decl: type = class_decl @PublicDestructor [concrete = constants.%PublicDestructor] {} {}
// CHECK:STDOUT:   %PublicDestructor.cpp_destructor.decl: %PublicDestructor.cpp_destructor.type = fn_decl @PublicDestructor.cpp_destructor [concrete = constants.%PublicDestructor.cpp_destructor] {
// CHECK:STDOUT:     %self.patt: %pattern_type.27d = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.27d = ref_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %PublicDestructor = ref_param call_param0
// CHECK:STDOUT:     %self: ref %PublicDestructor = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TrivialDestructor.decl: type = class_decl @TrivialDestructor [concrete = constants.%TrivialDestructor] {} {}
// CHECK:STDOUT:   %TrivialDestructor.cpp_destructor.decl: %TrivialDestructor.cpp_destructor.type = fn_decl @TrivialDestructor.cpp_destructor [concrete = constants.%TrivialDestructor.cpp_destructor] {
// CHECK:STDOUT:     %self.patt: %pattern_type.1b8 = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1b8 = ref_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %TrivialDestructor = ref_param call_param0
// CHECK:STDOUT:     %self: ref %TrivialDestructor = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PublicDestroy() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.27d = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.27d = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %PublicDestructor = var %a.var_patt
// CHECK:STDOUT:   %.loc8_34.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_34.2: init %PublicDestructor = class_init (), %a.var [concrete = constants.%PublicDestructor.val]
// CHECK:STDOUT:   %.loc8_3: init %PublicDestructor = converted %.loc8_34.1, %.loc8_34.2 [concrete = constants.%PublicDestructor.val]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %PublicDestructor.ref [concrete = constants.%PublicDestructor] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %PublicDestructor.ref: type = name_ref PublicDestructor, imports.%PublicDestructor.decl [concrete = constants.%PublicDestructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %PublicDestructor = ref_binding a, %a.var
// CHECK:STDOUT:   %impl_witness_table = impl_witness_table (imports.%PublicDestructor.cpp_destructor.decl), invalid [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness %impl_witness_table [concrete = constants.%impl_witness.b8f]
// CHECK:STDOUT:   %PublicDestructor.cpp_destructor.bound: <bound method> = bound_method %a.var, constants.%PublicDestructor.cpp_destructor
// CHECK:STDOUT:   %PublicDestructor.cpp_destructor.call: init %empty_tuple.type = call %PublicDestructor.cpp_destructor.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TrivialDestroy() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.1b8 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.1b8 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %TrivialDestructor = var %a.var_patt
// CHECK:STDOUT:   %.loc14_35.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc14_35.2: init %TrivialDestructor = class_init (), %a.var [concrete = constants.%TrivialDestructor.val]
// CHECK:STDOUT:   %.loc14_3: init %TrivialDestructor = converted %.loc14_35.1, %.loc14_35.2 [concrete = constants.%TrivialDestructor.val]
// CHECK:STDOUT:   assign %a.var, %.loc14_3
// CHECK:STDOUT:   %.loc14_13: type = splice_block %TrivialDestructor.ref [concrete = constants.%TrivialDestructor] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %TrivialDestructor.ref: type = name_ref TrivialDestructor, imports.%TrivialDestructor.decl [concrete = constants.%TrivialDestructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %TrivialDestructor = ref_binding a, %a.var
// CHECK:STDOUT:   %impl_witness_table = impl_witness_table (imports.%TrivialDestructor.cpp_destructor.decl), invalid [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness %impl_witness_table [concrete = constants.%impl_witness.2ea]
// CHECK:STDOUT:   %TrivialDestructor.cpp_destructor.bound: <bound method> = bound_method %a.var, constants.%TrivialDestructor.cpp_destructor
// CHECK:STDOUT:   %TrivialDestructor.cpp_destructor.call: init %empty_tuple.type = call %TrivialDestructor.cpp_destructor.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
