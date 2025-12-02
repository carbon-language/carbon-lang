// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/impls/copy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/impls/copy.carbon

// --- types.h

class Copyable {
 public:
  Copyable(const Copyable&);
};

class ExplicitCopy {
 public:
  explicit ExplicitCopy(const ExplicitCopy&);
};

class DeletedCopy {
 public:
  DeletedCopy(const DeletedCopy&) = delete;
};

class MoveOnly {
 public:
  MoveOnly(MoveOnly&&);
};

class DeletedMove {
 public:
  DeletedMove(DeletedMove&&) = delete;
};

class NonConstCopy {
 public:
  NonConstCopy(NonConstCopy&);
};

class AmbiguousCopy {
 public:
  AmbiguousCopy(const AmbiguousCopy&, int = 0);
  AmbiguousCopy(const AmbiguousCopy&, char = 0);
};

class PrivateCopy {
 private:
  PrivateCopy(const PrivateCopy&);
};

class ProtectedCopy {
 protected:
  ProtectedCopy(const ProtectedCopy&);
};

// --- copy_copyable.carbon

library "[[@TEST_NAME]]";

import Cpp library "types.h";

fn CopyCopyable(c: Cpp.Copyable) -> Cpp.Copyable {
  //@dump-sem-ir-begin
  return c;
  //@dump-sem-ir-end
}

fn CopyExplicitCopy(c: Cpp.ExplicitCopy) -> Cpp.ExplicitCopy {
  //@dump-sem-ir-begin
  return c;
  //@dump-sem-ir-end
}

// --- fail_copy_noncopyable.carbon

library "[[@TEST_NAME]]";

import Cpp library "types.h";

fn CopyDeletedCopy(c: Cpp.DeletedCopy) -> Cpp.DeletedCopy {
  // CHECK:STDERR: fail_copy_noncopyable.carbon:[[@LINE+8]]:10: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    15 |   return c;
  // CHECK:STDERR:       |          ^
  // CHECK:STDERR: fail_copy_noncopyable.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:14:3: note: 'DeletedCopy' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:    14 |   DeletedCopy(const DeletedCopy&) = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR:
  return c;
}

fn CopyMoveOnly(c: Cpp.MoveOnly) -> Cpp.MoveOnly {
  // CHECK:STDERR: fail_copy_noncopyable.carbon:[[@LINE+8]]:10: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    27 |   return c;
  // CHECK:STDERR:       |          ^
  // CHECK:STDERR: fail_copy_noncopyable.carbon:[[@LINE-18]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:19:3: note: copy constructor is implicitly deleted because 'MoveOnly' has a user-declared move constructor [CppInteropParseNote]
  // CHECK:STDERR:    19 |   MoveOnly(MoveOnly&&);
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR:
  return c;
}

fn CopyDeletedMove(c: Cpp.DeletedMove) -> Cpp.DeletedMove {
  // CHECK:STDERR: fail_copy_noncopyable.carbon:[[@LINE+8]]:10: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    39 |   return c;
  // CHECK:STDERR:       |          ^
  // CHECK:STDERR: fail_copy_noncopyable.carbon:[[@LINE-30]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:24:3: note: copy constructor is implicitly deleted because 'DeletedMove' has a user-declared move constructor [CppInteropParseNote]
  // CHECK:STDERR:    24 |   DeletedMove(DeletedMove&&) = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR:
  return c;
}

fn CopyNonConstCopy(c: Cpp.NonConstCopy) -> Cpp.NonConstCopy {
  // TODO: List candidates here.
  // CHECK:STDERR: fail_copy_noncopyable.carbon:[[@LINE+7]]:10: error: cannot copy value of type `Cpp.NonConstCopy` [CopyOfUncopyableType]
  // CHECK:STDERR:   return c;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_copy_noncopyable.carbon:[[@LINE+4]]:10: note: type `Cpp.NonConstCopy` does not implement interface `Core.Copy` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return c;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  return c;
}

fn CopyAmbiguousCopy(c: Cpp.AmbiguousCopy) -> Cpp.AmbiguousCopy {
  // TODO: List candidates here.
  // CHECK:STDERR: fail_copy_noncopyable.carbon:[[@LINE+7]]:10: error: cannot copy value of type `Cpp.AmbiguousCopy` [CopyOfUncopyableType]
  // CHECK:STDERR:   return c;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_copy_noncopyable.carbon:[[@LINE+4]]:10: note: type `Cpp.AmbiguousCopy` does not implement interface `Core.Copy` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return c;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  return c;
}

fn CopyPrivateCopy(c: Cpp.PrivateCopy) -> Cpp.PrivateCopy {
  // TODO: Note is missing location.
  // CHECK:STDERR: fail_copy_noncopyable.carbon:[[@LINE+5]]:10: error: cannot access private member `PrivateCopy` of type `Cpp.PrivateCopy` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   return c;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_copy_noncopyable.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  return c;
}

fn CopyProtectedCopy(c: Cpp.ProtectedCopy) -> Cpp.ProtectedCopy {
  // TODO: Note is missing location.
  // CHECK:STDERR: fail_copy_noncopyable.carbon:[[@LINE+5]]:10: error: cannot access protected member `ProtectedCopy` of type `Cpp.ProtectedCopy` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   return c;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_copy_noncopyable.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  return c;
}

// --- fail_todo_copy_generically.carbon

library "[[@TEST_NAME]]";

import Cpp library "types.h";

fn Copy[T:! Core.Copy](c: T) -> T {
  return c;
}

fn DoCopy(c: Cpp.Copyable) -> Cpp.Copyable {
  // TODO: We perform multiple impl lookups for `Cpp.Copyable as Core.Copy`
  // here, and each one produces a different impl witness, resulting in a
  // deduction failure.
  // CHECK:STDERR: fail_todo_copy_generically.carbon:[[@LINE+7]]:10: error: inconsistent deductions for value of generic parameter `T` [DeductionInconsistent]
  // CHECK:STDERR:   return Copy(c);
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR: fail_todo_copy_generically.carbon:[[@LINE-11]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Copy[T:! Core.Copy](c: T) -> T {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return Copy(c);
}

class Wrap(T:! Core.Copy) {}

// TODO: This should type-check: each conversion of `Cpp.Copyable` to
// `Core.Copy` should produce the same value.
fn EqualWitnesses(p: Wrap(Cpp.Copyable)*) -> Wrap(Cpp.Copyable)* {
  // CHECK:STDERR: fail_todo_copy_generically.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Wrap(Cpp.Copyable as Core.Copy)*` to `Wrap(Cpp.Copyable as Core.Copy)*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_copy_generically.carbon:[[@LINE+4]]:3: note: type `Wrap(Cpp.Copyable as Core.Copy)*` does not implement interface `Core.ImplicitAs(Wrap(Cpp.Copyable as Core.Copy)*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

// CHECK:STDOUT: --- copy_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Copyable: type = class_type @Copyable [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copyable.Copyable.type: type = fn_type @Copyable.Copyable [concrete]
// CHECK:STDOUT:   %Copyable.Copyable: %Copyable.Copyable.type = struct_value () [concrete]
// CHECK:STDOUT:   %const.622: type = const_type %Copyable [concrete]
// CHECK:STDOUT:   %ptr.29d: type = ptr_type %const.622 [concrete]
// CHECK:STDOUT:   %ptr.e47: type = ptr_type %Copyable [concrete]
// CHECK:STDOUT:   %Copyable__carbon_thunk.type: type = fn_type @Copyable__carbon_thunk [concrete]
// CHECK:STDOUT:   %Copyable__carbon_thunk: %Copyable__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %impl_witness.65e: <witness> = impl_witness @CopyCopyable.%impl_witness_table [concrete]
// CHECK:STDOUT:   %Copy.facet.26f: %Copy.type = facet_value %Copyable, (%impl_witness.65e) [concrete]
// CHECK:STDOUT:   %Copyable.Op.type: type = fn_type @Copyable.Op [concrete]
// CHECK:STDOUT:   %Copyable.Op: %Copyable.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %.667: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.26f [concrete]
// CHECK:STDOUT:   %ExplicitCopy: type = class_type @ExplicitCopy [concrete]
// CHECK:STDOUT:   %ExplicitCopy.ExplicitCopy.type: type = fn_type @ExplicitCopy.ExplicitCopy [concrete]
// CHECK:STDOUT:   %ExplicitCopy.ExplicitCopy: %ExplicitCopy.ExplicitCopy.type = struct_value () [concrete]
// CHECK:STDOUT:   %const.d8d: type = const_type %ExplicitCopy [concrete]
// CHECK:STDOUT:   %ptr.093: type = ptr_type %const.d8d [concrete]
// CHECK:STDOUT:   %ptr.84c: type = ptr_type %ExplicitCopy [concrete]
// CHECK:STDOUT:   %ExplicitCopy__carbon_thunk.type: type = fn_type @ExplicitCopy__carbon_thunk [concrete]
// CHECK:STDOUT:   %ExplicitCopy__carbon_thunk: %ExplicitCopy__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %impl_witness.215: <witness> = impl_witness @CopyExplicitCopy.%impl_witness_table [concrete]
// CHECK:STDOUT:   %Copy.facet.cfa: %Copy.type = facet_value %ExplicitCopy, (%impl_witness.215) [concrete]
// CHECK:STDOUT:   %ExplicitCopy.Op.type: type = fn_type @ExplicitCopy.Op [concrete]
// CHECK:STDOUT:   %ExplicitCopy.Op: %ExplicitCopy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %.a87: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.cfa [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Copyable.Copyable.decl: %Copyable.Copyable.type = fn_decl @Copyable.Copyable [concrete = constants.%Copyable.Copyable] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Copyable__carbon_thunk.decl: %Copyable__carbon_thunk.type = fn_decl @Copyable__carbon_thunk [concrete = constants.%Copyable__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ExplicitCopy.ExplicitCopy.decl: %ExplicitCopy.ExplicitCopy.type = fn_decl @ExplicitCopy.ExplicitCopy [concrete = constants.%ExplicitCopy.ExplicitCopy] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ExplicitCopy__carbon_thunk.decl: %ExplicitCopy__carbon_thunk.type = fn_decl @ExplicitCopy__carbon_thunk [concrete = constants.%ExplicitCopy__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyCopyable(%c.param: %Copyable) -> %return.param: %Copyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Copyable = name_ref c, %c
// CHECK:STDOUT:   %impl_witness_table = impl_witness_table (%Copyable.Op.decl), invalid [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness %impl_witness_table [concrete = constants.%impl_witness.65e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0: %.667 = impl_witness_access constants.%impl_witness.65e, element0 [concrete = constants.%Copyable.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %.loc8_10.1: ref %Copyable = temporary_storage
// CHECK:STDOUT:   %Op.ref: %Copyable.Copyable.type = name_ref Op, imports.%Copyable.Copyable.decl [concrete = constants.%Copyable.Copyable]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc8_10.2: ref %Copyable = value_as_ref %c.ref
// CHECK:STDOUT:   %addr.loc8_10.1: %ptr.e47 = addr_of %.loc8_10.2
// CHECK:STDOUT:   %.loc8_10.3: %ptr.29d = as_compatible %addr.loc8_10.1
// CHECK:STDOUT:   %.loc8_10.4: %ptr.29d = converted %addr.loc8_10.1, %.loc8_10.3
// CHECK:STDOUT:   %addr.loc8_10.2: %ptr.e47 = addr_of %.loc6_34
// CHECK:STDOUT:   %Copyable__carbon_thunk.call: init %empty_tuple.type = call imports.%Copyable__carbon_thunk.decl(%.loc8_10.4, %addr.loc8_10.2)
// CHECK:STDOUT:   %.loc8_10.5: init %Copyable = in_place_init %Copyable__carbon_thunk.call, %.loc6_34
// CHECK:STDOUT:   return %.loc8_10.5 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyExplicitCopy(%c.param: %ExplicitCopy) -> %return.param: %ExplicitCopy {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %ExplicitCopy = name_ref c, %c
// CHECK:STDOUT:   %impl_witness_table = impl_witness_table (%ExplicitCopy.Op.decl), invalid [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness %impl_witness_table [concrete = constants.%impl_witness.215]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0: %.a87 = impl_witness_access constants.%impl_witness.215, element0 [concrete = constants.%ExplicitCopy.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %.loc14_10.1: ref %ExplicitCopy = temporary_storage
// CHECK:STDOUT:   %Op.ref: %ExplicitCopy.ExplicitCopy.type = name_ref Op, imports.%ExplicitCopy.ExplicitCopy.decl [concrete = constants.%ExplicitCopy.ExplicitCopy]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc14_10.2: ref %ExplicitCopy = value_as_ref %c.ref
// CHECK:STDOUT:   %addr.loc14_10.1: %ptr.84c = addr_of %.loc14_10.2
// CHECK:STDOUT:   %.loc14_10.3: %ptr.093 = as_compatible %addr.loc14_10.1
// CHECK:STDOUT:   %.loc14_10.4: %ptr.093 = converted %addr.loc14_10.1, %.loc14_10.3
// CHECK:STDOUT:   %addr.loc14_10.2: %ptr.84c = addr_of %.loc12_42
// CHECK:STDOUT:   %ExplicitCopy__carbon_thunk.call: init %empty_tuple.type = call imports.%ExplicitCopy__carbon_thunk.decl(%.loc14_10.4, %addr.loc14_10.2)
// CHECK:STDOUT:   %.loc14_10.5: init %ExplicitCopy = in_place_init %ExplicitCopy__carbon_thunk.call, %.loc12_42
// CHECK:STDOUT:   return %.loc14_10.5 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
