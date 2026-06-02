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
  // CHECK:STDERR: fail_copy_noncopyable.carbon:[[@LINE+4]]:10: note: type `Cpp.NonConstCopy` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
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
  // CHECK:STDERR: fail_copy_noncopyable.carbon:[[@LINE+4]]:10: note: type `Cpp.AmbiguousCopy` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return c;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  return c;
}

fn CopyPrivateCopy(c: Cpp.PrivateCopy) -> Cpp.PrivateCopy {
  // TODO: Note is missing location.
  // CHECK:STDERR: fail_copy_noncopyable.carbon:[[@LINE+8]]:10: error: cannot access private member `PrivateCopy` of type `Cpp.PrivateCopy` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   return c;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_copy_noncopyable.carbon:[[@LINE-67]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:40:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   PrivateCopy(const PrivateCopy&);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  return c;
}

fn CopyProtectedCopy(c: Cpp.ProtectedCopy) -> Cpp.ProtectedCopy {
  // TODO: Note is missing location.
  // CHECK:STDERR: fail_copy_noncopyable.carbon:[[@LINE+8]]:10: error: cannot access protected member `ProtectedCopy` of type `Cpp.ProtectedCopy` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   return c;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_copy_noncopyable.carbon:[[@LINE-80]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:45:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   ProtectedCopy(const ProtectedCopy&);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  return c;
}

// --- copy_generically.carbon

library "[[@TEST_NAME]]";

import Cpp library "types.h";

fn Copy[T:! Core.Copy](c: T) -> T {
  return c;
}

fn DoCopy(c: Cpp.Copyable) -> Cpp.Copyable {
  //@dump-sem-ir-begin
  return Copy(c);
  //@dump-sem-ir-end
}

class Wrap(T:! Core.Copy) {}

fn EqualWitnesses(p: Wrap(Cpp.Copyable)*) -> Wrap(Cpp.Copyable)* {
  //@dump-sem-ir-begin
  return p;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- copy_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Copyable: type = class_type @Copyable [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copyable.Copyable.type: type = fn_type @Copyable.Copyable [concrete]
// CHECK:STDOUT:   %Copyable.Copyable: %Copyable.Copyable.type = struct_value () [concrete]
// CHECK:STDOUT:   %const.151: type = const_type %Copyable [concrete]
// CHECK:STDOUT:   %ptr.765: type = ptr_type %const.151 [concrete]
// CHECK:STDOUT:   %ptr.fc9: type = ptr_type %Copyable [concrete]
// CHECK:STDOUT:   %Copyable__carbon_thunk.type: type = fn_type @Copyable__carbon_thunk [concrete]
// CHECK:STDOUT:   %Copyable__carbon_thunk: %Copyable__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copyable.Op.type: type = fn_type @Copyable.Op [concrete]
// CHECK:STDOUT:   %Copyable.Op: %Copyable.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b8df63.1: <witness> = custom_witness (%Copyable.Op), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.0af: %Copy.type = facet_value %Copyable, (%custom_witness.b8df63.1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.b36: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.0af) [concrete]
// CHECK:STDOUT:   %.8c1: type = fn_type_with_self_type %Copy.WithSelf.Op.type.b36, %Copy.facet.0af [concrete]
// CHECK:STDOUT:   %ExplicitCopy: type = class_type @ExplicitCopy [concrete]
// CHECK:STDOUT:   %ExplicitCopy.ExplicitCopy.type: type = fn_type @ExplicitCopy.ExplicitCopy [concrete]
// CHECK:STDOUT:   %ExplicitCopy.ExplicitCopy: %ExplicitCopy.ExplicitCopy.type = struct_value () [concrete]
// CHECK:STDOUT:   %const.869: type = const_type %ExplicitCopy [concrete]
// CHECK:STDOUT:   %ptr.39a: type = ptr_type %const.869 [concrete]
// CHECK:STDOUT:   %ptr.82f: type = ptr_type %ExplicitCopy [concrete]
// CHECK:STDOUT:   %ExplicitCopy__carbon_thunk.type: type = fn_type @ExplicitCopy__carbon_thunk [concrete]
// CHECK:STDOUT:   %ExplicitCopy__carbon_thunk: %ExplicitCopy__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitCopy.Op.type: type = fn_type @ExplicitCopy.Op [concrete]
// CHECK:STDOUT:   %ExplicitCopy.Op: %ExplicitCopy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b8df63.2: <witness> = custom_witness (%ExplicitCopy.Op), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.b31: %Copy.type = facet_value %ExplicitCopy, (%custom_witness.b8df63.2) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.baa: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.b31) [concrete]
// CHECK:STDOUT:   %.fcd: type = fn_type_with_self_type %Copy.WithSelf.Op.type.baa, %Copy.facet.b31 [concrete]
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
// CHECK:STDOUT: fn @CopyCopyable(%c.param: %Copyable) -> out %return.param: %Copyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Copyable = name_ref c, %c
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0: %.8c1 = impl_witness_access constants.%custom_witness.b8df63.1, element0 [concrete = constants.%Copyable.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %.loc8_10.1: ref %Copyable = temporary_storage
// CHECK:STDOUT:   %Op.ref: %Copyable.Copyable.type = name_ref Op, imports.%Copyable.Copyable.decl [concrete = constants.%Copyable.Copyable]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc8_10.2: ref %Copyable = value_as_ref %c.ref
// CHECK:STDOUT:   %addr.loc8_10.1: %ptr.fc9 = addr_of %.loc8_10.2
// CHECK:STDOUT:   %.loc8_10.3: %ptr.765 = as_compatible %addr.loc8_10.1
// CHECK:STDOUT:   %.loc8_10.4: %ptr.765 = converted %addr.loc8_10.1, %.loc8_10.3
// CHECK:STDOUT:   %addr.loc8_10.2: %ptr.fc9 = addr_of %.loc6_40.1
// CHECK:STDOUT:   %Copyable__carbon_thunk.call: init %empty_tuple.type = call imports.%Copyable__carbon_thunk.decl(%.loc8_10.4, %addr.loc8_10.2)
// CHECK:STDOUT:   %.loc8_10.5: init %Copyable to %.loc6_40.1 = mark_in_place_init %Copyable__carbon_thunk.call
// CHECK:STDOUT:   return %.loc8_10.5 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CopyExplicitCopy(%c.param: %ExplicitCopy) -> out %return.param: %ExplicitCopy {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %ExplicitCopy = name_ref c, %c
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0: %.fcd = impl_witness_access constants.%custom_witness.b8df63.2, element0 [concrete = constants.%ExplicitCopy.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %.loc14_10.1: ref %ExplicitCopy = temporary_storage
// CHECK:STDOUT:   %Op.ref: %ExplicitCopy.ExplicitCopy.type = name_ref Op, imports.%ExplicitCopy.ExplicitCopy.decl [concrete = constants.%ExplicitCopy.ExplicitCopy]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc14_10.2: ref %ExplicitCopy = value_as_ref %c.ref
// CHECK:STDOUT:   %addr.loc14_10.1: %ptr.82f = addr_of %.loc14_10.2
// CHECK:STDOUT:   %.loc14_10.3: %ptr.39a = as_compatible %addr.loc14_10.1
// CHECK:STDOUT:   %.loc14_10.4: %ptr.39a = converted %addr.loc14_10.1, %.loc14_10.3
// CHECK:STDOUT:   %addr.loc14_10.2: %ptr.82f = addr_of %.loc12_48.1
// CHECK:STDOUT:   %ExplicitCopy__carbon_thunk.call: init %empty_tuple.type = call imports.%ExplicitCopy__carbon_thunk.decl(%.loc14_10.4, %addr.loc14_10.2)
// CHECK:STDOUT:   %.loc14_10.5: init %ExplicitCopy to %.loc12_48.1 = mark_in_place_init %ExplicitCopy__carbon_thunk.call
// CHECK:STDOUT:   return %.loc14_10.5 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- copy_generically.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Copy.type.62b: type = facet_type <@Copy.1> [concrete]
// CHECK:STDOUT:   %Copy.type.543: type = fn_type @Copy.loc6 [concrete]
// CHECK:STDOUT:   %Copy: %Copy.type.543 = struct_value () [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.b03: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.aed: %ptr.as.Copy.impl.Op.type.b03 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copyable: type = class_type @Copyable [concrete]
// CHECK:STDOUT:   %Copyable.Op.type: type = fn_type @Copyable.Op [concrete]
// CHECK:STDOUT:   %Copyable.Op: %Copyable.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b8d: <witness> = custom_witness (%Copyable.Op), @Copy.1 [concrete]
// CHECK:STDOUT:   %Copy.facet.0af: %Copy.type.62b = facet_value %Copyable, (%custom_witness.b8d) [concrete]
// CHECK:STDOUT:   %Copy.specific_fn: <specific function> = specific_function %Copy, @Copy.loc6(%Copy.facet.0af) [concrete]
// CHECK:STDOUT:   %Wrap.cb4: type = class_type @Wrap, @Wrap(%Copy.facet.0af) [concrete]
// CHECK:STDOUT:   %ptr.727: type = ptr_type %Wrap.cb4 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.b80: <witness> = impl_witness imports.%Copy.impl_witness_table.760, @ptr.as.Copy.impl(%Wrap.cb4) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.b3b: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Wrap.cb4) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.b43: %ptr.as.Copy.impl.Op.type.b3b = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.86d: %Copy.type.62b = facet_value %ptr.727, (%Copy.impl_witness.b80) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.111: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.86d) [concrete]
// CHECK:STDOUT:   %.378: type = fn_type_with_self_type %Copy.WithSelf.Op.type.111, %Copy.facet.86d [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.b43, @ptr.as.Copy.impl.Op(%Wrap.cb4) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.13c: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.b03) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.aed)]
// CHECK:STDOUT:   %Copy.impl_witness_table.760 = impl_witness_table (%Core.import_ref.13c), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DoCopy(%c.param: %Copyable) -> out %return.param: %Copyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Copy.ref: %Copy.type.543 = name_ref Copy, file.%Copy.decl [concrete = constants.%Copy]
// CHECK:STDOUT:   %c.ref: %Copyable = name_ref c, %c
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Copy.facet: %Copy.type.62b = facet_value constants.%Copyable, (constants.%custom_witness.b8d) [concrete = constants.%Copy.facet.0af]
// CHECK:STDOUT:   %.loc12: %Copy.type.62b = converted constants.%Copyable, %Copy.facet [concrete = constants.%Copy.facet.0af]
// CHECK:STDOUT:   %Copy.specific_fn: <specific function> = specific_function %Copy.ref, @Copy.loc6(constants.%Copy.facet.0af) [concrete = constants.%Copy.specific_fn]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Copy.call: init %Copyable to %.loc10_34.1 = call %Copy.specific_fn(%c.ref)
// CHECK:STDOUT:   return %Copy.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @EqualWitnesses(%p.param: %ptr.727) -> out %return.param: %ptr.727 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.727 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.378 = impl_witness_access constants.%Copy.impl_witness.b80, element0 [concrete = constants.%ptr.as.Copy.impl.Op.b43]
// CHECK:STDOUT:   %bound_method.loc20_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%Wrap.cb4) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.727 = call %bound_method.loc20_10.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
