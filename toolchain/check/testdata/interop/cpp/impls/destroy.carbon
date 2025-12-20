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

template<int N = 0>
class AmbiguousDestructorT {
 public:
  ~AmbiguousDestructorT() requires (N != 1);
  ~AmbiguousDestructorT() requires (N != 2);
};

using AmbiguousDestructor = AmbiguousDestructorT<>;

class DeletedDestructor {
 public:
  ~DeletedDestructor() = delete;
};

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

// Dump the whole file so we can see that the trivial destructor maps to a
// "no_op" builtin.
//@include-in-dumps

fn PublicDestroy() {
  var a: Cpp.PublicDestructor = {};
}

fn TrivialDestroy() {
  var a: Cpp.TrivialDestructor = {};
}

// --- destroy_protected_base_destructor.carbon

library "[[@TEST_NAME]]";

import Cpp library "types.h";

class Derived {
  extend base: Cpp.ProtectedDestructor;
}

fn DestroyClassWithProtectedBaseDestructor() {
  //@dump-sem-ir-begin
  var a: Derived = {.base = {}};
  //@dump-sem-ir-end
}

// --- todo_fail_destroy_private_base_destructor.carbon

library "[[@TEST_NAME]]";

import Cpp library "types.h";

class Derived {
  extend base: Cpp.PrivateDestructor;
}

fn DestroyClassWithPrivateBaseDestructor() {
  //@dump-sem-ir-begin
  // TODO: We should not allow this, as the destructor of the base class is private.
  var a: Derived = {.base = {}};
  //@dump-sem-ir-end
}

// --- fail_destroy_nondestroyable.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+16]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./types.h:13:7: error: destructor of class 'AmbiguousDestructorT<>' is ambiguous [CppInteropParseError]
// CHECK:STDERR:    13 | class AmbiguousDestructorT {
// CHECK:STDERR:       |       ^
// CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+12]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./types.h:13:7: note: in instantiation of template class 'AmbiguousDestructorT<>' requested here [CppInteropParseNote]
// CHECK:STDERR:    13 | class AmbiguousDestructorT {
// CHECK:STDERR:       |       ^
// CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./types.h:15:3: note: candidate function [CppInteropParseNote]
// CHECK:STDERR:    15 |   ~AmbiguousDestructorT() requires (N != 1);
// CHECK:STDERR:       |   ^
// CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./types.h:16:3: note: candidate function [CppInteropParseNote]
// CHECK:STDERR:    16 |   ~AmbiguousDestructorT() requires (N != 2);
// CHECK:STDERR:       |   ^
import Cpp library "types.h";

fn AmbiguousDestroy() {
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+4]]:10: note: while completing C++ type `Cpp.AmbiguousDestructorT` [InCppTypeCompletion]
  // CHECK:STDERR:   var a: Cpp.AmbiguousDestructor = {};
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var a: Cpp.AmbiguousDestructor = {};
}

fn DeletedDestroy() {
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+8]]:3: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    39 |   var a: Cpp.DeletedDestructor = {};
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE-14]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:23:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:    23 |   ~DeletedDestructor() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR:
  var a: Cpp.DeletedDestructor = {};
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

// --- destroy_generically.carbon

library "[[@TEST_NAME]]";

import Cpp library "types.h";

fn Destroy[T:! Core.Copy & Core.Destroy](c: T) {
  var d: T = c;
}

fn DoDestroy() {
  Destroy({} as Cpp.PublicDestructor);
}

class Wrap(T:! Core.Destroy) {}

fn EqualWitnesses(p: Wrap(Cpp.PublicDestructor)*) -> Wrap(Cpp.PublicDestructor)* {
  return p;
}

// CHECK:STDOUT: --- destroy_destroyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %PublicDestroy.type: type = fn_type @PublicDestroy [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %PublicDestroy: %PublicDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %PublicDestructor: type = class_type @PublicDestructor [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.27d: type = pattern_type %PublicDestructor [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %PublicDestructor.val: %PublicDestructor = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %PublicDestructor.cpp_destructor.type: type = fn_type @PublicDestructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %PublicDestructor.cpp_destructor: %PublicDestructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = fn_type @TrivialDestroy [concrete]
// CHECK:STDOUT:   %TrivialDestroy: %TrivialDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestructor: type = class_type @TrivialDestructor [concrete]
// CHECK:STDOUT:   %pattern_type.1b8: type = pattern_type %TrivialDestructor [concrete]
// CHECK:STDOUT:   %TrivialDestructor.val: %TrivialDestructor = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestructor.cpp_destructor.type: type = fn_type @TrivialDestructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %TrivialDestructor.cpp_destructor: %TrivialDestructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .PublicDestructor = %PublicDestructor.decl
// CHECK:STDOUT:     .TrivialDestructor = %TrivialDestructor.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PublicDestructor.decl: type = class_decl @PublicDestructor [concrete = constants.%PublicDestructor] {} {}
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %TrivialDestructor.decl: type = class_decl @TrivialDestructor [concrete = constants.%TrivialDestructor] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .PublicDestroy = %PublicDestroy.decl
// CHECK:STDOUT:     .TrivialDestroy = %TrivialDestroy.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "types.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PublicDestroy.decl: %PublicDestroy.type = fn_decl @PublicDestroy [concrete = constants.%PublicDestroy] {} {}
// CHECK:STDOUT:   %TrivialDestroy.decl: %TrivialDestroy.type = fn_decl @TrivialDestroy [concrete = constants.%TrivialDestroy] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @PublicDestructor {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @TrivialDestructor {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PublicDestroy() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.27d = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.27d = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %PublicDestructor = var %a.var_patt
// CHECK:STDOUT:   %.loc11_34.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_34.2: init %PublicDestructor = class_init (), %a.var [concrete = constants.%PublicDestructor.val]
// CHECK:STDOUT:   %.loc11_3: init %PublicDestructor = converted %.loc11_34.1, %.loc11_34.2 [concrete = constants.%PublicDestructor.val]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_13: type = splice_block %PublicDestructor.ref [concrete = constants.%PublicDestructor] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %PublicDestructor.ref: type = name_ref PublicDestructor, imports.%PublicDestructor.decl [concrete = constants.%PublicDestructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %PublicDestructor = ref_binding a, %a.var
// CHECK:STDOUT:   %PublicDestructor.cpp_destructor.bound: <bound method> = bound_method %a.var, constants.%PublicDestructor.cpp_destructor
// CHECK:STDOUT:   %PublicDestructor.cpp_destructor.call: init %empty_tuple.type = call %PublicDestructor.cpp_destructor.bound(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc11(%self.param: %PublicDestructor) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PublicDestructor.cpp_destructor(%self.param: %PublicDestructor);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TrivialDestroy() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.1b8 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.1b8 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %TrivialDestructor = var %a.var_patt
// CHECK:STDOUT:   %.loc15_35.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc15_35.2: init %TrivialDestructor = class_init (), %a.var [concrete = constants.%TrivialDestructor.val]
// CHECK:STDOUT:   %.loc15_3: init %TrivialDestructor = converted %.loc15_35.1, %.loc15_35.2 [concrete = constants.%TrivialDestructor.val]
// CHECK:STDOUT:   assign %a.var, %.loc15_3
// CHECK:STDOUT:   %.loc15_13: type = splice_block %TrivialDestructor.ref [concrete = constants.%TrivialDestructor] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %TrivialDestructor.ref: type = name_ref TrivialDestructor, imports.%TrivialDestructor.decl [concrete = constants.%TrivialDestructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %TrivialDestructor = ref_binding a, %a.var
// CHECK:STDOUT:   %TrivialDestructor.cpp_destructor.bound: <bound method> = bound_method %a.var, constants.%TrivialDestructor.cpp_destructor
// CHECK:STDOUT:   %TrivialDestructor.cpp_destructor.call: init %empty_tuple.type = call %TrivialDestructor.cpp_destructor.bound(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc15(%self.param: %TrivialDestructor) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TrivialDestructor.cpp_destructor(%self.param: %TrivialDestructor) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- destroy_protected_base_destructor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ProtectedDestructor: type = class_type @ProtectedDestructor [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.9f6: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.f5e: type = struct_type {.base: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.base.f5e = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %ProtectedDestructor.val: %ProtectedDestructor = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.val: %Derived = struct_value (%ProtectedDestructor.val) [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyClassWithProtectedBaseDestructor() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.9f6 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.9f6 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Derived = var %a.var_patt
// CHECK:STDOUT:   %.loc12_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc12_31.1: %struct_type.base.f5e = struct_literal (%.loc12_30.1) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc12_31.2: ref %ProtectedDestructor = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc12_30.2: init %ProtectedDestructor = class_init (), %.loc12_31.2 [concrete = constants.%ProtectedDestructor.val]
// CHECK:STDOUT:   %.loc12_31.3: init %ProtectedDestructor = converted %.loc12_30.1, %.loc12_30.2 [concrete = constants.%ProtectedDestructor.val]
// CHECK:STDOUT:   %.loc12_31.4: init %Derived = class_init (%.loc12_31.3), %a.var [concrete = constants.%Derived.val]
// CHECK:STDOUT:   %.loc12_3: init %Derived = converted %.loc12_31.1, %.loc12_31.4 [concrete = constants.%Derived.val]
// CHECK:STDOUT:   assign %a.var, %.loc12_3
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %a: ref %Derived = ref_binding a, %a.var
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %a.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %Derived) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_destroy_private_base_destructor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %PrivateDestructor: type = class_type @PrivateDestructor [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.9f6: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.f5e: type = struct_type {.base: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.base.f5e = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %PrivateDestructor.val: %PrivateDestructor = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.val: %Derived = struct_value (%PrivateDestructor.val) [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyClassWithPrivateBaseDestructor() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.9f6 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.9f6 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Derived = var %a.var_patt
// CHECK:STDOUT:   %.loc13_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc13_31.1: %struct_type.base.f5e = struct_literal (%.loc13_30.1) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc13_31.2: ref %PrivateDestructor = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc13_30.2: init %PrivateDestructor = class_init (), %.loc13_31.2 [concrete = constants.%PrivateDestructor.val]
// CHECK:STDOUT:   %.loc13_31.3: init %PrivateDestructor = converted %.loc13_30.1, %.loc13_30.2 [concrete = constants.%PrivateDestructor.val]
// CHECK:STDOUT:   %.loc13_31.4: init %Derived = class_init (%.loc13_31.3), %a.var [concrete = constants.%Derived.val]
// CHECK:STDOUT:   %.loc13_3: init %Derived = converted %.loc13_31.1, %.loc13_31.4 [concrete = constants.%Derived.val]
// CHECK:STDOUT:   assign %a.var, %.loc13_3
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %a: ref %Derived = ref_binding a, %a.var
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %a.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %Derived) = "no_op";
// CHECK:STDOUT:
