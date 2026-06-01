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
  var unused a: Cpp.PublicDestructor = {};
}

fn TrivialDestroy() {
  var unused a: Cpp.TrivialDestructor = {};
}

// --- fail_todo_destroy_protected_base_destructor.carbon

library "[[@TEST_NAME]]";

import Cpp library "types.h";

class Derived {
  extend base: Cpp.ProtectedDestructor;
}

fn DestroyClassWithProtectedBaseDestructor() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_destroy_protected_base_destructor.carbon:[[@LINE+8]]:3: error: cannot access protected member `<C++ destructor>` of type `Cpp.ProtectedDestructor` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   var unused a: Derived = {.base = {}};
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_destroy_protected_base_destructor.carbon:[[@LINE-11]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:33:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   ~ProtectedDestructor();
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  var unused a: Derived = {.base = {}};
  //@dump-sem-ir-end
}

// --- fail_destroy_private_base_destructor.carbon

library "[[@TEST_NAME]]";

import Cpp library "types.h";

class Derived {
  extend base: Cpp.PrivateDestructor;
}

fn DestroyClassWithPrivateBaseDestructor() {
  //@dump-sem-ir-begin
  // TODO: We should not allow this, as the destructor of the base class is private.
  // CHECK:STDERR: fail_destroy_private_base_destructor.carbon:[[@LINE+8]]:3: error: cannot access private member `<C++ destructor>` of type `Cpp.PrivateDestructor` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   var unused a: Derived = {.base = {}};
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_destroy_private_base_destructor.carbon:[[@LINE-12]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:28:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   ~PrivateDestructor();
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  var unused a: Derived = {.base = {}};
  //@dump-sem-ir-end
}

// --- fail_destroy_nondestroyable.carbon

library "[[@TEST_NAME]]";

import Cpp library "types.h";

fn AmbiguousDestroy() {
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+20]]:17: error: binding pattern has incomplete type `AmbiguousDestructor` in name binding declaration [IncompleteTypeInBindingDecl]
  // CHECK:STDERR:   var unused a: Cpp.AmbiguousDestructor = {};
  // CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:13:7: note: destructor of class 'AmbiguousDestructorT<>' is ambiguous [CppInteropParseError]
  // CHECK:STDERR:    13 | class AmbiguousDestructorT {
  // CHECK:STDERR:       |       ^
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE-10]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:13:7: note: in instantiation of template class 'AmbiguousDestructorT<>' requested here [CppInteropParseNote]
  // CHECK:STDERR:    13 | class AmbiguousDestructorT {
  // CHECK:STDERR:       |       ^
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE-14]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:15:3: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR:    15 |   ~AmbiguousDestructorT() requires (N != 1);
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE-18]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:16:3: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR:    16 |   ~AmbiguousDestructorT() requires (N != 2);
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR:
  var unused a: Cpp.AmbiguousDestructor = {};
}

fn DeletedDestroy() {
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+8]]:3: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    39 |   var unused a: Cpp.DeletedDestructor = {};
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE-30]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:23:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:    23 |   ~DeletedDestructor() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR:
  var unused a: Cpp.DeletedDestructor = {};
}

fn PrivateDestroy() {
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+8]]:3: error: cannot access private member `<C++ destructor>` of type `Cpp.PrivateDestructor` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   var unused a: Cpp.PrivateDestructor = {};
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE-42]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:28:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   ~PrivateDestructor();
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  var unused a: Cpp.PrivateDestructor = {};
}

fn ProtectedDestroy() {
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+8]]:3: error: cannot access protected member `<C++ destructor>` of type `Cpp.ProtectedDestructor` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   var unused a: Cpp.ProtectedDestructor = {};
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE-54]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./types.h:33:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   ~ProtectedDestructor();
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  var unused a: Cpp.ProtectedDestructor = {};
}

// --- destroy_generically.carbon

library "[[@TEST_NAME]]";

import Cpp library "types.h";

fn Destroy[T:! Core.Copy & Core.Destroy](c: T) {
  var unused d: T = c;
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
// CHECK:STDOUT:   %pattern_type.c53: type = pattern_type %PublicDestructor [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %PublicDestructor.val: %PublicDestructor = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.12b: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.73c: %Destroy.WithSelf.Op.type.12b = struct_value () [symbolic]
// CHECK:STDOUT:   %PublicDestructor.cpp_destructor.type: type = fn_type @PublicDestructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %PublicDestructor.cpp_destructor: %PublicDestructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.cc4: <witness> = custom_witness (), @Destroy [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = fn_type @TrivialDestroy [concrete]
// CHECK:STDOUT:   %TrivialDestroy: %TrivialDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestructor: type = class_type @TrivialDestructor [concrete]
// CHECK:STDOUT:   %pattern_type.322: type = pattern_type %TrivialDestructor [concrete]
// CHECK:STDOUT:   %TrivialDestructor.val: %TrivialDestructor = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestructor.cpp_destructor.type: type = fn_type @TrivialDestructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %TrivialDestructor.cpp_destructor: %TrivialDestructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.b4f: %Destroy.type = facet_value %TrivialDestructor, (%custom_witness.cc4) [concrete]
// CHECK:STDOUT:   %TrivialDestructor.Op.type: type = fn_type @TrivialDestructor.Op [concrete]
// CHECK:STDOUT:   %TrivialDestructor.Op: %TrivialDestructor.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.12b = fn_decl @Destroy.WithSelf.Op [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.Op (constants.%Destroy.WithSelf.Op.73c)] {} {}
// CHECK:STDOUT:   %TrivialDestructor.decl: type = class_decl @TrivialDestructor [concrete = constants.%TrivialDestructor] {} {}
// CHECK:STDOUT:   %TrivialDestructor.cpp_destructor.decl: %TrivialDestructor.cpp_destructor.type = fn_decl @TrivialDestructor.cpp_destructor [concrete = constants.%TrivialDestructor.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.322 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.322 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %TrivialDestructor = ref_param call_param0
// CHECK:STDOUT:     %self: ref %TrivialDestructor = ref_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %a.patt: %pattern_type.c53 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.c53 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %PublicDestructor = var %a.var_patt
// CHECK:STDOUT:   %.loc11_41.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_41.2: init %PublicDestructor to %a.var = class_init () [concrete = constants.%PublicDestructor.val]
// CHECK:STDOUT:   %.loc11_3: init %PublicDestructor = converted %.loc11_41.1, %.loc11_41.2 [concrete = constants.%PublicDestructor.val]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_20: type = splice_block %PublicDestructor.ref [concrete = constants.%PublicDestructor] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %PublicDestructor.ref: type = name_ref PublicDestructor, imports.%PublicDestructor.decl [concrete = constants.%PublicDestructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %PublicDestructor = ref_binding a, %a.var
// CHECK:STDOUT:   %PublicDestructor.cpp_destructor.bound: <bound method> = bound_method %a.var, constants.%PublicDestructor.cpp_destructor
// CHECK:STDOUT:   %PublicDestructor.cpp_destructor.call: init %empty_tuple.type = call %PublicDestructor.cpp_destructor.bound(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PublicDestructor.cpp_destructor(%self.param: ref %PublicDestructor);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TrivialDestroy() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.322 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.322 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %TrivialDestructor = var %a.var_patt
// CHECK:STDOUT:   %.loc15_42.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc15_42.2: init %TrivialDestructor to %a.var = class_init () [concrete = constants.%TrivialDestructor.val]
// CHECK:STDOUT:   %.loc15_3: init %TrivialDestructor = converted %.loc15_42.1, %.loc15_42.2 [concrete = constants.%TrivialDestructor.val]
// CHECK:STDOUT:   assign %a.var, %.loc15_3
// CHECK:STDOUT:   %.loc15_20: type = splice_block %TrivialDestructor.ref [concrete = constants.%TrivialDestructor] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %TrivialDestructor.ref: type = name_ref TrivialDestructor, imports.%TrivialDestructor.decl [concrete = constants.%TrivialDestructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %TrivialDestructor = ref_binding a, %a.var
// CHECK:STDOUT:   %TrivialDestructor.Op.decl: %TrivialDestructor.Op.type = fn_decl @TrivialDestructor.Op [concrete = constants.%TrivialDestructor.Op] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.322 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.322 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %TrivialDestructor = ref_param call_param0
// CHECK:STDOUT:     %self: ref %TrivialDestructor = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TrivialDestructor.Op.bound: <bound method> = bound_method %a.var, constants.%TrivialDestructor.Op
// CHECK:STDOUT:   %Op.ref: %TrivialDestructor.cpp_destructor.type = name_ref Op, imports.%TrivialDestructor.cpp_destructor.decl [concrete = constants.%TrivialDestructor.cpp_destructor]
// CHECK:STDOUT:   %TrivialDestructor.cpp_destructor.bound: <bound method> = bound_method %a.var, %Op.ref
// CHECK:STDOUT:   %TrivialDestructor.cpp_destructor.call: init %empty_tuple.type = call %TrivialDestructor.cpp_destructor.bound(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TrivialDestructor.cpp_destructor(%self.param: ref %TrivialDestructor) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TrivialDestructor.Op(%self.param: ref %TrivialDestructor) [thunk imports.%TrivialDestructor.cpp_destructor.decl for imports.%Destroy.WithSelf.Op.decl, @Destroy.WithSelf.Op(constants.%Destroy.facet.b4f)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Op.ref: %TrivialDestructor.cpp_destructor.type = name_ref Op, imports.%TrivialDestructor.cpp_destructor.decl [concrete = constants.%TrivialDestructor.cpp_destructor]
// CHECK:STDOUT:   %TrivialDestructor.cpp_destructor.bound: <bound method> = bound_method %self.param, %Op.ref
// CHECK:STDOUT:   %TrivialDestructor.cpp_destructor.call: init %empty_tuple.type = call %TrivialDestructor.cpp_destructor.bound(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_destroy_protected_base_destructor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %ProtectedDestructor: type = class_type @ProtectedDestructor [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.base.233: type = struct_type {.base: %ProtectedDestructor} [concrete]
// CHECK:STDOUT:   %pattern_type.e25: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %empty_struct.a40: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.f5e: type = struct_type {.base: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.base.f5e = struct_value (%empty_struct.a40) [concrete]
// CHECK:STDOUT:   %.1d0: type = partial_type %ProtectedDestructor [concrete]
// CHECK:STDOUT:   %empty_struct.724: %.1d0 = struct_value () [concrete]
// CHECK:STDOUT:   %ProtectedDestructor.val: %ProtectedDestructor = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.val: %Derived = struct_value (%ProtectedDestructor.val) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.2: type = fn_type @Destroy.Op.loc20_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.2: %Destroy.Op.type.af7ec0.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyClassWithProtectedBaseDestructor() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.e25 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.e25 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Derived = var %a.var_patt
// CHECK:STDOUT:   %.loc20_37.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct.a40]
// CHECK:STDOUT:   %.loc20_38.1: %struct_type.base.f5e = struct_literal (%.loc20_37.1) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc20_38.2: ref %.1d0 = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc20_37.2: init %.1d0 to %.loc20_38.2 = class_init () [concrete = constants.%empty_struct.724]
// CHECK:STDOUT:   %.loc20_38.3: init %.1d0 = converted %.loc20_37.1, %.loc20_37.2 [concrete = constants.%empty_struct.724]
// CHECK:STDOUT:   %.loc20_38.4: init %ProtectedDestructor = as_compatible %.loc20_38.3 [concrete = constants.%ProtectedDestructor.val]
// CHECK:STDOUT:   %.loc20_38.5: init %Derived to %a.var = class_init (%.loc20_38.4) [concrete = constants.%Derived.val]
// CHECK:STDOUT:   %.loc20_3: init %Derived = converted %.loc20_38.1, %.loc20_38.5 [concrete = constants.%Derived.val]
// CHECK:STDOUT:   assign %a.var, %.loc20_3
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %a: ref %Derived = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1dc86d.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc20_3.1(%self.param: ref %struct_type.base.233) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc20_3.2(%self.param: ref %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_destroy_private_base_destructor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %PrivateDestructor: type = class_type @PrivateDestructor [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.base.dea: type = struct_type {.base: %PrivateDestructor} [concrete]
// CHECK:STDOUT:   %pattern_type.377: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %empty_struct.a40: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.f5e: type = struct_type {.base: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.base.f5e = struct_value (%empty_struct.a40) [concrete]
// CHECK:STDOUT:   %.ddb: type = partial_type %PrivateDestructor [concrete]
// CHECK:STDOUT:   %empty_struct.0b0: %.ddb = struct_value () [concrete]
// CHECK:STDOUT:   %PrivateDestructor.val: %PrivateDestructor = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.val: %Derived = struct_value (%PrivateDestructor.val) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.2: type = fn_type @Destroy.Op.loc21_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.2: %Destroy.Op.type.af7ec0.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyClassWithPrivateBaseDestructor() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.377 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.377 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Derived = var %a.var_patt
// CHECK:STDOUT:   %.loc21_37.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct.a40]
// CHECK:STDOUT:   %.loc21_38.1: %struct_type.base.f5e = struct_literal (%.loc21_37.1) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc21_38.2: ref %.ddb = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc21_37.2: init %.ddb to %.loc21_38.2 = class_init () [concrete = constants.%empty_struct.0b0]
// CHECK:STDOUT:   %.loc21_38.3: init %.ddb = converted %.loc21_37.1, %.loc21_37.2 [concrete = constants.%empty_struct.0b0]
// CHECK:STDOUT:   %.loc21_38.4: init %PrivateDestructor = as_compatible %.loc21_38.3 [concrete = constants.%PrivateDestructor.val]
// CHECK:STDOUT:   %.loc21_38.5: init %Derived to %a.var = class_init (%.loc21_38.4) [concrete = constants.%Derived.val]
// CHECK:STDOUT:   %.loc21_3: init %Derived = converted %.loc21_38.1, %.loc21_38.5 [concrete = constants.%Derived.val]
// CHECK:STDOUT:   assign %a.var, %.loc21_3
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %a: ref %Derived = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1dc86d.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc21_3.1(%self.param: ref %struct_type.base.dea) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc21_3.2(%self.param: ref %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
