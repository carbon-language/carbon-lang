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
  // CHECK:STDERR: fail_todo_destroy_protected_base_destructor.carbon:[[@LINE+5]]:3-23: error: cannot access protected member `<C++ destructor>` of type `Cpp.ProtectedDestructor` [ClassInvalidMemberAccess]
  // CHECK:STDERR: included from: fail_todo_destroy_protected_base_destructor.carbon:[[@LINE-9]]:10 [InCppInclude]
  // CHECK:STDERR: ./types.h:33:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
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
  // CHECK:STDERR: fail_destroy_private_base_destructor.carbon:[[@LINE+5]]:3-23: error: cannot access private member `<C++ destructor>` of type `Cpp.PrivateDestructor` [ClassInvalidMemberAccess]
  // CHECK:STDERR: included from: fail_destroy_private_base_destructor.carbon:[[@LINE-10]]:10 [InCppInclude]
  // CHECK:STDERR: ./types.h:28:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  var unused a: Derived = {.base = {}};
  //@dump-sem-ir-end
}

// --- fail_destroy_nondestroyable.carbon

library "[[@TEST_NAME]]";

import Cpp library "types.h";

fn AmbiguousDestroy() {
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+7]]:17-39: error: binding pattern has incomplete type `AmbiguousDestructor` in name binding declaration [IncompleteTypeInBindingDecl]
  // CHECK:STDERR: included from: fail_destroy_nondestroyable.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./types.h:13:7: note: destructor of class 'AmbiguousDestructorT<>' is ambiguous [CppInteropParseError]
  // CHECK:STDERR: ./types.h:13:7: note: in instantiation of template class 'AmbiguousDestructorT<>' requested here [CppInteropParseNote]
  // CHECK:STDERR: ./types.h:15:3: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR: ./types.h:16:3: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR:
  var unused a: Cpp.AmbiguousDestructor = {};
}

fn DeletedDestroy() {
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+8]]:3: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_destroy_nondestroyable.carbon:[[@LINE-15]]:10 [InCppInclude]
  // CHECK:STDERR: ./types.h:23:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+3]]:3-37: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  var unused a: Cpp.DeletedDestructor = {};
}

fn PrivateDestroy() {
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+5]]:3-37: error: cannot access private member `<C++ destructor>` of type `Cpp.PrivateDestructor` [ClassInvalidMemberAccess]
  // CHECK:STDERR: included from: fail_destroy_nondestroyable.carbon:[[@LINE-27]]:10 [InCppInclude]
  // CHECK:STDERR: ./types.h:28:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  var unused a: Cpp.PrivateDestructor = {};
}

fn ProtectedDestroy() {
  // CHECK:STDERR: fail_destroy_nondestroyable.carbon:[[@LINE+5]]:3-39: error: cannot access protected member `<C++ destructor>` of type `Cpp.ProtectedDestructor` [ClassInvalidMemberAccess]
  // CHECK:STDERR: included from: fail_destroy_nondestroyable.carbon:[[@LINE-36]]:10 [InCppInclude]
  // CHECK:STDERR: ./types.h:33:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  var unused a: Cpp.ProtectedDestructor = {};
}

// --- destroy_generically.carbon

library "[[@TEST_NAME]]";

import Cpp library "types.h";

fn Destroy[T: Core.Copy & Core.Destroy](c: T) {
  var unused d: T = c;
}

fn DoDestroy() {
  Destroy({} as Cpp.PublicDestructor);
}

class Wrap(T: Core.Destroy) {}

fn EqualWitnesses(p: Wrap(Cpp.PublicDestructor)*) -> Wrap(Cpp.PublicDestructor)* {
  return p;
}

// --- fail_no_destroy.carbon

library "[[@TEST_NAME]]";

import Cpp library "types.h";

class AdaptNoDestroy {
  adapt Cpp.DeletedDestructor;
}

fn Make() -> AdaptNoDestroy;

fn F() {
  // CHECK:STDERR: fail_no_destroy.carbon:[[@LINE+8]]:3: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: included from: fail_no_destroy.carbon:[[@LINE-10]]:10 [InCppInclude]
  // CHECK:STDERR: ./types.h:23:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_no_destroy.carbon:[[@LINE+3]]:3-23: error: cannot convert type `AdaptNoDestroy` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: min_prelude/parts/destroy.carbon:32:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  var _: AdaptNoDestroy = Make();
}

// CHECK:STDOUT: --- destroy_destroyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %PublicDestroy.type: type = fn_type @PublicDestroy [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %PublicDestroy: %PublicDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %PublicDestructor: type = class_type @PublicDestructor [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.bdc: type = pattern_type %PublicDestructor [concrete]
// CHECK:STDOUT:   %a.patt.e26: %pattern_type.bdc = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt.17b: %pattern_type.bdc = var_pattern %a.patt.e26 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %PublicDestructor.val: %PublicDestructor = struct_value () [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %PublicDestructor.cpp_destructor.type: type = fn_type @PublicDestructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %PublicDestructor.cpp_destructor: %PublicDestructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fe1: <witness> = custom_witness (%PublicDestructor.cpp_destructor), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a76: %SubobjectDestroy.type = facet_value %PublicDestructor, (%custom_witness.fe1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.981: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a76) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f4c: %Destroy.type = facet_value %PublicDestructor, (%DestroyImpl.impl_witness.981, %custom_witness.fe1) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.8f9: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.f4c) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = fn_type @TrivialDestroy [concrete]
// CHECK:STDOUT:   %TrivialDestroy: %TrivialDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestructor: type = class_type @TrivialDestructor [concrete]
// CHECK:STDOUT:   %pattern_type.a12: type = pattern_type %TrivialDestructor [concrete]
// CHECK:STDOUT:   %a.patt.b64: %pattern_type.a12 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt.d25: %pattern_type.a12 = var_pattern %a.patt.b64 [concrete]
// CHECK:STDOUT:   %TrivialDestructor.val: %TrivialDestructor = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.fda: %SubobjectDestroy.type = facet_value %TrivialDestructor, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.bf2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.fda) [concrete]
// CHECK:STDOUT:   %Destroy.facet.5b3: %Destroy.type = facet_value %TrivialDestructor, (%DestroyImpl.impl_witness.bf2, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.753: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.5b3) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .PublicDestructor = %PublicDestructor.decl
// CHECK:STDOUT:     .TrivialDestructor = %TrivialDestructor.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PublicDestructor.decl: type = class_decl @PublicDestructor [concrete = constants.%PublicDestructor] {} {}
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
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
// CHECK:STDOUT:   %a.var: ref %PublicDestructor = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc11_41.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_41.2: init %PublicDestructor to %a.var = class_init () [concrete = constants.%PublicDestructor.val]
// CHECK:STDOUT:   %.loc11_3.1: init %PublicDestructor = converted %.loc11_41.1, %.loc11_41.2 [concrete = constants.%PublicDestructor.val]
// CHECK:STDOUT:   assign %a.var, %.loc11_3.1
// CHECK:STDOUT:   %.loc11_20: type = splice_block %PublicDestructor.ref [concrete = constants.%PublicDestructor] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %PublicDestructor.ref: type = name_ref PublicDestructor, imports.%PublicDestructor.decl [concrete = constants.%PublicDestructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %PublicDestructor = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.bdc = ref_binding_pattern a [concrete = constants.%a.patt.e26]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.bdc = var_pattern %a.patt [concrete = constants.%a.var_patt.17b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc11_3.1: %Destroy.type = facet_value constants.%PublicDestructor, (constants.%DestroyImpl.impl_witness.981, constants.%custom_witness.fe1) [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %.loc11_3.2: %Destroy.type = converted constants.%PublicDestructor, %Destroy.facet.loc11_3.1 [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %Destroy.facet.loc11_3.2: %Destroy.type = facet_value constants.%PublicDestructor, (constants.%DestroyImpl.impl_witness.981, constants.%custom_witness.fe1) [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %.loc11_3.3: %Destroy.type = converted constants.%PublicDestructor, %Destroy.facet.loc11_3.2 [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.f4c) [concrete = constants.%SelfDestruct.specific_fn.8f9]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PublicDestructor.cpp_destructor(%self.param: ref %PublicDestructor);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TrivialDestroy() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %TrivialDestructor = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc15_42.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc15_42.2: init %TrivialDestructor to %a.var = class_init () [concrete = constants.%TrivialDestructor.val]
// CHECK:STDOUT:   %.loc15_3.1: init %TrivialDestructor = converted %.loc15_42.1, %.loc15_42.2 [concrete = constants.%TrivialDestructor.val]
// CHECK:STDOUT:   assign %a.var, %.loc15_3.1
// CHECK:STDOUT:   %.loc15_20: type = splice_block %TrivialDestructor.ref [concrete = constants.%TrivialDestructor] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %TrivialDestructor.ref: type = name_ref TrivialDestructor, imports.%TrivialDestructor.decl [concrete = constants.%TrivialDestructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %TrivialDestructor = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a12 = ref_binding_pattern a [concrete = constants.%a.patt.b64]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.a12 = var_pattern %a.patt [concrete = constants.%a.var_patt.d25]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc15_3.1: %Destroy.type = facet_value constants.%TrivialDestructor, (constants.%DestroyImpl.impl_witness.bf2, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet.5b3]
// CHECK:STDOUT:   %.loc15_3.2: %Destroy.type = converted constants.%TrivialDestructor, %Destroy.facet.loc15_3.1 [concrete = constants.%Destroy.facet.5b3]
// CHECK:STDOUT:   %Destroy.facet.loc15_3.2: %Destroy.type = facet_value constants.%TrivialDestructor, (constants.%DestroyImpl.impl_witness.bf2, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet.5b3]
// CHECK:STDOUT:   %.loc15_3.3: %Destroy.type = converted constants.%TrivialDestructor, %Destroy.facet.loc15_3.2 [concrete = constants.%Destroy.facet.5b3]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.5b3) [concrete = constants.%SelfDestruct.specific_fn.753]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15(%self.param: ref %TrivialDestructor) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_destroy_protected_base_destructor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %ProtectedDestructor: type = class_type @ProtectedDestructor [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %ProtectedDestructor [concrete]
// CHECK:STDOUT:   %struct_type.base.454: type = struct_type {.base: %ProtectedDestructor} [concrete]
// CHECK:STDOUT:   %pattern_type.746: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.746 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.746 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %empty_struct.a40: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.f5e: type = struct_type {.base: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.base.f5e = struct_value (%empty_struct.a40) [concrete]
// CHECK:STDOUT:   %.a6c: type = partial_type %ProtectedDestructor [concrete]
// CHECK:STDOUT:   %empty_struct.ec1: %.a6c = struct_value () [concrete]
// CHECK:STDOUT:   %ProtectedDestructor.val: %ProtectedDestructor = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.val: %Derived = struct_value (%ProtectedDestructor.val) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %ProtectedDestructor.cpp_destructor.type: type = fn_type @ProtectedDestructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %ProtectedDestructor.cpp_destructor: %ProtectedDestructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fe1: <witness> = custom_witness (%ProtectedDestructor.cpp_destructor), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e5d: %SubobjectDestroy.type = facet_value %ProtectedDestructor, (%custom_witness.fe1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.721: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e5d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.fe4: %Destroy.type = facet_value %ProtectedDestructor, (%DestroyImpl.impl_witness.721, %custom_witness.fe1) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1fc: %SubobjectDestroy.type = facet_value %Derived, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.660: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1fc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.35d: %Destroy.type = facet_value %Derived, (%DestroyImpl.impl_witness.660, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyClassWithProtectedBaseDestructor() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Derived = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc17_37.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct.a40]
// CHECK:STDOUT:   %.loc17_38.1: %struct_type.base.f5e = struct_literal (%.loc17_37.1) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc17_38.2: ref %.a6c = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc17_37.2: init %.a6c to %.loc17_38.2 = class_init () [concrete = constants.%empty_struct.ec1]
// CHECK:STDOUT:   %.loc17_38.3: init %.a6c = converted %.loc17_37.1, %.loc17_37.2 [concrete = constants.%empty_struct.ec1]
// CHECK:STDOUT:   %.loc17_38.4: init %ProtectedDestructor = as_compatible %.loc17_38.3 [concrete = constants.%ProtectedDestructor.val]
// CHECK:STDOUT:   %.loc17_38.5: init %Derived to %a.var = class_init (%.loc17_38.4) [concrete = constants.%Derived.val]
// CHECK:STDOUT:   %.loc17_3.1: init %Derived = converted %.loc17_38.1, %.loc17_38.5 [concrete = constants.%Derived.val]
// CHECK:STDOUT:   assign %a.var, %.loc17_3.1
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %a: ref %Derived = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.746 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.746 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc17_3.1: %Destroy.type = facet_value constants.%Derived, (constants.%DestroyImpl.impl_witness.660, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.35d]
// CHECK:STDOUT:   %.loc17_3.2: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc17_3.1 [concrete = constants.%Destroy.facet.35d]
// CHECK:STDOUT:   %Destroy.facet.loc17_3.2: %Destroy.type = facet_value constants.%Derived, (constants.%DestroyImpl.impl_witness.660, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.35d]
// CHECK:STDOUT:   %.loc17_3.3: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc17_3.2 [concrete = constants.%Destroy.facet.35d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_3.1(%self.param: ref %struct_type.base.454) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_3.1: ref %ProtectedDestructor = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc17_3.1: %Destroy.type = facet_value constants.%ProtectedDestructor, (constants.%DestroyImpl.impl_witness.721, constants.%custom_witness.fe1) [concrete = constants.%Destroy.facet.fe4]
// CHECK:STDOUT:   %.loc17_3.2: %Destroy.type = converted constants.%ProtectedDestructor, %Destroy.facet.loc17_3.1 [concrete = constants.%Destroy.facet.fe4]
// CHECK:STDOUT:   %Destroy.facet.loc17_3.2: %Destroy.type = facet_value constants.%ProtectedDestructor, (constants.%DestroyImpl.impl_witness.721, constants.%custom_witness.fe1) [concrete = constants.%Destroy.facet.fe4]
// CHECK:STDOUT:   %.loc17_3.3: %Destroy.type = converted constants.%ProtectedDestructor, %Destroy.facet.loc17_3.2 [concrete = constants.%Destroy.facet.fe4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc17_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_3.2(%self.param: ref %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %base.ref: %Derived.elem = name_ref base, @Derived.%.loc7 [concrete = @Derived.%.loc7]
// CHECK:STDOUT:   %.loc17_3.1: ref %ProtectedDestructor = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc17_3.1: %Destroy.type = facet_value constants.%ProtectedDestructor, (constants.%DestroyImpl.impl_witness.721, constants.%custom_witness.fe1) [concrete = constants.%Destroy.facet.fe4]
// CHECK:STDOUT:   %.loc17_3.2: %Destroy.type = converted constants.%ProtectedDestructor, %Destroy.facet.loc17_3.1 [concrete = constants.%Destroy.facet.fe4]
// CHECK:STDOUT:   %Destroy.facet.loc17_3.2: %Destroy.type = facet_value constants.%ProtectedDestructor, (constants.%DestroyImpl.impl_witness.721, constants.%custom_witness.fe1) [concrete = constants.%Destroy.facet.fe4]
// CHECK:STDOUT:   %.loc17_3.3: %Destroy.type = converted constants.%ProtectedDestructor, %Destroy.facet.loc17_3.2 [concrete = constants.%Destroy.facet.fe4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc17_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_destroy_private_base_destructor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %PrivateDestructor: type = class_type @PrivateDestructor [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %PrivateDestructor [concrete]
// CHECK:STDOUT:   %struct_type.base.9aa: type = struct_type {.base: %PrivateDestructor} [concrete]
// CHECK:STDOUT:   %pattern_type.746: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.746 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.746 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %empty_struct.a40: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.f5e: type = struct_type {.base: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.base.f5e = struct_value (%empty_struct.a40) [concrete]
// CHECK:STDOUT:   %.c73: type = partial_type %PrivateDestructor [concrete]
// CHECK:STDOUT:   %empty_struct.d8b: %.c73 = struct_value () [concrete]
// CHECK:STDOUT:   %PrivateDestructor.val: %PrivateDestructor = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.val: %Derived = struct_value (%PrivateDestructor.val) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %PrivateDestructor.cpp_destructor.type: type = fn_type @PrivateDestructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %PrivateDestructor.cpp_destructor: %PrivateDestructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fe1: <witness> = custom_witness (%PrivateDestructor.cpp_destructor), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.be1: %SubobjectDestroy.type = facet_value %PrivateDestructor, (%custom_witness.fe1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a8a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.be1) [concrete]
// CHECK:STDOUT:   %Destroy.facet.75a: %Destroy.type = facet_value %PrivateDestructor, (%DestroyImpl.impl_witness.a8a, %custom_witness.fe1) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1fc: %SubobjectDestroy.type = facet_value %Derived, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.660: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1fc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.35d: %Destroy.type = facet_value %Derived, (%DestroyImpl.impl_witness.660, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyClassWithPrivateBaseDestructor() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Derived = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc18_37.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct.a40]
// CHECK:STDOUT:   %.loc18_38.1: %struct_type.base.f5e = struct_literal (%.loc18_37.1) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc18_38.2: ref %.c73 = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc18_37.2: init %.c73 to %.loc18_38.2 = class_init () [concrete = constants.%empty_struct.d8b]
// CHECK:STDOUT:   %.loc18_38.3: init %.c73 = converted %.loc18_37.1, %.loc18_37.2 [concrete = constants.%empty_struct.d8b]
// CHECK:STDOUT:   %.loc18_38.4: init %PrivateDestructor = as_compatible %.loc18_38.3 [concrete = constants.%PrivateDestructor.val]
// CHECK:STDOUT:   %.loc18_38.5: init %Derived to %a.var = class_init (%.loc18_38.4) [concrete = constants.%Derived.val]
// CHECK:STDOUT:   %.loc18_3.1: init %Derived = converted %.loc18_38.1, %.loc18_38.5 [concrete = constants.%Derived.val]
// CHECK:STDOUT:   assign %a.var, %.loc18_3.1
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %a: ref %Derived = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.746 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.746 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc18_3.1: %Destroy.type = facet_value constants.%Derived, (constants.%DestroyImpl.impl_witness.660, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.35d]
// CHECK:STDOUT:   %.loc18_3.2: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc18_3.1 [concrete = constants.%Destroy.facet.35d]
// CHECK:STDOUT:   %Destroy.facet.loc18_3.2: %Destroy.type = facet_value constants.%Derived, (constants.%DestroyImpl.impl_witness.660, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.35d]
// CHECK:STDOUT:   %.loc18_3.3: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc18_3.2 [concrete = constants.%Destroy.facet.35d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_3.1(%self.param: ref %struct_type.base.9aa) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_3.1: ref %PrivateDestructor = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc18_3.1: %Destroy.type = facet_value constants.%PrivateDestructor, (constants.%DestroyImpl.impl_witness.a8a, constants.%custom_witness.fe1) [concrete = constants.%Destroy.facet.75a]
// CHECK:STDOUT:   %.loc18_3.2: %Destroy.type = converted constants.%PrivateDestructor, %Destroy.facet.loc18_3.1 [concrete = constants.%Destroy.facet.75a]
// CHECK:STDOUT:   %Destroy.facet.loc18_3.2: %Destroy.type = facet_value constants.%PrivateDestructor, (constants.%DestroyImpl.impl_witness.a8a, constants.%custom_witness.fe1) [concrete = constants.%Destroy.facet.75a]
// CHECK:STDOUT:   %.loc18_3.3: %Destroy.type = converted constants.%PrivateDestructor, %Destroy.facet.loc18_3.2 [concrete = constants.%Destroy.facet.75a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc18_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_3.2(%self.param: ref %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %base.ref: %Derived.elem = name_ref base, @Derived.%.loc7 [concrete = @Derived.%.loc7]
// CHECK:STDOUT:   %.loc18_3.1: ref %PrivateDestructor = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc18_3.1: %Destroy.type = facet_value constants.%PrivateDestructor, (constants.%DestroyImpl.impl_witness.a8a, constants.%custom_witness.fe1) [concrete = constants.%Destroy.facet.75a]
// CHECK:STDOUT:   %.loc18_3.2: %Destroy.type = converted constants.%PrivateDestructor, %Destroy.facet.loc18_3.1 [concrete = constants.%Destroy.facet.75a]
// CHECK:STDOUT:   %Destroy.facet.loc18_3.2: %Destroy.type = facet_value constants.%PrivateDestructor, (constants.%DestroyImpl.impl_witness.a8a, constants.%custom_witness.fe1) [concrete = constants.%Destroy.facet.75a]
// CHECK:STDOUT:   %.loc18_3.3: %Destroy.type = converted constants.%PrivateDestructor, %Destroy.facet.loc18_3.2 [concrete = constants.%Destroy.facet.75a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc18_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
