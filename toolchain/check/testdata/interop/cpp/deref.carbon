// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// EXTRA-ARGS: --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/deref.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/deref.carbon

// --- const_deref.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
class ConstDeref {
public:
  auto operator*() const -> int&;
};
''';

fn TestDerefPass(var int_ptr: Cpp.ConstDeref) -> ref i32 {
  //@dump-sem-ir-begin
  return int_ptr.(Core.CppUnsafeDeref.Op)();
  //@dump-sem-ir-end
}

// --- mutable_deref.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
class MutableDeref {
public:
  auto operator*() -> int&;
};
''';

fn TestDerefPass(var int_ptr: Cpp.MutableDeref) -> ref i32 {
  //@dump-sem-ir-begin
  return int_ptr.(Core.CppUnsafeDeref.Op)();
  //@dump-sem-ir-end
}

// --- fail_todo_multi_deref.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
class MultiDeref {
public:
  auto operator*() & -> int&;
  auto operator*() const& -> const int&;

  auto operator*() && -> int&&;
  auto operator*() const&& -> int const&&;
};
''';

fn TestDerefPass(var int_ptr: Cpp.MultiDeref) -> ref i32 {
  // CHECK:STDERR: fail_todo_multi_deref.carbon:[[@LINE+4]]:10: error: semantics TODO: `operator* overload sets not implemented yet` [SemanticsTodo]
  // CHECK:STDERR:   return int_ptr.(Core.CppUnsafeDeref.Op)();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return int_ptr.(Core.CppUnsafeDeref.Op)();
}

// --- fail_todo_maybe_deref.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
template<class T, class U>
concept SameImpl = __is_same(T, U);

template<class T, class U>
concept SameAs = SameImpl<T, U> && SameImpl<U, T>;

template<class T>
class MaybeDeref {
public:
  auto operator*() const -> int&
  requires SameAs<T, int>;

  auto operator*() const -> const T&;
};
''';

fn TestDerefPass(var int_ptr: Cpp.MaybeDeref(i32)) -> ref i32 {
  // CHECK:STDERR: fail_todo_maybe_deref.carbon:[[@LINE+4]]:10: error: semantics TODO: `operator* overload sets not implemented yet` [SemanticsTodo]
  // CHECK:STDERR:   return int_ptr.(Core.CppUnsafeDeref.Op)();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return int_ptr.(Core.CppUnsafeDeref.Op)();
}

// --- fail_not_a_ptr.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
class NotAPtr {};
''';

fn TestDerefFail(not_ptr: Cpp.NotAPtr) {
  // CHECK:STDERR: fail_not_a_ptr.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.CppUnsafeDeref` in type `Cpp.NotAPtr` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   not_ptr.(Core.CppUnsafeDeref.Op)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  not_ptr.(Core.CppUnsafeDeref.Op)();
}

// CHECK:STDOUT: --- const_deref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ConstDeref: type = class_type @ConstDeref [concrete]
// CHECK:STDOUT:   %pattern_type.3fb: type = pattern_type %ConstDeref [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %CppUnsafeDeref.type: type = facet_type <@CppUnsafeDeref> [concrete]
// CHECK:STDOUT:   %Self: %CppUnsafeDeref.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.WithSelf.Op.type.792: type = fn_type @CppUnsafeDeref.WithSelf.Op, @CppUnsafeDeref.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.WithSelf.Op.a80: %CppUnsafeDeref.WithSelf.Op.type.792 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.assoc_type: type = assoc_entity_type @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %assoc1: %CppUnsafeDeref.assoc_type = assoc_entity element1, imports.%Core.import_ref.ad7 [concrete]
// CHECK:STDOUT:   %ConstDeref.cpp_operator.type: type = fn_type @ConstDeref.cpp_operator [concrete]
// CHECK:STDOUT:   %ConstDeref.cpp_operator: %ConstDeref.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Star__carbon_thunk.type: type = fn_type @operator_Star__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Star__carbon_thunk: %operator_Star__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstDeref.Op.type: type = fn_type @ConstDeref.Op [concrete]
// CHECK:STDOUT:   %ConstDeref.Op: %ConstDeref.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5a5: <witness> = custom_witness (%i32, %ConstDeref.Op), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %CppUnsafeDeref.facet.5c0: %CppUnsafeDeref.type = facet_value %ConstDeref, (%custom_witness.5a5) [concrete]
// CHECK:STDOUT:   %CppUnsafeDeref.WithSelf.Op.type.d8f: type = fn_type @CppUnsafeDeref.WithSelf.Op, @CppUnsafeDeref.WithSelf(%CppUnsafeDeref.facet.5c0) [concrete]
// CHECK:STDOUT:   %.d81: type = fn_type_with_self_type %CppUnsafeDeref.WithSelf.Op.type.d8f, %CppUnsafeDeref.facet.5c0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .CppUnsafeDeref = %Core.CppUnsafeDeref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.CppUnsafeDeref: type = import_ref Core//prelude/operators/deref, CppUnsafeDeref, loaded [concrete = constants.%CppUnsafeDeref.type]
// CHECK:STDOUT:   %Core.import_ref.83d: %CppUnsafeDeref.assoc_type = import_ref Core//prelude/operators/deref, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc1]
// CHECK:STDOUT:   %Core.import_ref.ad7: @CppUnsafeDeref.WithSelf.%CppUnsafeDeref.WithSelf.Op.type (%CppUnsafeDeref.WithSelf.Op.type.792) = import_ref Core//prelude/operators/deref, loc{{\d+_\d+}}, loaded [symbolic = @CppUnsafeDeref.WithSelf.%CppUnsafeDeref.WithSelf.Op (constants.%CppUnsafeDeref.WithSelf.Op.a80)]
// CHECK:STDOUT:   %ConstDeref.cpp_operator.decl: %ConstDeref.cpp_operator.type = fn_decl @ConstDeref.cpp_operator [concrete = constants.%ConstDeref.cpp_operator] {
// CHECK:STDOUT:     %self.patt: %pattern_type.3fb = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.3fb = value_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %self.param: %ConstDeref = value_param call_param0
// CHECK:STDOUT:     %self: %ConstDeref = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Star__carbon_thunk.decl: %operator_Star__carbon_thunk.type = fn_decl @operator_Star__carbon_thunk [concrete = constants.%operator_Star__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDerefPass(%int_ptr.param: ref %ConstDeref) -> ref %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_ptr.ref: ref %ConstDeref = name_ref int_ptr, %int_ptr
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %CppUnsafeDeref.ref: type = name_ref CppUnsafeDeref, imports.%Core.CppUnsafeDeref [concrete = constants.%CppUnsafeDeref.type]
// CHECK:STDOUT:   %Op.ref.loc13_38: %CppUnsafeDeref.assoc_type = name_ref Op, imports.%Core.import_ref.83d [concrete = constants.%assoc1]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant @ConstDeref.cpp_operator.%i32.2 [concrete = constants.%i32]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem1: %.d81 = impl_witness_access constants.%custom_witness.5a5, element1 [concrete = constants.%ConstDeref.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_ptr.ref, %impl.elem1
// CHECK:STDOUT:   %Op.ref.loc13_43: %ConstDeref.cpp_operator.type = name_ref Op, imports.%ConstDeref.cpp_operator.decl [concrete = constants.%ConstDeref.cpp_operator]
// CHECK:STDOUT:   %ConstDeref.cpp_operator.bound: <bound method> = bound_method %int_ptr.ref, %Op.ref.loc13_43
// CHECK:STDOUT:   %.loc13: %ConstDeref = acquire_value %int_ptr.ref
// CHECK:STDOUT:   %operator_Star__carbon_thunk.call: ref %i32 = call imports.%operator_Star__carbon_thunk.decl(%.loc13)
// CHECK:STDOUT:   return %operator_Star__carbon_thunk.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- mutable_deref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MutableDeref: type = class_type @MutableDeref [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %CppUnsafeDeref.type: type = facet_type <@CppUnsafeDeref> [concrete]
// CHECK:STDOUT:   %Self: %CppUnsafeDeref.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.WithSelf.Op.type.792: type = fn_type @CppUnsafeDeref.WithSelf.Op, @CppUnsafeDeref.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.WithSelf.Op.a80: %CppUnsafeDeref.WithSelf.Op.type.792 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.assoc_type: type = assoc_entity_type @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %assoc1: %CppUnsafeDeref.assoc_type = assoc_entity element1, imports.%Core.import_ref.ad7 [concrete]
// CHECK:STDOUT:   %MutableDeref.cpp_operator.type: type = fn_type @MutableDeref.cpp_operator [concrete]
// CHECK:STDOUT:   %MutableDeref.cpp_operator: %MutableDeref.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e20: <witness> = custom_witness (%i32, %MutableDeref.cpp_operator), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %CppUnsafeDeref.facet.3f6: %CppUnsafeDeref.type = facet_value %MutableDeref, (%custom_witness.e20) [concrete]
// CHECK:STDOUT:   %CppUnsafeDeref.WithSelf.Op.type.cfd: type = fn_type @CppUnsafeDeref.WithSelf.Op, @CppUnsafeDeref.WithSelf(%CppUnsafeDeref.facet.3f6) [concrete]
// CHECK:STDOUT:   %.c6d: type = fn_type_with_self_type %CppUnsafeDeref.WithSelf.Op.type.cfd, %CppUnsafeDeref.facet.3f6 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .CppUnsafeDeref = %Core.CppUnsafeDeref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.CppUnsafeDeref: type = import_ref Core//prelude/operators/deref, CppUnsafeDeref, loaded [concrete = constants.%CppUnsafeDeref.type]
// CHECK:STDOUT:   %Core.import_ref.83d: %CppUnsafeDeref.assoc_type = import_ref Core//prelude/operators/deref, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc1]
// CHECK:STDOUT:   %Core.import_ref.ad7: @CppUnsafeDeref.WithSelf.%CppUnsafeDeref.WithSelf.Op.type (%CppUnsafeDeref.WithSelf.Op.type.792) = import_ref Core//prelude/operators/deref, loc{{\d+_\d+}}, loaded [symbolic = @CppUnsafeDeref.WithSelf.%CppUnsafeDeref.WithSelf.Op (constants.%CppUnsafeDeref.WithSelf.Op.a80)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDerefPass(%int_ptr.param: ref %MutableDeref) -> ref %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_ptr.ref: ref %MutableDeref = name_ref int_ptr, %int_ptr
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %CppUnsafeDeref.ref: type = name_ref CppUnsafeDeref, imports.%Core.CppUnsafeDeref [concrete = constants.%CppUnsafeDeref.type]
// CHECK:STDOUT:   %Op.ref: %CppUnsafeDeref.assoc_type = name_ref Op, imports.%Core.import_ref.83d [concrete = constants.%assoc1]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant @MutableDeref.cpp_operator.%i32.2 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem1: %.c6d = impl_witness_access constants.%custom_witness.e20, element1 [concrete = constants.%MutableDeref.cpp_operator]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_ptr.ref, %impl.elem1
// CHECK:STDOUT:   %MutableDeref.cpp_operator.call: ref %i32 = call %bound_method(%int_ptr.ref)
// CHECK:STDOUT:   return %MutableDeref.cpp_operator.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
