// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present --clang-arg=-std=c++20
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
  return int_ptr.(Core.CppUnsafeDeref.Op)();
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
  return int_ptr.(Core.CppUnsafeDeref.Op)();
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
// CHECK:STDOUT:   %.1da: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %TestDerefPass.type: type = fn_type @TestDerefPass [concrete]
// CHECK:STDOUT:   %TestDerefPass: %TestDerefPass.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %CppUnsafeDeref.type: type = facet_type <@CppUnsafeDeref> [concrete]
// CHECK:STDOUT:   %Self: %CppUnsafeDeref.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.WithSelf.Op.type.792: type = fn_type @CppUnsafeDeref.WithSelf.Op, @CppUnsafeDeref.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.WithSelf.Op.a80: %CppUnsafeDeref.WithSelf.Op.type.792 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %.b28: type = ref_form %impl.elem0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.assoc_type: type = assoc_entity_type @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %assoc1: %CppUnsafeDeref.assoc_type = assoc_entity element1, imports.%Core.import_ref.ad7 [concrete]
// CHECK:STDOUT:   %ConstDeref.cpp_operator.type: type = fn_type @ConstDeref.cpp_operator [concrete]
// CHECK:STDOUT:   %ConstDeref.cpp_operator: %ConstDeref.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Star__carbon_thunk.type: type = fn_type @operator_Star__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Star__carbon_thunk: %operator_Star__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3ff: <witness> = custom_witness (%i32), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %CppUnsafeDeref.facet.111: %CppUnsafeDeref.type = facet_value %ConstDeref, (%custom_witness.3ff) [concrete]
// CHECK:STDOUT:   %.7f7: type = ref_form %i32 [concrete]
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
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .ConstDeref = %ConstDeref.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstDeref.decl: type = class_decl @ConstDeref [concrete = constants.%ConstDeref] {} {}
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.CppUnsafeDeref: type = import_ref Core//prelude/operators/deref, CppUnsafeDeref, loaded [concrete = constants.%CppUnsafeDeref.type]
// CHECK:STDOUT:   %Core.import_ref.83d: %CppUnsafeDeref.assoc_type = import_ref Core//prelude/operators/deref, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc1]
// CHECK:STDOUT:   %Core.import_ref.fa7: type = import_ref Core//prelude/operators/deref, loc{{\d+_\d+}}, loaded [symbolic = @CppUnsafeDeref.WithSelf.Op.%impl.elem0 (constants.%impl.elem0)]
// CHECK:STDOUT:   %Core.import_ref.841: Core.Form = import_ref Core//prelude/operators/deref, loc{{\d+_\d+}}, loaded [symbolic = @CppUnsafeDeref.WithSelf.Op.%.1 (constants.%.b28)]
// CHECK:STDOUT:   %Core.import_ref.ad7: @CppUnsafeDeref.WithSelf.%CppUnsafeDeref.WithSelf.Op.type (%CppUnsafeDeref.WithSelf.Op.type.792) = import_ref Core//prelude/operators/deref, loc{{\d+_\d+}}, loaded [symbolic = @CppUnsafeDeref.WithSelf.%CppUnsafeDeref.WithSelf.Op (constants.%CppUnsafeDeref.WithSelf.Op.a80)]
// CHECK:STDOUT:   %ConstDeref.cpp_operator.decl: %ConstDeref.cpp_operator.type = fn_decl @ConstDeref.cpp_operator [concrete = constants.%ConstDeref.cpp_operator] {
// CHECK:STDOUT:     %self.patt: %pattern_type.3fb = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.3fb = value_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %ConstDeref = value_param call_param0
// CHECK:STDOUT:     %self: %ConstDeref = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Star__carbon_thunk.decl: %operator_Star__carbon_thunk.type = fn_decl @operator_Star__carbon_thunk [concrete = constants.%operator_Star__carbon_thunk] {
// CHECK:STDOUT:     %this.patt: %pattern_type.3fb = value_binding_pattern this [concrete]
// CHECK:STDOUT:     %this.param_patt: %pattern_type.3fb = value_param_pattern %this.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     %this.param: %ConstDeref = value_param call_param0
// CHECK:STDOUT:     %this: %ConstDeref = value_binding this, %this.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .TestDerefPass = %TestDerefPass.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp inline
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestDerefPass.decl: %TestDerefPass.type = fn_decl @TestDerefPass [concrete = constants.%TestDerefPass] {
// CHECK:STDOUT:     %int_ptr.patt: %pattern_type.3fb = ref_binding_pattern int_ptr [concrete]
// CHECK:STDOUT:     %int_ptr.param_patt: %pattern_type.3fb = var_param_pattern %int_ptr.patt [concrete]
// CHECK:STDOUT:     %int_ptr.var_patt: %pattern_type.3fb = var_pattern %int_ptr.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11_50.1: type = ref_tag %i32
// CHECK:STDOUT:     %.loc11_50.2: Core.Form = ref_form %i32 [concrete = constants.%.1da]
// CHECK:STDOUT:     %int_ptr.param: ref %ConstDeref = ref_param call_param0
// CHECK:STDOUT:     %.loc11_34: type = splice_block %ConstDeref.ref [concrete = constants.%ConstDeref] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %ConstDeref.ref: type = name_ref ConstDeref, imports.%ConstDeref.decl [concrete = constants.%ConstDeref]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %int_ptr: ref %ConstDeref = ref_binding int_ptr, %int_ptr.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ConstDeref {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDerefPass(%int_ptr.param: ref %ConstDeref) -> ref %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_ptr.ref: ref %ConstDeref = name_ref int_ptr, %int_ptr
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %CppUnsafeDeref.ref: type = name_ref CppUnsafeDeref, imports.%Core.CppUnsafeDeref [concrete = constants.%CppUnsafeDeref.type]
// CHECK:STDOUT:   %Op.ref.loc12_38: %CppUnsafeDeref.assoc_type = name_ref Op, imports.%Core.import_ref.83d [concrete = constants.%assoc1]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant @ConstDeref.cpp_operator.%i32.2 [concrete = constants.%i32]
// CHECK:STDOUT:   %.1: type = specific_constant imports.%Core.import_ref.fa7, @CppUnsafeDeref.WithSelf.Op(constants.%CppUnsafeDeref.facet.111) [concrete = constants.%i32]
// CHECK:STDOUT:   %.2: type = specific_constant imports.%Core.import_ref.841, @CppUnsafeDeref.WithSelf.Op(constants.%CppUnsafeDeref.facet.111) [concrete = constants.%.7f7]
// CHECK:STDOUT:   %ConstDeref.Op.decl: %ConstDeref.Op.type = fn_decl @ConstDeref.Op [concrete = constants.%ConstDeref.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.3fb = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.3fb = ref_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ConstDeref = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ConstDeref = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem1: %.d81 = impl_witness_access constants.%custom_witness.5a5, element1 [concrete = constants.%ConstDeref.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_ptr.ref, %impl.elem1
// CHECK:STDOUT:   %Op.ref.loc12_43: %ConstDeref.cpp_operator.type = name_ref Op, imports.%ConstDeref.cpp_operator.decl [concrete = constants.%ConstDeref.cpp_operator]
// CHECK:STDOUT:   %ConstDeref.cpp_operator.bound: <bound method> = bound_method %int_ptr.ref, %Op.ref.loc12_43
// CHECK:STDOUT:   %.loc12: %ConstDeref = acquire_value %int_ptr.ref
// CHECK:STDOUT:   %operator_Star__carbon_thunk.call: ref %i32 = call imports.%operator_Star__carbon_thunk.decl(%.loc12)
// CHECK:STDOUT:   return %operator_Star__carbon_thunk.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstDeref.cpp_operator(%self.param: %ConstDeref) -> ref %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @operator_Star__carbon_thunk(%this.param: %ConstDeref) -> ref %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstDeref.Op(%self.param: ref %ConstDeref) -> ref %i32 [thunk imports.%ConstDeref.cpp_operator.decl] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Op.ref: %ConstDeref.cpp_operator.type = name_ref Op, imports.%ConstDeref.cpp_operator.decl [concrete = constants.%ConstDeref.cpp_operator]
// CHECK:STDOUT:   %self.ref: ref %ConstDeref = name_ref self, %self.param
// CHECK:STDOUT:   %ConstDeref.cpp_operator.bound: <bound method> = bound_method %self.ref, %Op.ref
// CHECK:STDOUT:   %.1: %ConstDeref = acquire_value %self.ref
// CHECK:STDOUT:   %operator_Star__carbon_thunk.call: ref %i32 = call imports.%operator_Star__carbon_thunk.decl(%.1)
// CHECK:STDOUT:   return %operator_Star__carbon_thunk.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- mutable_deref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MutableDeref: type = class_type @MutableDeref [concrete]
// CHECK:STDOUT:   %pattern_type.b76: type = pattern_type %MutableDeref [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.1da: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %TestDerefPass.type: type = fn_type @TestDerefPass [concrete]
// CHECK:STDOUT:   %TestDerefPass: %TestDerefPass.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
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
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .MutableDeref = %MutableDeref.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MutableDeref.decl: type = class_decl @MutableDeref [concrete = constants.%MutableDeref] {} {}
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.CppUnsafeDeref: type = import_ref Core//prelude/operators/deref, CppUnsafeDeref, loaded [concrete = constants.%CppUnsafeDeref.type]
// CHECK:STDOUT:   %Core.import_ref.83d: %CppUnsafeDeref.assoc_type = import_ref Core//prelude/operators/deref, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc1]
// CHECK:STDOUT:   %Core.import_ref.ad7: @CppUnsafeDeref.WithSelf.%CppUnsafeDeref.WithSelf.Op.type (%CppUnsafeDeref.WithSelf.Op.type.792) = import_ref Core//prelude/operators/deref, loc{{\d+_\d+}}, loaded [symbolic = @CppUnsafeDeref.WithSelf.%CppUnsafeDeref.WithSelf.Op (constants.%CppUnsafeDeref.WithSelf.Op.a80)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .TestDerefPass = %TestDerefPass.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp inline
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestDerefPass.decl: %TestDerefPass.type = fn_decl @TestDerefPass [concrete = constants.%TestDerefPass] {
// CHECK:STDOUT:     %int_ptr.patt: %pattern_type.b76 = ref_binding_pattern int_ptr [concrete]
// CHECK:STDOUT:     %int_ptr.param_patt: %pattern_type.b76 = var_param_pattern %int_ptr.patt [concrete]
// CHECK:STDOUT:     %int_ptr.var_patt: %pattern_type.b76 = var_pattern %int_ptr.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11_52.1: type = ref_tag %i32
// CHECK:STDOUT:     %.loc11_52.2: Core.Form = ref_form %i32 [concrete = constants.%.1da]
// CHECK:STDOUT:     %int_ptr.param: ref %MutableDeref = ref_param call_param0
// CHECK:STDOUT:     %.loc11_34: type = splice_block %MutableDeref.ref [concrete = constants.%MutableDeref] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %MutableDeref.ref: type = name_ref MutableDeref, imports.%MutableDeref.decl [concrete = constants.%MutableDeref]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %int_ptr: ref %MutableDeref = ref_binding int_ptr, %int_ptr.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @MutableDeref {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   import Cpp//...
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
// CHECK:STDOUT: fn @MutableDeref.cpp_operator(%self.param: ref %MutableDeref) -> ref %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_multi_deref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MultiDeref: type = class_type @MultiDeref [concrete]
// CHECK:STDOUT:   %pattern_type.b40: type = pattern_type %MultiDeref [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.1da: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %TestDerefPass.type: type = fn_type @TestDerefPass [concrete]
// CHECK:STDOUT:   %TestDerefPass: %TestDerefPass.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %CppUnsafeDeref.type: type = facet_type <@CppUnsafeDeref> [concrete]
// CHECK:STDOUT:   %Self: %CppUnsafeDeref.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.WithSelf.Op.type: type = fn_type @CppUnsafeDeref.WithSelf.Op, @CppUnsafeDeref.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.WithSelf.Op: %CppUnsafeDeref.WithSelf.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.assoc_type: type = assoc_entity_type @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %assoc1: %CppUnsafeDeref.assoc_type = assoc_entity element1, imports.%Core.import_ref.ad7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .CppUnsafeDeref = %Core.CppUnsafeDeref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .MultiDeref = %MultiDeref.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MultiDeref.decl: type = class_decl @MultiDeref [concrete = constants.%MultiDeref] {} {}
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.CppUnsafeDeref: type = import_ref Core//prelude/operators/deref, CppUnsafeDeref, loaded [concrete = constants.%CppUnsafeDeref.type]
// CHECK:STDOUT:   %Core.import_ref.83d: %CppUnsafeDeref.assoc_type = import_ref Core//prelude/operators/deref, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc1]
// CHECK:STDOUT:   %Core.import_ref.ad7: @CppUnsafeDeref.WithSelf.%CppUnsafeDeref.WithSelf.Op.type (%CppUnsafeDeref.WithSelf.Op.type) = import_ref Core//prelude/operators/deref, loc{{\d+_\d+}}, loaded [symbolic = @CppUnsafeDeref.WithSelf.%CppUnsafeDeref.WithSelf.Op (constants.%CppUnsafeDeref.WithSelf.Op)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .TestDerefPass = %TestDerefPass.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp inline
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestDerefPass.decl: %TestDerefPass.type = fn_decl @TestDerefPass [concrete = constants.%TestDerefPass] {
// CHECK:STDOUT:     %int_ptr.patt: %pattern_type.b40 = ref_binding_pattern int_ptr [concrete]
// CHECK:STDOUT:     %int_ptr.param_patt: %pattern_type.b40 = var_param_pattern %int_ptr.patt [concrete]
// CHECK:STDOUT:     %int_ptr.var_patt: %pattern_type.b40 = var_pattern %int_ptr.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15_50.1: type = ref_tag %i32
// CHECK:STDOUT:     %.loc15_50.2: Core.Form = ref_form %i32 [concrete = constants.%.1da]
// CHECK:STDOUT:     %int_ptr.param: ref %MultiDeref = ref_param call_param0
// CHECK:STDOUT:     %.loc15_34: type = splice_block %MultiDeref.ref [concrete = constants.%MultiDeref] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %MultiDeref.ref: type = name_ref MultiDeref, imports.%MultiDeref.decl [concrete = constants.%MultiDeref]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %int_ptr: ref %MultiDeref = ref_binding int_ptr, %int_ptr.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @MultiDeref {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDerefPass(%int_ptr.param: ref %MultiDeref) -> ref %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_ptr.ref: ref %MultiDeref = name_ref int_ptr, %int_ptr
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %CppUnsafeDeref.ref: type = name_ref CppUnsafeDeref, imports.%Core.CppUnsafeDeref [concrete = constants.%CppUnsafeDeref.type]
// CHECK:STDOUT:   %Op.ref: %CppUnsafeDeref.assoc_type = name_ref Op, imports.%Core.import_ref.83d [concrete = constants.%assoc1]
// CHECK:STDOUT:   %impl.elem1: <error> = impl_witness_access <error>, element1 [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_maybe_deref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MaybeDeref.type: type = cpp_type_template_type MaybeDeref [concrete]
// CHECK:STDOUT:   %MaybeDeref.template: %MaybeDeref.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %MaybeDeref: type = class_type @MaybeDeref [concrete]
// CHECK:STDOUT:   %pattern_type.fc8: type = pattern_type %MaybeDeref [concrete]
// CHECK:STDOUT:   %.1da: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %TestDerefPass.type: type = fn_type @TestDerefPass [concrete]
// CHECK:STDOUT:   %TestDerefPass: %TestDerefPass.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %CppUnsafeDeref.type: type = facet_type <@CppUnsafeDeref> [concrete]
// CHECK:STDOUT:   %Self: %CppUnsafeDeref.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.WithSelf.Op.type: type = fn_type @CppUnsafeDeref.WithSelf.Op, @CppUnsafeDeref.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.WithSelf.Op: %CppUnsafeDeref.WithSelf.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.assoc_type: type = assoc_entity_type @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %assoc1: %CppUnsafeDeref.assoc_type = assoc_entity element1, imports.%Core.import_ref.ad7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .CppUnsafeDeref = %Core.CppUnsafeDeref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .MaybeDeref = %MaybeDeref.template
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MaybeDeref.template: %MaybeDeref.type = struct_value () [concrete = constants.%MaybeDeref.template]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %MaybeDeref.decl: type = class_decl @MaybeDeref [concrete = constants.%MaybeDeref] {} {}
// CHECK:STDOUT:   %Core.CppUnsafeDeref: type = import_ref Core//prelude/operators/deref, CppUnsafeDeref, loaded [concrete = constants.%CppUnsafeDeref.type]
// CHECK:STDOUT:   %Core.import_ref.83d: %CppUnsafeDeref.assoc_type = import_ref Core//prelude/operators/deref, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc1]
// CHECK:STDOUT:   %Core.import_ref.ad7: @CppUnsafeDeref.WithSelf.%CppUnsafeDeref.WithSelf.Op.type (%CppUnsafeDeref.WithSelf.Op.type) = import_ref Core//prelude/operators/deref, loc{{\d+_\d+}}, loaded [symbolic = @CppUnsafeDeref.WithSelf.%CppUnsafeDeref.WithSelf.Op (constants.%CppUnsafeDeref.WithSelf.Op)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .TestDerefPass = %TestDerefPass.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp inline
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestDerefPass.decl: %TestDerefPass.type = fn_decl @TestDerefPass [concrete = constants.%TestDerefPass] {
// CHECK:STDOUT:     %int_ptr.patt: %pattern_type.fc8 = ref_binding_pattern int_ptr [concrete]
// CHECK:STDOUT:     %int_ptr.param_patt: %pattern_type.fc8 = var_param_pattern %int_ptr.patt [concrete]
// CHECK:STDOUT:     %int_ptr.var_patt: %pattern_type.fc8 = var_pattern %int_ptr.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc21_59: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc21_55.1: type = ref_tag %i32.loc21_59
// CHECK:STDOUT:     %.loc21_55.2: Core.Form = ref_form %i32.loc21_59 [concrete = constants.%.1da]
// CHECK:STDOUT:     %int_ptr.param: ref %MaybeDeref = ref_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block imports.%MaybeDeref.decl [concrete = constants.%MaybeDeref] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %MaybeDeref.ref: %MaybeDeref.type = name_ref MaybeDeref, imports.%MaybeDeref.template [concrete = constants.%MaybeDeref.template]
// CHECK:STDOUT:       %i32.loc21_46: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %int_ptr: ref %MaybeDeref = ref_binding int_ptr, %int_ptr.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @MaybeDeref {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDerefPass(%int_ptr.param: ref %MaybeDeref) -> ref %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_ptr.ref: ref %MaybeDeref = name_ref int_ptr, %int_ptr
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %CppUnsafeDeref.ref: type = name_ref CppUnsafeDeref, imports.%Core.CppUnsafeDeref [concrete = constants.%CppUnsafeDeref.type]
// CHECK:STDOUT:   %Op.ref: %CppUnsafeDeref.assoc_type = name_ref Op, imports.%Core.import_ref.83d [concrete = constants.%assoc1]
// CHECK:STDOUT:   %impl.elem1: <error> = impl_witness_access <error>, element1 [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_not_a_ptr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NotAPtr: type = class_type @NotAPtr [concrete]
// CHECK:STDOUT:   %pattern_type.c62: type = pattern_type %NotAPtr [concrete]
// CHECK:STDOUT:   %TestDerefFail.type: type = fn_type @TestDerefFail [concrete]
// CHECK:STDOUT:   %TestDerefFail: %TestDerefFail.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %CppUnsafeDeref.type: type = facet_type <@CppUnsafeDeref> [concrete]
// CHECK:STDOUT:   %Self: %CppUnsafeDeref.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.WithSelf.Op.type: type = fn_type @CppUnsafeDeref.WithSelf.Op, @CppUnsafeDeref.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.WithSelf.Op: %CppUnsafeDeref.WithSelf.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.assoc_type: type = assoc_entity_type @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %assoc1: %CppUnsafeDeref.assoc_type = assoc_entity element1, imports.%Core.import_ref.ad7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .CppUnsafeDeref = %Core.CppUnsafeDeref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NotAPtr = %NotAPtr.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotAPtr.decl: type = class_decl @NotAPtr [concrete = constants.%NotAPtr] {} {}
// CHECK:STDOUT:   %Core.CppUnsafeDeref: type = import_ref Core//prelude/operators/deref, CppUnsafeDeref, loaded [concrete = constants.%CppUnsafeDeref.type]
// CHECK:STDOUT:   %Core.import_ref.83d: %CppUnsafeDeref.assoc_type = import_ref Core//prelude/operators/deref, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc1]
// CHECK:STDOUT:   %Core.import_ref.ad7: @CppUnsafeDeref.WithSelf.%CppUnsafeDeref.WithSelf.Op.type (%CppUnsafeDeref.WithSelf.Op.type) = import_ref Core//prelude/operators/deref, loc{{\d+_\d+}}, loaded [symbolic = @CppUnsafeDeref.WithSelf.%CppUnsafeDeref.WithSelf.Op (constants.%CppUnsafeDeref.WithSelf.Op)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .TestDerefFail = %TestDerefFail.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp inline
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestDerefFail.decl: %TestDerefFail.type = fn_decl @TestDerefFail [concrete = constants.%TestDerefFail] {
// CHECK:STDOUT:     %not_ptr.patt: %pattern_type.c62 = value_binding_pattern not_ptr [concrete]
// CHECK:STDOUT:     %not_ptr.param_patt: %pattern_type.c62 = value_param_pattern %not_ptr.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %not_ptr.param: %NotAPtr = value_param call_param0
// CHECK:STDOUT:     %.loc8: type = splice_block %NotAPtr.ref [concrete = constants.%NotAPtr] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %NotAPtr.ref: type = name_ref NotAPtr, imports.%NotAPtr.decl [concrete = constants.%NotAPtr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %not_ptr: %NotAPtr = value_binding not_ptr, %not_ptr.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NotAPtr {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDerefFail(%not_ptr.param: %NotAPtr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %not_ptr.ref: %NotAPtr = name_ref not_ptr, %not_ptr
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %CppUnsafeDeref.ref: type = name_ref CppUnsafeDeref, imports.%Core.CppUnsafeDeref [concrete = constants.%CppUnsafeDeref.type]
// CHECK:STDOUT:   %Op.ref: %CppUnsafeDeref.assoc_type = name_ref Op, imports.%Core.import_ref.83d [concrete = constants.%assoc1]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
