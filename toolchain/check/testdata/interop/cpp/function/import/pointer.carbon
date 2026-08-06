// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/pointer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/pointer.carbon

// ============================================================================
// Non-nullable pointer as a parameter type
// ============================================================================

// --- non_nullable_pointer_param.h

struct NonNullableParamS {};

auto non_nullable_param_foo(NonNullableParamS* _Nonnull) -> void;

// --- import_non_nullable_pointer_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_nullable_pointer_param.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.NonNullableParamS = {};
  Cpp.non_nullable_param_foo(&s);
  //@dump-sem-ir-end
}

// --- fail_import_non_nullable_pointer_param_using_const_value.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_nullable_pointer_param.h";

fn G() -> const Cpp.NonNullableParamS;

fn F() {
  var s: const Cpp.NonNullableParamS = G();
  // CHECK:STDERR: fail_import_non_nullable_pointer_param_using_const_value.carbon:[[@LINE+8]]:32: error: no matching function for call to 'non_nullable_param_foo' [CppInteropParseError]
  // CHECK:STDERR:    18 |   Cpp.non_nullable_param_foo(&s);
  // CHECK:STDERR:       |                                ^
  // CHECK:STDERR: fail_import_non_nullable_pointer_param_using_const_value.carbon:[[@LINE-9]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./non_nullable_pointer_param.h:4:6: note: candidate function not viable: 1st argument ('const NonNullableParamS * _Nonnull') would lose const qualifier [CppInteropParseNote]
  // CHECK:STDERR:     4 | auto non_nullable_param_foo(NonNullableParamS* _Nonnull) -> void;
  // CHECK:STDERR:       |      ^                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.non_nullable_param_foo(&s);
}

// --- import_non_nullable_pointer_param_using_const_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_nullable_pointer_param.h";

fn G() -> const (Cpp.NonNullableParamS*);

fn F() {
  //@dump-sem-ir-begin
  var p: const (Cpp.NonNullableParamS*) = G();
  Cpp.non_nullable_param_foo(p);
  //@dump-sem-ir-end
}

// ============================================================================
// Double non-nullable pointer as a parameter type
// ============================================================================

// --- double_non_nullable_pointer_param.h

struct DoubleNonNullableParamS {};

auto double_non_nullable_param_foo(DoubleNonNullableParamS* _Nonnull * _Nonnull) -> void;

// --- import_double_non_nullable_pointer_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "double_non_nullable_pointer_param.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.DoubleNonNullableParamS = {};
  var p: Cpp.DoubleNonNullableParamS* = &s;
  Cpp.double_non_nullable_param_foo(&p);
  //@dump-sem-ir-end
}

// ============================================================================
// Non-nullable pointer to const as a parameter type
// ============================================================================

// --- non_nullable_pointer_to_const_param.h

struct NonNullablePointerToConstParamS {};

auto non_nullable_to_const_foo(const NonNullablePointerToConstParamS* _Nonnull) -> void;

// --- import_non_nullable_pointer_to_const_param_using_const.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_nullable_pointer_to_const_param.h";

fn G() -> const Cpp.NonNullablePointerToConstParamS;

fn F() {
  //@dump-sem-ir-begin
  var s: const Cpp.NonNullablePointerToConstParamS = G();
  Cpp.non_nullable_to_const_foo(&s);
  //@dump-sem-ir-end
}

// --- import_non_nullable_pointer_to_const_param_using_non_const.carbon

import Cpp library "non_nullable_pointer_to_const_param.h";

fn G() -> Cpp.NonNullablePointerToConstParamS;

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.NonNullablePointerToConstParamS = G();
  Cpp.non_nullable_to_const_foo(&s);
  //@dump-sem-ir-end
}

// ============================================================================
// Const non-nullable pointer as a parameter type
// ============================================================================

// --- const_non_nullable_pointer_param.h

struct ConstNonNullableParamS {};

auto const_non_nullable_foo(ConstNonNullableParamS* _Nonnull const) -> void;

// --- import_const_non_nullable_pointer_param_using_const.carbon

library "[[@TEST_NAME]]";

import Cpp library "const_non_nullable_pointer_param.h";

fn G() -> const (Cpp.ConstNonNullableParamS*);

fn F() {
  //@dump-sem-ir-begin
  var p: const (Cpp.ConstNonNullableParamS*) = G();
  Cpp.const_non_nullable_foo(p);
  //@dump-sem-ir-end
}

// --- import_const_non_nullable_pointer_param_using_non_const.carbon

library "[[@TEST_NAME]]";

import Cpp library "const_non_nullable_pointer_param.h";

fn G() -> Cpp.ConstNonNullableParamS*;

fn F() {
  //@dump-sem-ir-begin
  var p: Cpp.ConstNonNullableParamS* = G();
  Cpp.const_non_nullable_foo(p);
  //@dump-sem-ir-end
}

// ============================================================================
// Const nullable pointer as a parameter type
// ============================================================================

// --- const_nullable_pointer_param.h

struct ConstNullableParamS {};

auto const_nullable_foo(ConstNullableParamS* const) -> void;

// --- import_const_nullable_pointer_param_using_const.carbon

library "[[@TEST_NAME]]";

import Cpp library "const_nullable_pointer_param.h";

fn G() -> const (Cpp.ConstNullableParamS*);

fn F() {
  //@dump-sem-ir-begin
  var p: const (Cpp.ConstNullableParamS*) = G();
  Cpp.const_nullable_foo(p);
  //@dump-sem-ir-end
}

// --- import_const_nullable_pointer_param_using_non_const.carbon

library "[[@TEST_NAME]]";

import Cpp library "const_nullable_pointer_param.h";

fn G() -> Cpp.ConstNullableParamS*;

fn F() {
  //@dump-sem-ir-begin
  var p: Cpp.ConstNullableParamS* = G();
  Cpp.const_nullable_foo(p);
  //@dump-sem-ir-end
}

// ============================================================================
// Non-nullable pointer as a return value
// ============================================================================

// --- non_nullable_pointer_return.h

struct NonNullableReturnS {};

auto non_nullable_return_foo() -> NonNullableReturnS* _Nonnull;

// --- import_non_nullable_pointer_return.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_nullable_pointer_return.h";

fn IngestDoublePointer(s: Cpp.NonNullableReturnS*);

fn F() {
  //@dump-sem-ir-begin
  IngestDoublePointer(Cpp.non_nullable_return_foo());
  Cpp.non_nullable_return_foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Double non-nullable pointer as a return value
// ============================================================================

// --- double_non_nullable_pointer_return.h

struct DoubleNonNullableReturnS {};

auto double_non_nullable_return_foo() -> DoubleNonNullableReturnS* _Nonnull * _Nonnull;

// --- import_double_pointer_return.carbon

library "[[@TEST_NAME]]";

import Cpp library "double_non_nullable_pointer_return.h";

fn IngestDoublePointer(s: Cpp.DoubleNonNullableReturnS**);

fn F() {
  //@dump-sem-ir-begin
  IngestDoublePointer(Cpp.double_non_nullable_return_foo());
  Cpp.double_non_nullable_return_foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Const non-nullable pointer as a return value
// ============================================================================

// --- const_non_nullable_pointer_return.h

struct ConstNonNullableReturnS {};

auto const_non_nullable_return_foo() -> const ConstNonNullableReturnS* _Nonnull;

// --- import_const_non_nullable_pointer_return.carbon

library "[[@TEST_NAME]]";

import Cpp library "const_non_nullable_pointer_return.h";

fn IngestConstPointer(s: const Cpp.ConstNonNullableReturnS*);

fn F() {
  //@dump-sem-ir-begin
  IngestConstPointer(Cpp.const_non_nullable_return_foo());
  Cpp.const_non_nullable_return_foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Nullable pointer
// ============================================================================

// --- nullable_pointer_param.h

struct NullableParamS {};

auto nullable_param_foo(NullableParamS*) -> void;

auto nullable_param_get() -> NullableParamS*;

// --- non_nullable_pointer_arg_to_pointer_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "nullable_pointer_param.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.NullableParamS = {};
  Cpp.nullable_param_foo(&s);
  //@dump-sem-ir-end
}

// --- null_pointer_arg_to_pointer_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "nullable_pointer_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.nullable_param_foo(Core.Optional(Cpp.NullableParamS*).None());
  //@dump-sem-ir-end
}

// --- nonnull_pointer_arg_to_pointer_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "nullable_pointer_param.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.NullableParamS = {};
  Cpp.nullable_param_foo(Core.Optional(Cpp.NullableParamS*).Some(&s));
  //@dump-sem-ir-end
}

// --- forward_nullable_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp library "nullable_pointer_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.nullable_param_foo(Cpp.nullable_param_get());
  //@dump-sem-ir-end
}

// ============================================================================
// Deduced pointer type as template argument
// ============================================================================

// --- deduced_any_param.h

struct DeducedAnyS {};

template<typename T>
auto deduced_any_foo(T) -> T;

// --- import_deduced_any_param_as_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp library "deduced_any_param.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.DeducedAnyS = {};

  // We assume that this cannot return a null pointer, because the function is
  // generic in the type `T`.
  // TODO: This isn't guaranteed; is this the behavior we want?
  let unused p: Cpp.DeducedAnyS* = Cpp.deduced_any_foo(&s);
  //@dump-sem-ir-end
}

// --- deduced_pointer_param.h

struct DeducedPointerS {};

template<typename T>
auto Direct(T*) -> T*;

template<typename T>
using Wrap = T;

template<typename U>
using AddPointer = Wrap<U*>;

template<typename V>
auto Indirect(V) -> AddPointer<V>;

// --- import_deduced_pointer_param_as_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp library "deduced_pointer_param.h";

fn F() {
  // These pass or return an Optional(S*) instead of an S*.

  //@dump-sem-ir-begin
  var s: Cpp.DeducedPointerS = {};
  Cpp.Direct(&s);

  let unused a: Core.Optional(Cpp.DeducedPointerS*) = Cpp.Indirect({} as Cpp.DeducedPointerS);
  //@dump-sem-ir-end
}

// --- fail_import_deduced_pointer_param_as_optional_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp library "deduced_pointer_param.h";

fn F() {
  // This could return a null pointer, so should not be given the type `Cpp.DeducedPointerS*`.
  // CHECK:STDERR: fail_import_deduced_pointer_param_as_optional_pointer.carbon:[[@LINE+7]]:40: error: cannot implicitly convert expression of type `Core.Optional(Cpp.DeducedPointerS* as Core.OptionalStorage)` to `Cpp.DeducedPointerS*` [ConversionFailure]
  // CHECK:STDERR:   let unused a: Cpp.DeducedPointerS* = Cpp.Indirect({} as Cpp.DeducedPointerS);
  // CHECK:STDERR:                                        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_deduced_pointer_param_as_optional_pointer.carbon:[[@LINE+4]]:40: note: type `Core.Optional(Cpp.DeducedPointerS* as Core.OptionalStorage)` does not implement interface `Core.ImplicitAs(Cpp.DeducedPointerS*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused a: Cpp.DeducedPointerS* = Cpp.Indirect({} as Cpp.DeducedPointerS);
  // CHECK:STDERR:                                        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused a: Cpp.DeducedPointerS* = Cpp.Indirect({} as Cpp.DeducedPointerS);
}

// CHECK:STDOUT: --- import_non_nullable_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonNullableParamS: type = class_type @NonNullableParamS [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.3b5: type = pattern_type %NonNullableParamS [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.3b5 = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.3b5 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %NonNullableParamS.val: %NonNullableParamS = struct_value () [concrete]
// CHECK:STDOUT:   %non_nullable_param_foo.cpp_overload_set.type: type = cpp_overload_set_type @non_nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %non_nullable_param_foo.cpp_overload_set.value: %non_nullable_param_foo.cpp_overload_set.type = cpp_overload_set_value @non_nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.1dc: type = ptr_type %NonNullableParamS [concrete]
// CHECK:STDOUT:   %non_nullable_param_foo.type: type = fn_type @non_nullable_param_foo [concrete]
// CHECK:STDOUT:   %non_nullable_param_foo: %non_nullable_param_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonNullableParamS.cpp_destructor.type: type = fn_type @NonNullableParamS.cpp_destructor [concrete]
// CHECK:STDOUT:   %NonNullableParamS.cpp_destructor: %NonNullableParamS.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NonNullableParamS = %NonNullableParamS.decl
// CHECK:STDOUT:     .non_nullable_param_foo = %non_nullable_param_foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonNullableParamS.decl: type = class_decl @NonNullableParamS [concrete = constants.%NonNullableParamS] {} {}
// CHECK:STDOUT:   %non_nullable_param_foo.cpp_overload_set.value: %non_nullable_param_foo.cpp_overload_set.type = cpp_overload_set_value @non_nullable_param_foo.cpp_overload_set [concrete = constants.%non_nullable_param_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %non_nullable_param_foo.decl: %non_nullable_param_foo.type = fn_decl @non_nullable_param_foo [concrete = constants.%non_nullable_param_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref %NonNullableParamS = var_storage %s.var_patt
// CHECK:STDOUT:   %.loc8_35.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_35.2: init %NonNullableParamS to %s.var = class_init () [concrete = constants.%NonNullableParamS.val]
// CHECK:STDOUT:   %.loc8_3: init %NonNullableParamS = converted %.loc8_35.1, %.loc8_35.2 [concrete = constants.%NonNullableParamS.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %NonNullableParamS.ref [concrete = constants.%NonNullableParamS] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonNullableParamS.ref: type = name_ref NonNullableParamS, imports.%NonNullableParamS.decl [concrete = constants.%NonNullableParamS]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %NonNullableParamS = wrapper_binding s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.3b5 = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.3b5 = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %non_nullable_param_foo.ref: %non_nullable_param_foo.cpp_overload_set.type = name_ref non_nullable_param_foo, imports.%non_nullable_param_foo.cpp_overload_set.value [concrete = constants.%non_nullable_param_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %NonNullableParamS = name_ref s, %s
// CHECK:STDOUT:   %addr: %ptr.1dc = addr_of %s.ref
// CHECK:STDOUT:   %non_nullable_param_foo.call: init %empty_tuple.type = call imports.%non_nullable_param_foo.decl(%addr)
// CHECK:STDOUT:   %NonNullableParamS.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%NonNullableParamS.cpp_destructor
// CHECK:STDOUT:   %NonNullableParamS.cpp_destructor.call: init %empty_tuple.type = call %NonNullableParamS.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_nullable_pointer_param_using_const_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonNullableParamS: type = class_type @NonNullableParamS [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %NonNullableParamS [concrete]
// CHECK:STDOUT:   %const: type = const_type %ptr [concrete]
// CHECK:STDOUT:   %pattern_type.937: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.937 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:   %p.var_patt: %pattern_type.937 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   %non_nullable_param_foo.cpp_overload_set.type: type = cpp_overload_set_type @non_nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %non_nullable_param_foo.cpp_overload_set.value: %non_nullable_param_foo.cpp_overload_set.type = cpp_overload_set_value @non_nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %non_nullable_param_foo.type: type = fn_type @non_nullable_param_foo [concrete]
// CHECK:STDOUT:   %non_nullable_param_foo: %non_nullable_param_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc10_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NonNullableParamS = %NonNullableParamS.decl
// CHECK:STDOUT:     .non_nullable_param_foo = %non_nullable_param_foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonNullableParamS.decl: type = class_decl @NonNullableParamS [concrete = constants.%NonNullableParamS] {} {}
// CHECK:STDOUT:   %non_nullable_param_foo.cpp_overload_set.value: %non_nullable_param_foo.cpp_overload_set.type = cpp_overload_set_value @non_nullable_param_foo.cpp_overload_set [concrete = constants.%non_nullable_param_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %non_nullable_param_foo.decl: %non_nullable_param_foo.type = fn_decl @non_nullable_param_foo [concrete = constants.%non_nullable_param_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.var: ref %const = var_storage %p.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %const = call %G.ref()
// CHECK:STDOUT:   assign %p.var, %G.call
// CHECK:STDOUT:   %.loc10: type = splice_block %const [concrete = constants.%const] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonNullableParamS.ref: type = name_ref NonNullableParamS, imports.%NonNullableParamS.decl [concrete = constants.%NonNullableParamS]
// CHECK:STDOUT:     %ptr: type = ptr_type %NonNullableParamS.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     %const: type = const_type %ptr [concrete = constants.%const]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %const = wrapper_binding p, %p.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.937 = ref_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.937 = var_pattern %p.patt [concrete = constants.%p.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %non_nullable_param_foo.ref: %non_nullable_param_foo.cpp_overload_set.type = name_ref non_nullable_param_foo, imports.%non_nullable_param_foo.cpp_overload_set.value [concrete = constants.%non_nullable_param_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %p.ref: ref %const = name_ref p, %p
// CHECK:STDOUT:   %.loc11_30.1: ref %ptr = as_compatible %p.ref
// CHECK:STDOUT:   %.loc11_30.2: ref %ptr = converted %p.ref, %.loc11_30.1
// CHECK:STDOUT:   %.loc11_30.3: %ptr = acquire_value %.loc11_30.2
// CHECK:STDOUT:   %non_nullable_param_foo.call: init %empty_tuple.type = call imports.%non_nullable_param_foo.decl(%.loc11_30.3)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %p.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%p.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_3.1(%self.param: ref %ptr) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_3.2(%self.param: ref %const) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_double_non_nullable_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DoubleNonNullableParamS: type = class_type @DoubleNonNullableParamS [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %DoubleNonNullableParamS [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.cb1 = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.cb1 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %DoubleNonNullableParamS.val: %DoubleNonNullableParamS = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.40d: type = ptr_type %DoubleNonNullableParamS [concrete]
// CHECK:STDOUT:   %pattern_type.4da: type = pattern_type %ptr.40d [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.4da = ref_binding_pattern p [concrete]
// CHECK:STDOUT:   %p.var_patt: %pattern_type.4da = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.59f: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%DoubleNonNullableParamS) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.390: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%DoubleNonNullableParamS) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.a1c: %ptr.as.Copy.impl.Op.type.390 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.40d, (%Copy.impl_witness.59f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.744: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.3f1: type = fn_type_with_self_type %Copy.WithSelf.Op.type.744, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.a1c, @ptr.as.Copy.impl.Op(%DoubleNonNullableParamS) [concrete]
// CHECK:STDOUT:   %double_non_nullable_param_foo.cpp_overload_set.type: type = cpp_overload_set_type @double_non_nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %double_non_nullable_param_foo.cpp_overload_set.value: %double_non_nullable_param_foo.cpp_overload_set.type = cpp_overload_set_value @double_non_nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.2a5: type = ptr_type %ptr.40d [concrete]
// CHECK:STDOUT:   %double_non_nullable_param_foo.type: type = fn_type @double_non_nullable_param_foo [concrete]
// CHECK:STDOUT:   %double_non_nullable_param_foo: %double_non_nullable_param_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %DoubleNonNullableParamS.cpp_destructor.type: type = fn_type @DoubleNonNullableParamS.cpp_destructor [concrete]
// CHECK:STDOUT:   %DoubleNonNullableParamS.cpp_destructor: %DoubleNonNullableParamS.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DoubleNonNullableParamS = %DoubleNonNullableParamS.decl
// CHECK:STDOUT:     .double_non_nullable_param_foo = %double_non_nullable_param_foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DoubleNonNullableParamS.decl: type = class_decl @DoubleNonNullableParamS [concrete = constants.%DoubleNonNullableParamS] {} {}
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %double_non_nullable_param_foo.cpp_overload_set.value: %double_non_nullable_param_foo.cpp_overload_set.type = cpp_overload_set_value @double_non_nullable_param_foo.cpp_overload_set [concrete = constants.%double_non_nullable_param_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %double_non_nullable_param_foo.decl: %double_non_nullable_param_foo.type = fn_decl @double_non_nullable_param_foo [concrete = constants.%double_non_nullable_param_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref %DoubleNonNullableParamS = var_storage %s.var_patt
// CHECK:STDOUT:   %.loc8_41.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_41.2: init %DoubleNonNullableParamS to %s.var = class_init () [concrete = constants.%DoubleNonNullableParamS.val]
// CHECK:STDOUT:   %.loc8_3: init %DoubleNonNullableParamS = converted %.loc8_41.1, %.loc8_41.2 [concrete = constants.%DoubleNonNullableParamS.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %DoubleNonNullableParamS.ref.loc8 [concrete = constants.%DoubleNonNullableParamS] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DoubleNonNullableParamS.ref.loc8: type = name_ref DoubleNonNullableParamS, imports.%DoubleNonNullableParamS.decl [concrete = constants.%DoubleNonNullableParamS]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %DoubleNonNullableParamS = wrapper_binding s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.cb1 = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.cb1 = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.40d = var_storage %p.var_patt
// CHECK:STDOUT:   %s.ref: ref %DoubleNonNullableParamS = name_ref s, %s
// CHECK:STDOUT:   %addr.loc9: %ptr.40d = addr_of %s.ref
// CHECK:STDOUT:   %impl.elem0: %.3f1 = impl_witness_access constants.%Copy.impl_witness.59f, element0 [concrete = constants.%ptr.as.Copy.impl.Op.a1c]
// CHECK:STDOUT:   %bound_method.loc9_41.1: <bound method> = bound_method %addr.loc9, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%DoubleNonNullableParamS) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_41.2: <bound method> = bound_method %addr.loc9, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.40d = call %bound_method.loc9_41.2(%addr.loc9)
// CHECK:STDOUT:   assign %p.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9: type = splice_block %ptr [concrete = constants.%ptr.40d] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DoubleNonNullableParamS.ref.loc9: type = name_ref DoubleNonNullableParamS, imports.%DoubleNonNullableParamS.decl [concrete = constants.%DoubleNonNullableParamS]
// CHECK:STDOUT:     %ptr: type = ptr_type %DoubleNonNullableParamS.ref.loc9 [concrete = constants.%ptr.40d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.40d = wrapper_binding p, %p.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.4da = ref_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.4da = var_pattern %p.patt [concrete = constants.%p.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %double_non_nullable_param_foo.ref: %double_non_nullable_param_foo.cpp_overload_set.type = name_ref double_non_nullable_param_foo, imports.%double_non_nullable_param_foo.cpp_overload_set.value [concrete = constants.%double_non_nullable_param_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %p.ref: ref %ptr.40d = name_ref p, %p
// CHECK:STDOUT:   %addr.loc10: %ptr.2a5 = addr_of %p.ref
// CHECK:STDOUT:   %double_non_nullable_param_foo.call: init %empty_tuple.type = call imports.%double_non_nullable_param_foo.decl(%addr.loc10)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %p.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%p.var)
// CHECK:STDOUT:   %DoubleNonNullableParamS.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%DoubleNonNullableParamS.cpp_destructor
// CHECK:STDOUT:   %DoubleNonNullableParamS.cpp_destructor.call: init %empty_tuple.type = call %DoubleNonNullableParamS.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %ptr.40d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_nullable_pointer_to_const_param_using_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonNullablePointerToConstParamS: type = class_type @NonNullablePointerToConstParamS [concrete]
// CHECK:STDOUT:   %const: type = const_type %NonNullablePointerToConstParamS [concrete]
// CHECK:STDOUT:   %pattern_type.291: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.291 = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.291 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %non_nullable_to_const_foo.cpp_overload_set.type: type = cpp_overload_set_type @non_nullable_to_const_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %non_nullable_to_const_foo.cpp_overload_set.value: %non_nullable_to_const_foo.cpp_overload_set.type = cpp_overload_set_value @non_nullable_to_const_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.62c: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %non_nullable_to_const_foo.type: type = fn_type @non_nullable_to_const_foo [concrete]
// CHECK:STDOUT:   %non_nullable_to_const_foo: %non_nullable_to_const_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NonNullablePointerToConstParamS = %NonNullablePointerToConstParamS.decl
// CHECK:STDOUT:     .non_nullable_to_const_foo = %non_nullable_to_const_foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonNullablePointerToConstParamS.decl: type = class_decl @NonNullablePointerToConstParamS [concrete = constants.%NonNullablePointerToConstParamS] {} {}
// CHECK:STDOUT:   %non_nullable_to_const_foo.cpp_overload_set.value: %non_nullable_to_const_foo.cpp_overload_set.type = cpp_overload_set_value @non_nullable_to_const_foo.cpp_overload_set [concrete = constants.%non_nullable_to_const_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %non_nullable_to_const_foo.decl: %non_nullable_to_const_foo.type = fn_decl @non_nullable_to_const_foo [concrete = constants.%non_nullable_to_const_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref %const = var_storage %s.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc10_3: ref %const = splice_block %s.var {}
// CHECK:STDOUT:   %G.call: init %const to %.loc10_3 = call %G.ref()
// CHECK:STDOUT:   assign %s.var, %G.call
// CHECK:STDOUT:   %.loc10_10: type = splice_block %const [concrete = constants.%const] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonNullablePointerToConstParamS.ref: type = name_ref NonNullablePointerToConstParamS, imports.%NonNullablePointerToConstParamS.decl [concrete = constants.%NonNullablePointerToConstParamS]
// CHECK:STDOUT:     %const: type = const_type %NonNullablePointerToConstParamS.ref [concrete = constants.%const]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %const = wrapper_binding s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.291 = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.291 = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %non_nullable_to_const_foo.ref: %non_nullable_to_const_foo.cpp_overload_set.type = name_ref non_nullable_to_const_foo, imports.%non_nullable_to_const_foo.cpp_overload_set.value [concrete = constants.%non_nullable_to_const_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %const = name_ref s, %s
// CHECK:STDOUT:   %addr: %ptr.62c = addr_of %s.ref
// CHECK:STDOUT:   %non_nullable_to_const_foo.call: init %empty_tuple.type = call imports.%non_nullable_to_const_foo.decl(%addr)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %s.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %const) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_nullable_pointer_to_const_param_using_non_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonNullablePointerToConstParamS: type = class_type @NonNullablePointerToConstParamS [concrete]
// CHECK:STDOUT:   %pattern_type.2f7: type = pattern_type %NonNullablePointerToConstParamS [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.2f7 = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.2f7 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %non_nullable_to_const_foo.cpp_overload_set.type: type = cpp_overload_set_type @non_nullable_to_const_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %non_nullable_to_const_foo.cpp_overload_set.value: %non_nullable_to_const_foo.cpp_overload_set.type = cpp_overload_set_value @non_nullable_to_const_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.faa: type = ptr_type %NonNullablePointerToConstParamS [concrete]
// CHECK:STDOUT:   %const: type = const_type %NonNullablePointerToConstParamS [concrete]
// CHECK:STDOUT:   %ptr.62c: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %non_nullable_to_const_foo.type: type = fn_type @non_nullable_to_const_foo [concrete]
// CHECK:STDOUT:   %non_nullable_to_const_foo: %non_nullable_to_const_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonNullablePointerToConstParamS.cpp_destructor.type: type = fn_type @NonNullablePointerToConstParamS.cpp_destructor [concrete]
// CHECK:STDOUT:   %NonNullablePointerToConstParamS.cpp_destructor: %NonNullablePointerToConstParamS.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NonNullablePointerToConstParamS = %NonNullablePointerToConstParamS.decl
// CHECK:STDOUT:     .non_nullable_to_const_foo = %non_nullable_to_const_foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonNullablePointerToConstParamS.decl: type = class_decl @NonNullablePointerToConstParamS [concrete = constants.%NonNullablePointerToConstParamS] {} {}
// CHECK:STDOUT:   %non_nullable_to_const_foo.cpp_overload_set.value: %non_nullable_to_const_foo.cpp_overload_set.type = cpp_overload_set_value @non_nullable_to_const_foo.cpp_overload_set [concrete = constants.%non_nullable_to_const_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %non_nullable_to_const_foo.decl: %non_nullable_to_const_foo.type = fn_decl @non_nullable_to_const_foo [concrete = constants.%non_nullable_to_const_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref %NonNullablePointerToConstParamS = var_storage %s.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc8_3: ref %NonNullablePointerToConstParamS = splice_block %s.var {}
// CHECK:STDOUT:   %G.call: init %NonNullablePointerToConstParamS to %.loc8_3 = call %G.ref()
// CHECK:STDOUT:   assign %s.var, %G.call
// CHECK:STDOUT:   %.loc8_13: type = splice_block %NonNullablePointerToConstParamS.ref [concrete = constants.%NonNullablePointerToConstParamS] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonNullablePointerToConstParamS.ref: type = name_ref NonNullablePointerToConstParamS, imports.%NonNullablePointerToConstParamS.decl [concrete = constants.%NonNullablePointerToConstParamS]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %NonNullablePointerToConstParamS = wrapper_binding s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.2f7 = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.2f7 = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %non_nullable_to_const_foo.ref: %non_nullable_to_const_foo.cpp_overload_set.type = name_ref non_nullable_to_const_foo, imports.%non_nullable_to_const_foo.cpp_overload_set.value [concrete = constants.%non_nullable_to_const_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %NonNullablePointerToConstParamS = name_ref s, %s
// CHECK:STDOUT:   %addr: %ptr.faa = addr_of %s.ref
// CHECK:STDOUT:   %.loc9_33.1: %ptr.62c = as_compatible %addr
// CHECK:STDOUT:   %.loc9_33.2: %ptr.62c = converted %addr, %.loc9_33.1
// CHECK:STDOUT:   %non_nullable_to_const_foo.call: init %empty_tuple.type = call imports.%non_nullable_to_const_foo.decl(%.loc9_33.2)
// CHECK:STDOUT:   %NonNullablePointerToConstParamS.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%NonNullablePointerToConstParamS.cpp_destructor
// CHECK:STDOUT:   %NonNullablePointerToConstParamS.cpp_destructor.call: init %empty_tuple.type = call %NonNullablePointerToConstParamS.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_const_non_nullable_pointer_param_using_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ConstNonNullableParamS: type = class_type @ConstNonNullableParamS [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %ConstNonNullableParamS [concrete]
// CHECK:STDOUT:   %const: type = const_type %ptr [concrete]
// CHECK:STDOUT:   %pattern_type.6b4: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.6b4 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:   %p.var_patt: %pattern_type.6b4 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   %const_non_nullable_foo.cpp_overload_set.type: type = cpp_overload_set_type @const_non_nullable_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const_non_nullable_foo.cpp_overload_set.value: %const_non_nullable_foo.cpp_overload_set.type = cpp_overload_set_value @const_non_nullable_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const_non_nullable_foo.type: type = fn_type @const_non_nullable_foo [concrete]
// CHECK:STDOUT:   %const_non_nullable_foo: %const_non_nullable_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc10_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .ConstNonNullableParamS = %ConstNonNullableParamS.decl
// CHECK:STDOUT:     .const_non_nullable_foo = %const_non_nullable_foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstNonNullableParamS.decl: type = class_decl @ConstNonNullableParamS [concrete = constants.%ConstNonNullableParamS] {} {}
// CHECK:STDOUT:   %const_non_nullable_foo.cpp_overload_set.value: %const_non_nullable_foo.cpp_overload_set.type = cpp_overload_set_value @const_non_nullable_foo.cpp_overload_set [concrete = constants.%const_non_nullable_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %const_non_nullable_foo.decl: %const_non_nullable_foo.type = fn_decl @const_non_nullable_foo [concrete = constants.%const_non_nullable_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.var: ref %const = var_storage %p.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %const = call %G.ref()
// CHECK:STDOUT:   assign %p.var, %G.call
// CHECK:STDOUT:   %.loc10: type = splice_block %const [concrete = constants.%const] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ConstNonNullableParamS.ref: type = name_ref ConstNonNullableParamS, imports.%ConstNonNullableParamS.decl [concrete = constants.%ConstNonNullableParamS]
// CHECK:STDOUT:     %ptr: type = ptr_type %ConstNonNullableParamS.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     %const: type = const_type %ptr [concrete = constants.%const]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %const = wrapper_binding p, %p.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.6b4 = ref_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.6b4 = var_pattern %p.patt [concrete = constants.%p.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %const_non_nullable_foo.ref: %const_non_nullable_foo.cpp_overload_set.type = name_ref const_non_nullable_foo, imports.%const_non_nullable_foo.cpp_overload_set.value [concrete = constants.%const_non_nullable_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %p.ref: ref %const = name_ref p, %p
// CHECK:STDOUT:   %.loc11_30.1: ref %ptr = as_compatible %p.ref
// CHECK:STDOUT:   %.loc11_30.2: ref %ptr = converted %p.ref, %.loc11_30.1
// CHECK:STDOUT:   %.loc11_30.3: %ptr = acquire_value %.loc11_30.2
// CHECK:STDOUT:   %const_non_nullable_foo.call: init %empty_tuple.type = call imports.%const_non_nullable_foo.decl(%.loc11_30.3)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %p.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%p.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_3.1(%self.param: ref %ptr) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_3.2(%self.param: ref %const) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_const_non_nullable_pointer_param_using_non_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ConstNonNullableParamS: type = class_type @ConstNonNullableParamS [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %ConstNonNullableParamS [concrete]
// CHECK:STDOUT:   %pattern_type.c28: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.c28 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:   %p.var_patt: %pattern_type.c28 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   %const_non_nullable_foo.cpp_overload_set.type: type = cpp_overload_set_type @const_non_nullable_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const_non_nullable_foo.cpp_overload_set.value: %const_non_nullable_foo.cpp_overload_set.type = cpp_overload_set_value @const_non_nullable_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const_non_nullable_foo.type: type = fn_type @const_non_nullable_foo [concrete]
// CHECK:STDOUT:   %const_non_nullable_foo: %const_non_nullable_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .ConstNonNullableParamS = %ConstNonNullableParamS.decl
// CHECK:STDOUT:     .const_non_nullable_foo = %const_non_nullable_foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstNonNullableParamS.decl: type = class_decl @ConstNonNullableParamS [concrete = constants.%ConstNonNullableParamS] {} {}
// CHECK:STDOUT:   %const_non_nullable_foo.cpp_overload_set.value: %const_non_nullable_foo.cpp_overload_set.type = cpp_overload_set_value @const_non_nullable_foo.cpp_overload_set [concrete = constants.%const_non_nullable_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %const_non_nullable_foo.decl: %const_non_nullable_foo.type = fn_decl @const_non_nullable_foo [concrete = constants.%const_non_nullable_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.var: ref %ptr = var_storage %p.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %ptr = call %G.ref()
// CHECK:STDOUT:   assign %p.var, %G.call
// CHECK:STDOUT:   %.loc10: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ConstNonNullableParamS.ref: type = name_ref ConstNonNullableParamS, imports.%ConstNonNullableParamS.decl [concrete = constants.%ConstNonNullableParamS]
// CHECK:STDOUT:     %ptr: type = ptr_type %ConstNonNullableParamS.ref [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr = wrapper_binding p, %p.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.c28 = ref_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.c28 = var_pattern %p.patt [concrete = constants.%p.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %const_non_nullable_foo.ref: %const_non_nullable_foo.cpp_overload_set.type = name_ref const_non_nullable_foo, imports.%const_non_nullable_foo.cpp_overload_set.value [concrete = constants.%const_non_nullable_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %p.ref: ref %ptr = name_ref p, %p
// CHECK:STDOUT:   %.loc11: %ptr = acquire_value %p.ref
// CHECK:STDOUT:   %const_non_nullable_foo.call: init %empty_tuple.type = call imports.%const_non_nullable_foo.decl(%.loc11)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %p.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%p.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %ptr) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_const_nullable_pointer_param_using_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ConstNullableParamS: type = class_type @ConstNullableParamS [concrete]
// CHECK:STDOUT:   %ptr.13b: type = ptr_type %ConstNullableParamS [concrete]
// CHECK:STDOUT:   %const.2f4: type = const_type %ptr.13b [concrete]
// CHECK:STDOUT:   %pattern_type.170: type = pattern_type %const.2f4 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.170 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:   %p.var_patt: %pattern_type.170 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   %const_nullable_foo.cpp_overload_set.type: type = cpp_overload_set_type @const_nullable_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const_nullable_foo.cpp_overload_set.value: %const_nullable_foo.cpp_overload_set.type = cpp_overload_set_value @const_nullable_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.781: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.e46: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.781) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.ddb: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.e46) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.71e: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.caa: %ptr.as.OptionalStorage.impl.Some.type.71e = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.1: <witness> = custom_witness (%Destroy.Op.1a2547.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.504: %Destroy.type = facet_value %ptr.13b, (%custom_witness.df9cc1.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.c09: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.504) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.cb6: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%ConstNullableParamS) [concrete]
// CHECK:STDOUT:   %.c5e2: type = maybe_unformed_type %ptr.13b [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.f04: %OptionalStorage.type = facet_value %ptr.13b, (%OptionalStorage.impl_witness.cb6) [concrete]
// CHECK:STDOUT:   %Optional.7c0: type = class_type @Optional, @Optional(%OptionalStorage.facet.f04) [concrete]
// CHECK:STDOUT:   %const_nullable_foo.type: type = fn_type @const_nullable_foo [concrete]
// CHECK:STDOUT:   %const_nullable_foo: %const_nullable_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3dc: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.7c0)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3aa489.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T.67d)> [symbolic]
// CHECK:STDOUT:   %U.294: %ImplicitAs.type.3aa489.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.type.f7e: type = fn_type @const.as.ImplicitAs.impl.Convert, @const.as.ImplicitAs.impl(%T.67d, %U.294) [symbolic]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.2f7: %const.as.ImplicitAs.impl.Convert.type.f7e = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.3b8: type = facet_type <@OptionalAs, @OptionalAs(%T.220)> [symbolic]
// CHECK:STDOUT:   %U.042: %OptionalAs.type.3b8 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.ad5: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.167(%T.220, %U.042) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.331: %U.as_type.as.ImplicitAs.impl.Convert.type.ad5 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.cda: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.f04)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.774: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.220) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.5f9: %T.as_type.as.OptionalAs.impl.Convert.type.774 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.f8a: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.2e9, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.f04) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.cda = facet_value %ptr.13b, (%OptionalAs.impl_witness.f8a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.855: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bb5, @U.as_type.as.ImplicitAs.impl.167(%OptionalStorage.facet.f04, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.464: %ImplicitAs.type.3dc = facet_value %ptr.13b, (%ImplicitAs.impl_witness.855) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4fe: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5de, @const.as.ImplicitAs.impl(%Optional.7c0, %ImplicitAs.facet.464) [concrete]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.type.129: type = fn_type @const.as.ImplicitAs.impl.Convert, @const.as.ImplicitAs.impl(%Optional.7c0, %ImplicitAs.facet.464) [concrete]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.7ef: %const.as.ImplicitAs.impl.Convert.type.129 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.7f0: %ImplicitAs.type.3dc = facet_value %const.2f4, (%ImplicitAs.impl_witness.4fe) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.25f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.7c0, %ImplicitAs.facet.7f0) [concrete]
// CHECK:STDOUT:   %.a05: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.25f, %ImplicitAs.facet.7f0 [concrete]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %const.as.ImplicitAs.impl.Convert.7ef, @const.as.ImplicitAs.impl.Convert(%Optional.7c0, %ImplicitAs.facet.464) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc11_26.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.5: type = fn_type @Destroy.Op.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.5: %Destroy.Op.type.1d8f74.5 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .ConstNullableParamS = %ConstNullableParamS.decl
// CHECK:STDOUT:     .const_nullable_foo = %const_nullable_foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstNullableParamS.decl: type = class_decl @ConstNullableParamS [concrete = constants.%ConstNullableParamS] {} {}
// CHECK:STDOUT:   %const_nullable_foo.cpp_overload_set.value: %const_nullable_foo.cpp_overload_set.type = cpp_overload_set_value @const_nullable_foo.cpp_overload_set [concrete = constants.%const_nullable_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.509: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.ddb)]
// CHECK:STDOUT:   %Core.import_ref.105 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.fa7: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.71e) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.caa)]
// CHECK:STDOUT:   %Core.import_ref.a90 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.306 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3e3 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.509, %Core.import_ref.105, %Core.import_ref.fa7, %Core.import_ref.a90, %Core.import_ref.306, %Core.import_ref.3e3), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %const_nullable_foo.decl: %const_nullable_foo.type = fn_decl @const_nullable_foo [concrete = constants.%const_nullable_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_27.1: type = splice_block %Optional [concrete = constants.%Optional.7c0] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.13b, (constants.%OptionalStorage.impl_witness.cb6) [concrete = constants.%OptionalStorage.facet.f04]
// CHECK:STDOUT:       %.loc11_27.2: %OptionalStorage.type = converted constants.%ptr.13b, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.f04]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.f04) [concrete = constants.%Optional.7c0]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.a74: @const.as.ImplicitAs.impl.%const.as.ImplicitAs.impl.Convert.type (%const.as.ImplicitAs.impl.Convert.type.f7e) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @const.as.ImplicitAs.impl.%const.as.ImplicitAs.impl.Convert (constants.%const.as.ImplicitAs.impl.Convert.2f7)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5de = impl_witness_table (%Core.import_ref.a74), @const.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.eb9: @U.as_type.as.ImplicitAs.impl.167.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.ad5) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.167.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.331)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bb5 = impl_witness_table (%Core.import_ref.eb9), @U.as_type.as.ImplicitAs.impl.167 [concrete]
// CHECK:STDOUT:   %Core.import_ref.017d8: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.774) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.5f9)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.2e9 = impl_witness_table (%Core.import_ref.017d8), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.var: ref %const.2f4 = var_storage %p.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %const.2f4 = call %G.ref()
// CHECK:STDOUT:   assign %p.var, %G.call
// CHECK:STDOUT:   %.loc10: type = splice_block %const [concrete = constants.%const.2f4] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ConstNullableParamS.ref: type = name_ref ConstNullableParamS, imports.%ConstNullableParamS.decl [concrete = constants.%ConstNullableParamS]
// CHECK:STDOUT:     %ptr: type = ptr_type %ConstNullableParamS.ref [concrete = constants.%ptr.13b]
// CHECK:STDOUT:     %const: type = const_type %ptr [concrete = constants.%const.2f4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %const.2f4 = wrapper_binding p, %p.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.170 = ref_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.170 = var_pattern %p.patt [concrete = constants.%p.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %const_nullable_foo.ref: %const_nullable_foo.cpp_overload_set.type = name_ref const_nullable_foo, imports.%const_nullable_foo.cpp_overload_set.value [concrete = constants.%const_nullable_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %p.ref: ref %const.2f4 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.a05 = impl_witness_access constants.%ImplicitAs.impl_witness.4fe, element0 [concrete = constants.%const.as.ImplicitAs.impl.Convert.7ef]
// CHECK:STDOUT:   %bound_method.loc11_26.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @const.as.ImplicitAs.impl.Convert(constants.%Optional.7c0, constants.%ImplicitAs.facet.464) [concrete = constants.%const.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_26.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %.loc11_26.1: %const.2f4 = acquire_value %p.ref
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.call: init %Optional.7c0 = call %bound_method.loc11_26.2(%.loc11_26.1)
// CHECK:STDOUT:   %.loc11_26.2: init %Optional.7c0 = converted %p.ref, %const.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_26.3: ref %Optional.7c0 = temporary_storage
// CHECK:STDOUT:   %.loc11_26.4: ref %Optional.7c0 = temporary %.loc11_26.3, %.loc11_26.2
// CHECK:STDOUT:   %.loc11_26.5: %Optional.7c0 = acquire_value %.loc11_26.4
// CHECK:STDOUT:   %const_nullable_foo.call: init %empty_tuple.type = call imports.%const_nullable_foo.decl(%.loc11_26.5)
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %.loc11_26.4, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%.loc11_26.4)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10: <bound method> = bound_method %p.var, constants.%Destroy.Op.1a2547.5
// CHECK:STDOUT:   %Destroy.Op.call.loc10: init %empty_tuple.type = call %Destroy.Op.bound.loc10(%p.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_26.1(%self.param: ref %.c5e2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_26.2(%self.param: ref %MaybeUnformed.c09) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_26.3(%self.param: ref %Optional.7c0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10(%self.param: ref %const.2f4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_const_nullable_pointer_param_using_non_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ConstNullableParamS: type = class_type @ConstNullableParamS [concrete]
// CHECK:STDOUT:   %ptr.13b: type = ptr_type %ConstNullableParamS [concrete]
// CHECK:STDOUT:   %pattern_type.550: type = pattern_type %ptr.13b [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.550 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:   %p.var_patt: %pattern_type.550 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   %const_nullable_foo.cpp_overload_set.type: type = cpp_overload_set_type @const_nullable_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const_nullable_foo.cpp_overload_set.value: %const_nullable_foo.cpp_overload_set.type = cpp_overload_set_value @const_nullable_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.781: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.e46: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.781) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.ddb: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.e46) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.71e: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.caa: %ptr.as.OptionalStorage.impl.Some.type.71e = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.1: <witness> = custom_witness (%Destroy.Op.1a2547.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.504: %Destroy.type = facet_value %ptr.13b, (%custom_witness.df9cc1.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.c09: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.504) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.cb6: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%ConstNullableParamS) [concrete]
// CHECK:STDOUT:   %.c5e2: type = maybe_unformed_type %ptr.13b [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.f04: %OptionalStorage.type = facet_value %ptr.13b, (%OptionalStorage.impl_witness.cb6) [concrete]
// CHECK:STDOUT:   %Optional.7c0: type = class_type @Optional, @Optional(%OptionalStorage.facet.f04) [concrete]
// CHECK:STDOUT:   %const_nullable_foo.type: type = fn_type @const_nullable_foo [concrete]
// CHECK:STDOUT:   %const_nullable_foo: %const_nullable_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3dc: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.7c0)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.3b8: type = facet_type <@OptionalAs, @OptionalAs(%T.220)> [symbolic]
// CHECK:STDOUT:   %U.042: %OptionalAs.type.3b8 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.ad5: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.167(%T.220, %U.042) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.331: %U.as_type.as.ImplicitAs.impl.Convert.type.ad5 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.cda: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.f04)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.774: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.220) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.5f9: %T.as_type.as.OptionalAs.impl.Convert.type.774 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.f8a: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.2e9, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.f04) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.cda = facet_value %ptr.13b, (%OptionalAs.impl_witness.f8a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.855: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bb5, @U.as_type.as.ImplicitAs.impl.167(%OptionalStorage.facet.f04, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.217: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.167(%OptionalStorage.facet.f04, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.a46: %U.as_type.as.ImplicitAs.impl.Convert.type.217 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.464: %ImplicitAs.type.3dc = facet_value %ptr.13b, (%ImplicitAs.impl_witness.855) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.34b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.7c0, %ImplicitAs.facet.464) [concrete]
// CHECK:STDOUT:   %.f82: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.34b, %ImplicitAs.facet.464 [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.as_type.as.ImplicitAs.impl.Convert.a46, @U.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.f04, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc11_26.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .ConstNullableParamS = %ConstNullableParamS.decl
// CHECK:STDOUT:     .const_nullable_foo = %const_nullable_foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstNullableParamS.decl: type = class_decl @ConstNullableParamS [concrete = constants.%ConstNullableParamS] {} {}
// CHECK:STDOUT:   %const_nullable_foo.cpp_overload_set.value: %const_nullable_foo.cpp_overload_set.type = cpp_overload_set_value @const_nullable_foo.cpp_overload_set [concrete = constants.%const_nullable_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.509: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.ddb)]
// CHECK:STDOUT:   %Core.import_ref.105 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.fa7: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.71e) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.caa)]
// CHECK:STDOUT:   %Core.import_ref.a90 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.306 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3e3 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.509, %Core.import_ref.105, %Core.import_ref.fa7, %Core.import_ref.a90, %Core.import_ref.306, %Core.import_ref.3e3), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %const_nullable_foo.decl: %const_nullable_foo.type = fn_decl @const_nullable_foo [concrete = constants.%const_nullable_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_27.1: type = splice_block %Optional [concrete = constants.%Optional.7c0] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.13b, (constants.%OptionalStorage.impl_witness.cb6) [concrete = constants.%OptionalStorage.facet.f04]
// CHECK:STDOUT:       %.loc11_27.2: %OptionalStorage.type = converted constants.%ptr.13b, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.f04]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.f04) [concrete = constants.%Optional.7c0]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.eb9: @U.as_type.as.ImplicitAs.impl.167.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.ad5) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.167.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.331)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bb5 = impl_witness_table (%Core.import_ref.eb9), @U.as_type.as.ImplicitAs.impl.167 [concrete]
// CHECK:STDOUT:   %Core.import_ref.017d8: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.774) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.5f9)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.2e9 = impl_witness_table (%Core.import_ref.017d8), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.var: ref %ptr.13b = var_storage %p.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %ptr.13b = call %G.ref()
// CHECK:STDOUT:   assign %p.var, %G.call
// CHECK:STDOUT:   %.loc10: type = splice_block %ptr [concrete = constants.%ptr.13b] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ConstNullableParamS.ref: type = name_ref ConstNullableParamS, imports.%ConstNullableParamS.decl [concrete = constants.%ConstNullableParamS]
// CHECK:STDOUT:     %ptr: type = ptr_type %ConstNullableParamS.ref [concrete = constants.%ptr.13b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.13b = wrapper_binding p, %p.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.550 = ref_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.550 = var_pattern %p.patt [concrete = constants.%p.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %const_nullable_foo.ref: %const_nullable_foo.cpp_overload_set.type = name_ref const_nullable_foo, imports.%const_nullable_foo.cpp_overload_set.value [concrete = constants.%const_nullable_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %p.ref: ref %ptr.13b = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.f82 = impl_witness_access constants.%ImplicitAs.impl_witness.855, element0 [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.a46]
// CHECK:STDOUT:   %bound_method.loc11_26.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @U.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.f04, constants.%OptionalAs.facet) [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_26.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %.loc11_26.1: %ptr.13b = acquire_value %p.ref
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.7c0 = call %bound_method.loc11_26.2(%.loc11_26.1)
// CHECK:STDOUT:   %.loc11_26.2: init %Optional.7c0 = converted %p.ref, %U.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_26.3: ref %Optional.7c0 = temporary_storage
// CHECK:STDOUT:   %.loc11_26.4: ref %Optional.7c0 = temporary %.loc11_26.3, %.loc11_26.2
// CHECK:STDOUT:   %.loc11_26.5: %Optional.7c0 = acquire_value %.loc11_26.4
// CHECK:STDOUT:   %const_nullable_foo.call: init %empty_tuple.type = call imports.%const_nullable_foo.decl(%.loc11_26.5)
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %.loc11_26.4, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%.loc11_26.4)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10: <bound method> = bound_method %p.var, constants.%Destroy.Op.1a2547.1
// CHECK:STDOUT:   %Destroy.Op.call.loc10: init %empty_tuple.type = call %Destroy.Op.bound.loc10(%p.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_26.1(%self.param: ref %.c5e2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_26.2(%self.param: ref %MaybeUnformed.c09) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_26.3(%self.param: ref %Optional.7c0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_nullable_pointer_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonNullableReturnS: type = class_type @NonNullableReturnS [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %NonNullableReturnS [concrete]
// CHECK:STDOUT:   %IngestDoublePointer.type: type = fn_type @IngestDoublePointer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IngestDoublePointer: %IngestDoublePointer.type = struct_value () [concrete]
// CHECK:STDOUT:   %non_nullable_return_foo.cpp_overload_set.type: type = cpp_overload_set_type @non_nullable_return_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %non_nullable_return_foo.cpp_overload_set.value: %non_nullable_return_foo.cpp_overload_set.type = cpp_overload_set_value @non_nullable_return_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %non_nullable_return_foo.type: type = fn_type @non_nullable_return_foo [concrete]
// CHECK:STDOUT:   %non_nullable_return_foo: %non_nullable_return_foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NonNullableReturnS = %NonNullableReturnS.decl
// CHECK:STDOUT:     .non_nullable_return_foo = %non_nullable_return_foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonNullableReturnS.decl: type = class_decl @NonNullableReturnS [concrete = constants.%NonNullableReturnS] {} {}
// CHECK:STDOUT:   %non_nullable_return_foo.cpp_overload_set.value: %non_nullable_return_foo.cpp_overload_set.type = cpp_overload_set_value @non_nullable_return_foo.cpp_overload_set [concrete = constants.%non_nullable_return_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %non_nullable_return_foo.decl: %non_nullable_return_foo.type = fn_decl @non_nullable_return_foo [concrete = constants.%non_nullable_return_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IngestDoublePointer.ref: %IngestDoublePointer.type = name_ref IngestDoublePointer, file.%IngestDoublePointer.decl [concrete = constants.%IngestDoublePointer]
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %non_nullable_return_foo.ref.loc10: %non_nullable_return_foo.cpp_overload_set.type = name_ref non_nullable_return_foo, imports.%non_nullable_return_foo.cpp_overload_set.value [concrete = constants.%non_nullable_return_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %non_nullable_return_foo.call.loc10: init %ptr = call imports.%non_nullable_return_foo.decl()
// CHECK:STDOUT:   %.loc10_51.1: %ptr = value_of_initializer %non_nullable_return_foo.call.loc10
// CHECK:STDOUT:   %.loc10_51.2: %ptr = converted %non_nullable_return_foo.call.loc10, %.loc10_51.1
// CHECK:STDOUT:   %IngestDoublePointer.call: init %empty_tuple.type = call %IngestDoublePointer.ref(%.loc10_51.2)
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %non_nullable_return_foo.ref.loc11: %non_nullable_return_foo.cpp_overload_set.type = name_ref non_nullable_return_foo, imports.%non_nullable_return_foo.cpp_overload_set.value [concrete = constants.%non_nullable_return_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %non_nullable_return_foo.call.loc11: init %ptr = call imports.%non_nullable_return_foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_double_pointer_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DoubleNonNullableReturnS: type = class_type @DoubleNonNullableReturnS [concrete]
// CHECK:STDOUT:   %ptr.33a: type = ptr_type %DoubleNonNullableReturnS [concrete]
// CHECK:STDOUT:   %ptr.297: type = ptr_type %ptr.33a [concrete]
// CHECK:STDOUT:   %IngestDoublePointer.type: type = fn_type @IngestDoublePointer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IngestDoublePointer: %IngestDoublePointer.type = struct_value () [concrete]
// CHECK:STDOUT:   %double_non_nullable_return_foo.cpp_overload_set.type: type = cpp_overload_set_type @double_non_nullable_return_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %double_non_nullable_return_foo.cpp_overload_set.value: %double_non_nullable_return_foo.cpp_overload_set.type = cpp_overload_set_value @double_non_nullable_return_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %double_non_nullable_return_foo.type: type = fn_type @double_non_nullable_return_foo [concrete]
// CHECK:STDOUT:   %double_non_nullable_return_foo: %double_non_nullable_return_foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DoubleNonNullableReturnS = %DoubleNonNullableReturnS.decl
// CHECK:STDOUT:     .double_non_nullable_return_foo = %double_non_nullable_return_foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DoubleNonNullableReturnS.decl: type = class_decl @DoubleNonNullableReturnS [concrete = constants.%DoubleNonNullableReturnS] {} {}
// CHECK:STDOUT:   %double_non_nullable_return_foo.cpp_overload_set.value: %double_non_nullable_return_foo.cpp_overload_set.type = cpp_overload_set_value @double_non_nullable_return_foo.cpp_overload_set [concrete = constants.%double_non_nullable_return_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %double_non_nullable_return_foo.decl: %double_non_nullable_return_foo.type = fn_decl @double_non_nullable_return_foo [concrete = constants.%double_non_nullable_return_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IngestDoublePointer.ref: %IngestDoublePointer.type = name_ref IngestDoublePointer, file.%IngestDoublePointer.decl [concrete = constants.%IngestDoublePointer]
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %double_non_nullable_return_foo.ref.loc10: %double_non_nullable_return_foo.cpp_overload_set.type = name_ref double_non_nullable_return_foo, imports.%double_non_nullable_return_foo.cpp_overload_set.value [concrete = constants.%double_non_nullable_return_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %double_non_nullable_return_foo.call.loc10: init %ptr.297 = call imports.%double_non_nullable_return_foo.decl()
// CHECK:STDOUT:   %.loc10_58.1: %ptr.297 = value_of_initializer %double_non_nullable_return_foo.call.loc10
// CHECK:STDOUT:   %.loc10_58.2: %ptr.297 = converted %double_non_nullable_return_foo.call.loc10, %.loc10_58.1
// CHECK:STDOUT:   %IngestDoublePointer.call: init %empty_tuple.type = call %IngestDoublePointer.ref(%.loc10_58.2)
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %double_non_nullable_return_foo.ref.loc11: %double_non_nullable_return_foo.cpp_overload_set.type = name_ref double_non_nullable_return_foo, imports.%double_non_nullable_return_foo.cpp_overload_set.value [concrete = constants.%double_non_nullable_return_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %double_non_nullable_return_foo.call.loc11: init %ptr.297 = call imports.%double_non_nullable_return_foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_const_non_nullable_pointer_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ConstNonNullableReturnS: type = class_type @ConstNonNullableReturnS [concrete]
// CHECK:STDOUT:   %const: type = const_type %ConstNonNullableReturnS [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %IngestConstPointer.type: type = fn_type @IngestConstPointer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IngestConstPointer: %IngestConstPointer.type = struct_value () [concrete]
// CHECK:STDOUT:   %const_non_nullable_return_foo.cpp_overload_set.type: type = cpp_overload_set_type @const_non_nullable_return_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const_non_nullable_return_foo.cpp_overload_set.value: %const_non_nullable_return_foo.cpp_overload_set.type = cpp_overload_set_value @const_non_nullable_return_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const_non_nullable_return_foo.type: type = fn_type @const_non_nullable_return_foo [concrete]
// CHECK:STDOUT:   %const_non_nullable_return_foo: %const_non_nullable_return_foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .ConstNonNullableReturnS = %ConstNonNullableReturnS.decl
// CHECK:STDOUT:     .const_non_nullable_return_foo = %const_non_nullable_return_foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstNonNullableReturnS.decl: type = class_decl @ConstNonNullableReturnS [concrete = constants.%ConstNonNullableReturnS] {} {}
// CHECK:STDOUT:   %const_non_nullable_return_foo.cpp_overload_set.value: %const_non_nullable_return_foo.cpp_overload_set.type = cpp_overload_set_value @const_non_nullable_return_foo.cpp_overload_set [concrete = constants.%const_non_nullable_return_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %const_non_nullable_return_foo.decl: %const_non_nullable_return_foo.type = fn_decl @const_non_nullable_return_foo [concrete = constants.%const_non_nullable_return_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IngestConstPointer.ref: %IngestConstPointer.type = name_ref IngestConstPointer, file.%IngestConstPointer.decl [concrete = constants.%IngestConstPointer]
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %const_non_nullable_return_foo.ref.loc10: %const_non_nullable_return_foo.cpp_overload_set.type = name_ref const_non_nullable_return_foo, imports.%const_non_nullable_return_foo.cpp_overload_set.value [concrete = constants.%const_non_nullable_return_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %const_non_nullable_return_foo.call.loc10: init %ptr = call imports.%const_non_nullable_return_foo.decl()
// CHECK:STDOUT:   %.loc10_56.1: %ptr = value_of_initializer %const_non_nullable_return_foo.call.loc10
// CHECK:STDOUT:   %.loc10_56.2: %ptr = converted %const_non_nullable_return_foo.call.loc10, %.loc10_56.1
// CHECK:STDOUT:   %IngestConstPointer.call: init %empty_tuple.type = call %IngestConstPointer.ref(%.loc10_56.2)
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %const_non_nullable_return_foo.ref.loc11: %const_non_nullable_return_foo.cpp_overload_set.type = name_ref const_non_nullable_return_foo, imports.%const_non_nullable_return_foo.cpp_overload_set.value [concrete = constants.%const_non_nullable_return_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %const_non_nullable_return_foo.call.loc11: init %ptr = call imports.%const_non_nullable_return_foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_nullable_pointer_arg_to_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NullableParamS: type = class_type @NullableParamS [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.c3f: type = pattern_type %NullableParamS [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.c3f = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.c3f = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %NullableParamS.val: %NullableParamS = struct_value () [concrete]
// CHECK:STDOUT:   %nullable_param_foo.cpp_overload_set.type: type = cpp_overload_set_type @nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %nullable_param_foo.cpp_overload_set.value: %nullable_param_foo.cpp_overload_set.type = cpp_overload_set_value @nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.1f0: type = ptr_type %NullableParamS [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.781: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.e46: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.781) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.ddb: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.e46) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.71e: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.caa: %ptr.as.OptionalStorage.impl.Some.type.71e = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.1: <witness> = custom_witness (%Destroy.Op.1a2547.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.8f0: %Destroy.type = facet_value %ptr.1f0, (%custom_witness.df9cc1.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.2d7: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.8f0) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.0df: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%NullableParamS) [concrete]
// CHECK:STDOUT:   %.280: type = maybe_unformed_type %ptr.1f0 [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.8bf: %OptionalStorage.type = facet_value %ptr.1f0, (%OptionalStorage.impl_witness.0df) [concrete]
// CHECK:STDOUT:   %Optional.e08: type = class_type @Optional, @Optional(%OptionalStorage.facet.8bf) [concrete]
// CHECK:STDOUT:   %nullable_param_foo.type: type = fn_type @nullable_param_foo [concrete]
// CHECK:STDOUT:   %nullable_param_foo: %nullable_param_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.f6b: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.e08)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.3b8d: type = facet_type <@OptionalAs, @OptionalAs(%T.220)> [symbolic]
// CHECK:STDOUT:   %U.042: %OptionalAs.type.3b8d = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.ad5: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.167(%T.220, %U.042) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.331: %U.as_type.as.ImplicitAs.impl.Convert.type.ad5 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.3b80: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.8bf)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.774: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.220) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.5f9: %T.as_type.as.OptionalAs.impl.Convert.type.774 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.f31: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.2e9, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.8bf) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.3b80 = facet_value %ptr.1f0, (%OptionalAs.impl_witness.f31) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.7a9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bb5, @U.as_type.as.ImplicitAs.impl.167(%OptionalStorage.facet.8bf, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.27b: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.167(%OptionalStorage.facet.8bf, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.2e2: %U.as_type.as.ImplicitAs.impl.Convert.type.27b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2d4: %ImplicitAs.type.f6b = facet_value %ptr.1f0, (%ImplicitAs.impl_witness.7a9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.767: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.e08, %ImplicitAs.facet.2d4) [concrete]
// CHECK:STDOUT:   %.642: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.767, %ImplicitAs.facet.2d4 [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.as_type.as.ImplicitAs.impl.Convert.2e2, @U.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.8bf, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc9_26.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT:   %NullableParamS.cpp_destructor.type: type = fn_type @NullableParamS.cpp_destructor [concrete]
// CHECK:STDOUT:   %NullableParamS.cpp_destructor: %NullableParamS.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NullableParamS = %NullableParamS.decl
// CHECK:STDOUT:     .nullable_param_foo = %nullable_param_foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NullableParamS.decl: type = class_decl @NullableParamS [concrete = constants.%NullableParamS] {} {}
// CHECK:STDOUT:   %nullable_param_foo.cpp_overload_set.value: %nullable_param_foo.cpp_overload_set.type = cpp_overload_set_value @nullable_param_foo.cpp_overload_set [concrete = constants.%nullable_param_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.509: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.ddb)]
// CHECK:STDOUT:   %Core.import_ref.105 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.fa7: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.71e) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.caa)]
// CHECK:STDOUT:   %Core.import_ref.a90 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.306 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3e3 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.509, %Core.import_ref.105, %Core.import_ref.fa7, %Core.import_ref.a90, %Core.import_ref.306, %Core.import_ref.3e3), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %nullable_param_foo.decl: %nullable_param_foo.type = fn_decl @nullable_param_foo [concrete = constants.%nullable_param_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc9_28.1: type = splice_block %Optional [concrete = constants.%Optional.e08] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.1f0, (constants.%OptionalStorage.impl_witness.0df) [concrete = constants.%OptionalStorage.facet.8bf]
// CHECK:STDOUT:       %.loc9_28.2: %OptionalStorage.type = converted constants.%ptr.1f0, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.8bf]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.8bf) [concrete = constants.%Optional.e08]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.eb9: @U.as_type.as.ImplicitAs.impl.167.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.ad5) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.167.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.331)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bb5 = impl_witness_table (%Core.import_ref.eb9), @U.as_type.as.ImplicitAs.impl.167 [concrete]
// CHECK:STDOUT:   %Core.import_ref.017d8: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.774) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.5f9)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.2e9 = impl_witness_table (%Core.import_ref.017d8), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref %NullableParamS = var_storage %s.var_patt
// CHECK:STDOUT:   %.loc8_32.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_32.2: init %NullableParamS to %s.var = class_init () [concrete = constants.%NullableParamS.val]
// CHECK:STDOUT:   %.loc8_3: init %NullableParamS = converted %.loc8_32.1, %.loc8_32.2 [concrete = constants.%NullableParamS.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %NullableParamS.ref [concrete = constants.%NullableParamS] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NullableParamS.ref: type = name_ref NullableParamS, imports.%NullableParamS.decl [concrete = constants.%NullableParamS]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %NullableParamS = wrapper_binding s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.c3f = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.c3f = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %nullable_param_foo.ref: %nullable_param_foo.cpp_overload_set.type = name_ref nullable_param_foo, imports.%nullable_param_foo.cpp_overload_set.value [concrete = constants.%nullable_param_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %NullableParamS = name_ref s, %s
// CHECK:STDOUT:   %addr: %ptr.1f0 = addr_of %s.ref
// CHECK:STDOUT:   %impl.elem0: %.642 = impl_witness_access constants.%ImplicitAs.impl_witness.7a9, element0 [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.2e2]
// CHECK:STDOUT:   %bound_method.loc9_26.1: <bound method> = bound_method %addr, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @U.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.8bf, constants.%OptionalAs.facet) [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_26.2: <bound method> = bound_method %addr, %specific_fn
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.e08 = call %bound_method.loc9_26.2(%addr)
// CHECK:STDOUT:   %.loc9_26.1: init %Optional.e08 = converted %addr, %U.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc9_26.2: ref %Optional.e08 = temporary_storage
// CHECK:STDOUT:   %.loc9_26.3: ref %Optional.e08 = temporary %.loc9_26.2, %.loc9_26.1
// CHECK:STDOUT:   %.loc9_26.4: %Optional.e08 = acquire_value %.loc9_26.3
// CHECK:STDOUT:   %nullable_param_foo.call: init %empty_tuple.type = call imports.%nullable_param_foo.decl(%.loc9_26.4)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc9_26.3, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc9_26.3)
// CHECK:STDOUT:   %NullableParamS.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%NullableParamS.cpp_destructor
// CHECK:STDOUT:   %NullableParamS.cpp_destructor.call: init %empty_tuple.type = call %NullableParamS.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_26.1(%self.param: ref %.280) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_26.2(%self.param: ref %MaybeUnformed.2d7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_26.3(%self.param: ref %Optional.e08) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- null_pointer_arg_to_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %nullable_param_foo.cpp_overload_set.type: type = cpp_overload_set_type @nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %nullable_param_foo.cpp_overload_set.value: %nullable_param_foo.cpp_overload_set.type = cpp_overload_set_value @nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Optional.type: type = generic_class_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.None.type.bdc: type = fn_type @Optional.None, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.None.01e: %Optional.None.type.bdc = struct_value () [symbolic]
// CHECK:STDOUT:   %NullableParamS: type = class_type @NullableParamS [concrete]
// CHECK:STDOUT:   %ptr.1f0: type = ptr_type %NullableParamS [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.781: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.e46: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.781) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.ddb: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.e46) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.type.2a7: type = fn_type @ptr.as.OptionalStorage.impl.None, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.c23: %ptr.as.OptionalStorage.impl.None.type.2a7 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.1: <witness> = custom_witness (%Destroy.Op.1a2547.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.8f0: %Destroy.type = facet_value %ptr.1f0, (%custom_witness.df9cc1.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.2d7: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.8f0) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.0df: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%NullableParamS) [concrete]
// CHECK:STDOUT:   %.280: type = maybe_unformed_type %ptr.1f0 [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.1f0, (%OptionalStorage.impl_witness.0df) [concrete]
// CHECK:STDOUT:   %Optional.e08: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.type.12d: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.82d: %Optional.None.type.12d = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %Optional.None.82d, @Optional.None(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %nullable_param_foo.type: type = fn_type @nullable_param_foo [concrete]
// CHECK:STDOUT:   %nullable_param_foo: %nullable_param_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc8_66.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .nullable_param_foo = %nullable_param_foo.cpp_overload_set.value
// CHECK:STDOUT:     .NullableParamS = %NullableParamS.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nullable_param_foo.cpp_overload_set.value: %nullable_param_foo.cpp_overload_set.type = cpp_overload_set_value @nullable_param_foo.cpp_overload_set [concrete = constants.%nullable_param_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.import_ref.e9f: @Optional.%Optional.None.type (%Optional.None.type.bdc) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @Optional.%Optional.None (constants.%Optional.None.01e)]
// CHECK:STDOUT:   %NullableParamS.decl: type = class_decl @NullableParamS [concrete = constants.%NullableParamS] {} {}
// CHECK:STDOUT:   %Core.import_ref.509: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.ddb)]
// CHECK:STDOUT:   %Core.import_ref.307: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None.type (%ptr.as.OptionalStorage.impl.None.type.2a7) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None (constants.%ptr.as.OptionalStorage.impl.None.c23)]
// CHECK:STDOUT:   %Core.import_ref.c9a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.a90 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.306 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3e3 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.509, %Core.import_ref.307, %Core.import_ref.c9a, %Core.import_ref.a90, %Core.import_ref.306, %Core.import_ref.3e3), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %nullable_param_foo.decl: %nullable_param_foo.type = fn_decl @nullable_param_foo [concrete = constants.%nullable_param_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc8_67.1: type = splice_block %Optional [concrete = constants.%Optional.e08] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.1f0, (constants.%OptionalStorage.impl_witness.0df) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %.loc8_67.2: %OptionalStorage.type = converted constants.%ptr.1f0, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.e08]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %nullable_param_foo.ref: %nullable_param_foo.cpp_overload_set.type = name_ref nullable_param_foo, imports.%nullable_param_foo.cpp_overload_set.value [concrete = constants.%nullable_param_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Optional.ref: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Cpp.ref.loc8_40: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %NullableParamS.ref: type = name_ref NullableParamS, imports.%NullableParamS.decl [concrete = constants.%NullableParamS]
// CHECK:STDOUT:   %ptr: type = ptr_type %NullableParamS.ref [concrete = constants.%ptr.1f0]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.0df) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc8_59: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.e08]
// CHECK:STDOUT:   %.loc8_60: %Optional.None.type.12d = specific_constant imports.%Core.import_ref.e9f, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.82d]
// CHECK:STDOUT:   %None.ref: %Optional.None.type.12d = name_ref None, %.loc8_60 [concrete = constants.%Optional.None.82d]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %None.ref, @Optional.None(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.specific_fn]
// CHECK:STDOUT:   %Optional.None.call: init %Optional.e08 = call %Optional.None.specific_fn()
// CHECK:STDOUT:   %.loc8_66.1: ref %Optional.e08 = temporary_storage
// CHECK:STDOUT:   %.loc8_66.2: ref %Optional.e08 = temporary %.loc8_66.1, %Optional.None.call
// CHECK:STDOUT:   %.loc8_66.3: %Optional.e08 = acquire_value %.loc8_66.2
// CHECK:STDOUT:   %nullable_param_foo.call: init %empty_tuple.type = call imports.%nullable_param_foo.decl(%.loc8_66.3)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc8_66.2, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc8_66.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_66.1(%self.param: ref %.280) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_66.2(%self.param: ref %MaybeUnformed.2d7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_66.3(%self.param: ref %Optional.e08) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nonnull_pointer_arg_to_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NullableParamS: type = class_type @NullableParamS [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.c3f: type = pattern_type %NullableParamS [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.c3f = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.c3f = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %NullableParamS.val: %NullableParamS = struct_value () [concrete]
// CHECK:STDOUT:   %nullable_param_foo.cpp_overload_set.type: type = cpp_overload_set_type @nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %nullable_param_foo.cpp_overload_set.value: %nullable_param_foo.cpp_overload_set.type = cpp_overload_set_value @nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Optional.type: type = generic_class_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.304: type = fn_type @Optional.Some, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Some.2a0: %Optional.Some.type.304 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.1f0: type = ptr_type %NullableParamS [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.781: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.e46: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.781) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.ddb: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.e46) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.71e: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.caa: %ptr.as.OptionalStorage.impl.Some.type.71e = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.1: <witness> = custom_witness (%Destroy.Op.1a2547.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.8f0: %Destroy.type = facet_value %ptr.1f0, (%custom_witness.df9cc1.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.2d7: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.8f0) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.0df: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%NullableParamS) [concrete]
// CHECK:STDOUT:   %.280: type = maybe_unformed_type %ptr.1f0 [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.1f0, (%OptionalStorage.impl_witness.0df) [concrete]
// CHECK:STDOUT:   %Optional.e08: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.Some.type.129: type = fn_type @Optional.Some, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.Some.399: %Optional.Some.type.129 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Some.specific_fn: <specific function> = specific_function %Optional.Some.399, @Optional.Some(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %nullable_param_foo.type: type = fn_type @nullable_param_foo [concrete]
// CHECK:STDOUT:   %nullable_param_foo: %nullable_param_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc9_68.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT:   %NullableParamS.cpp_destructor.type: type = fn_type @NullableParamS.cpp_destructor [concrete]
// CHECK:STDOUT:   %NullableParamS.cpp_destructor: %NullableParamS.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NullableParamS = %NullableParamS.decl
// CHECK:STDOUT:     .nullable_param_foo = %nullable_param_foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NullableParamS.decl: type = class_decl @NullableParamS [concrete = constants.%NullableParamS] {} {}
// CHECK:STDOUT:   %nullable_param_foo.cpp_overload_set.value: %nullable_param_foo.cpp_overload_set.type = cpp_overload_set_value @nullable_param_foo.cpp_overload_set [concrete = constants.%nullable_param_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.import_ref.48e: @Optional.%Optional.Some.type (%Optional.Some.type.304) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @Optional.%Optional.Some (constants.%Optional.Some.2a0)]
// CHECK:STDOUT:   %Core.import_ref.509: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.ddb)]
// CHECK:STDOUT:   %Core.import_ref.105 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.fa7: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.71e) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.caa)]
// CHECK:STDOUT:   %Core.import_ref.a90 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.306 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3e3 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.509, %Core.import_ref.105, %Core.import_ref.fa7, %Core.import_ref.a90, %Core.import_ref.306, %Core.import_ref.3e3), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %nullable_param_foo.decl: %nullable_param_foo.type = fn_decl @nullable_param_foo [concrete = constants.%nullable_param_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc9_69.1: type = splice_block %Optional [concrete = constants.%Optional.e08] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.1f0, (constants.%OptionalStorage.impl_witness.0df) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %.loc9_69.2: %OptionalStorage.type = converted constants.%ptr.1f0, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.e08]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref %NullableParamS = var_storage %s.var_patt
// CHECK:STDOUT:   %.loc8_32.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_32.2: init %NullableParamS to %s.var = class_init () [concrete = constants.%NullableParamS.val]
// CHECK:STDOUT:   %.loc8_3: init %NullableParamS = converted %.loc8_32.1, %.loc8_32.2 [concrete = constants.%NullableParamS.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %NullableParamS.ref.loc8 [concrete = constants.%NullableParamS] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NullableParamS.ref.loc8: type = name_ref NullableParamS, imports.%NullableParamS.decl [concrete = constants.%NullableParamS]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %NullableParamS = wrapper_binding s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.c3f = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.c3f = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %nullable_param_foo.ref: %nullable_param_foo.cpp_overload_set.type = name_ref nullable_param_foo, imports.%nullable_param_foo.cpp_overload_set.value [concrete = constants.%nullable_param_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Optional.ref: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Cpp.ref.loc9_40: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %NullableParamS.ref.loc9: type = name_ref NullableParamS, imports.%NullableParamS.decl [concrete = constants.%NullableParamS]
// CHECK:STDOUT:   %ptr: type = ptr_type %NullableParamS.ref.loc9 [concrete = constants.%ptr.1f0]
// CHECK:STDOUT:   %OptionalStorage.facet.loc9_59: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.0df) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc9_59: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet.loc9_59 [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.e08]
// CHECK:STDOUT:   %.loc9_60: %Optional.Some.type.129 = specific_constant imports.%Core.import_ref.48e, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.Some.399]
// CHECK:STDOUT:   %Some.ref: %Optional.Some.type.129 = name_ref Some, %.loc9_60 [concrete = constants.%Optional.Some.399]
// CHECK:STDOUT:   %s.ref: ref %NullableParamS = name_ref s, %s
// CHECK:STDOUT:   %addr: %ptr.1f0 = addr_of %s.ref
// CHECK:STDOUT:   %OptionalStorage.facet.loc9_68.1: %OptionalStorage.type = facet_value constants.%ptr.1f0, (constants.%OptionalStorage.impl_witness.0df) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc9_68.1: %OptionalStorage.type = converted constants.%ptr.1f0, %OptionalStorage.facet.loc9_68.1 [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %OptionalStorage.facet.loc9_68.2: %OptionalStorage.type = facet_value constants.%ptr.1f0, (constants.%OptionalStorage.impl_witness.0df) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc9_68.2: %OptionalStorage.type = converted constants.%ptr.1f0, %OptionalStorage.facet.loc9_68.2 [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %Optional.Some.specific_fn: <specific function> = specific_function %Some.ref, @Optional.Some(constants.%OptionalStorage.facet) [concrete = constants.%Optional.Some.specific_fn]
// CHECK:STDOUT:   %Optional.Some.call: init %Optional.e08 = call %Optional.Some.specific_fn(%addr)
// CHECK:STDOUT:   %.loc9_68.3: ref %Optional.e08 = temporary_storage
// CHECK:STDOUT:   %.loc9_68.4: ref %Optional.e08 = temporary %.loc9_68.3, %Optional.Some.call
// CHECK:STDOUT:   %.loc9_68.5: %Optional.e08 = acquire_value %.loc9_68.4
// CHECK:STDOUT:   %nullable_param_foo.call: init %empty_tuple.type = call imports.%nullable_param_foo.decl(%.loc9_68.5)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc9_68.4, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc9_68.4)
// CHECK:STDOUT:   %NullableParamS.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%NullableParamS.cpp_destructor
// CHECK:STDOUT:   %NullableParamS.cpp_destructor.call: init %empty_tuple.type = call %NullableParamS.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_68.1(%self.param: ref %.280) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_68.2(%self.param: ref %MaybeUnformed.2d7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_68.3(%self.param: ref %Optional.e08) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- forward_nullable_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %nullable_param_foo.cpp_overload_set.type: type = cpp_overload_set_type @nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %nullable_param_foo.cpp_overload_set.value: %nullable_param_foo.cpp_overload_set.type = cpp_overload_set_value @nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %nullable_param_get.cpp_overload_set.type: type = cpp_overload_set_type @nullable_param_get.cpp_overload_set [concrete]
// CHECK:STDOUT:   %nullable_param_get.cpp_overload_set.value: %nullable_param_get.cpp_overload_set.type = cpp_overload_set_value @nullable_param_get.cpp_overload_set [concrete]
// CHECK:STDOUT:   %NullableParamS: type = class_type @NullableParamS [concrete]
// CHECK:STDOUT:   %ptr.1f0: type = ptr_type %NullableParamS [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.781: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.e46: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.781) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.ddb: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.e46) [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.1: <witness> = custom_witness (%Destroy.Op.1a2547.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.8f0: %Destroy.type = facet_value %ptr.1f0, (%custom_witness.df9cc1.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.2d7: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.8f0) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.0df: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%NullableParamS) [concrete]
// CHECK:STDOUT:   %.280: type = maybe_unformed_type %ptr.1f0 [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.1f0, (%OptionalStorage.impl_witness.0df) [concrete]
// CHECK:STDOUT:   %Optional.e08: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %nullable_param_get.type: type = fn_type @nullable_param_get [concrete]
// CHECK:STDOUT:   %nullable_param_get: %nullable_param_get.type = struct_value () [concrete]
// CHECK:STDOUT:   %nullable_param_foo.type: type = fn_type @nullable_param_foo [concrete]
// CHECK:STDOUT:   %nullable_param_foo: %nullable_param_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc8_49.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .nullable_param_foo = %nullable_param_foo.cpp_overload_set.value
// CHECK:STDOUT:     .nullable_param_get = %nullable_param_get.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nullable_param_foo.cpp_overload_set.value: %nullable_param_foo.cpp_overload_set.type = cpp_overload_set_value @nullable_param_foo.cpp_overload_set [concrete = constants.%nullable_param_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %nullable_param_get.cpp_overload_set.value: %nullable_param_get.cpp_overload_set.type = cpp_overload_set_value @nullable_param_get.cpp_overload_set [concrete = constants.%nullable_param_get.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.509: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.ddb)]
// CHECK:STDOUT:   %Core.import_ref.105 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.c9a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.a90 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.306 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3e3 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.509, %Core.import_ref.105, %Core.import_ref.c9a, %Core.import_ref.a90, %Core.import_ref.306, %Core.import_ref.3e3), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %nullable_param_get.decl: %nullable_param_get.type = fn_decl @nullable_param_get [concrete = constants.%nullable_param_get] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.1f0, (constants.%OptionalStorage.impl_witness.0df) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %.loc8: %OptionalStorage.type = converted constants.%ptr.1f0, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.e08]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nullable_param_foo.decl: %nullable_param_foo.type = fn_decl @nullable_param_foo [concrete = constants.%nullable_param_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc8_50.1: type = splice_block %Optional [concrete = constants.%Optional.e08] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.1f0, (constants.%OptionalStorage.impl_witness.0df) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %.loc8_50.2: %OptionalStorage.type = converted constants.%ptr.1f0, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.e08]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %nullable_param_foo.ref: %nullable_param_foo.cpp_overload_set.type = name_ref nullable_param_foo, imports.%nullable_param_foo.cpp_overload_set.value [concrete = constants.%nullable_param_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc8_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %nullable_param_get.ref: %nullable_param_get.cpp_overload_set.type = name_ref nullable_param_get, imports.%nullable_param_get.cpp_overload_set.value [concrete = constants.%nullable_param_get.cpp_overload_set.value]
// CHECK:STDOUT:   %nullable_param_get.call: init %Optional.e08 = call imports.%nullable_param_get.decl()
// CHECK:STDOUT:   %.loc8_49.1: ref %Optional.e08 = temporary_storage
// CHECK:STDOUT:   %.loc8_49.2: ref %Optional.e08 = temporary %.loc8_49.1, %nullable_param_get.call
// CHECK:STDOUT:   %.loc8_49.3: %Optional.e08 = acquire_value %.loc8_49.2
// CHECK:STDOUT:   %nullable_param_foo.call: init %empty_tuple.type = call imports.%nullable_param_foo.decl(%.loc8_49.3)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc8_49.2, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc8_49.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_49.1(%self.param: ref %.280) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_49.2(%self.param: ref %MaybeUnformed.2d7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_49.3(%self.param: ref %Optional.e08) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_deduced_any_param_as_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DeducedAnyS: type = class_type @DeducedAnyS [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.1ad: type = pattern_type %DeducedAnyS [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.1ad = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.1ad = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %DeducedAnyS.val: %DeducedAnyS = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.7e6: type = ptr_type %DeducedAnyS [concrete]
// CHECK:STDOUT:   %pattern_type.060: type = pattern_type %ptr.7e6 [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.060 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   %deduced_any_foo.cpp_overload_set.type: type = cpp_overload_set_type @deduced_any_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %deduced_any_foo.cpp_overload_set.value: %deduced_any_foo.cpp_overload_set.type = cpp_overload_set_value @deduced_any_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %deduced_any_foo.type: type = fn_type @deduced_any_foo [concrete]
// CHECK:STDOUT:   %deduced_any_foo: %deduced_any_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %DeducedAnyS.cpp_destructor.type: type = fn_type @DeducedAnyS.cpp_destructor [concrete]
// CHECK:STDOUT:   %DeducedAnyS.cpp_destructor: %DeducedAnyS.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DeducedAnyS = %DeducedAnyS.decl
// CHECK:STDOUT:     .deduced_any_foo = %deduced_any_foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeducedAnyS.decl: type = class_decl @DeducedAnyS [concrete = constants.%DeducedAnyS] {} {}
// CHECK:STDOUT:   %deduced_any_foo.cpp_overload_set.value: %deduced_any_foo.cpp_overload_set.type = cpp_overload_set_value @deduced_any_foo.cpp_overload_set [concrete = constants.%deduced_any_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %deduced_any_foo.decl: %deduced_any_foo.type = fn_decl @deduced_any_foo [concrete = constants.%deduced_any_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref %DeducedAnyS = var_storage %s.var_patt
// CHECK:STDOUT:   %.loc8_29.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_29.2: init %DeducedAnyS to %s.var = class_init () [concrete = constants.%DeducedAnyS.val]
// CHECK:STDOUT:   %.loc8_3: init %DeducedAnyS = converted %.loc8_29.1, %.loc8_29.2 [concrete = constants.%DeducedAnyS.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %DeducedAnyS.ref.loc8 [concrete = constants.%DeducedAnyS] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DeducedAnyS.ref.loc8: type = name_ref DeducedAnyS, imports.%DeducedAnyS.decl [concrete = constants.%DeducedAnyS]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %DeducedAnyS = wrapper_binding s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.1ad = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.1ad = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13_36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %deduced_any_foo.ref: %deduced_any_foo.cpp_overload_set.type = name_ref deduced_any_foo, imports.%deduced_any_foo.cpp_overload_set.value [concrete = constants.%deduced_any_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %DeducedAnyS = name_ref s, %s
// CHECK:STDOUT:   %addr: %ptr.7e6 = addr_of %s.ref
// CHECK:STDOUT:   %deduced_any_foo.call: init %ptr.7e6 = call imports.%deduced_any_foo.decl(%addr)
// CHECK:STDOUT:   %.loc13_58.1: %ptr.7e6 = value_of_initializer %deduced_any_foo.call
// CHECK:STDOUT:   %.loc13_58.2: %ptr.7e6 = converted %deduced_any_foo.call, %.loc13_58.1
// CHECK:STDOUT:   %.loc13_32: type = splice_block %ptr [concrete = constants.%ptr.7e6] {
// CHECK:STDOUT:     %Cpp.ref.loc13_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DeducedAnyS.ref.loc13: type = name_ref DeducedAnyS, imports.%DeducedAnyS.decl [concrete = constants.%DeducedAnyS]
// CHECK:STDOUT:     %ptr: type = ptr_type %DeducedAnyS.ref.loc13 [concrete = constants.%ptr.7e6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.7e6 = wrapper_binding p, %.loc13_58.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.060 = value_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeducedAnyS.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%DeducedAnyS.cpp_destructor
// CHECK:STDOUT:   %DeducedAnyS.cpp_destructor.call: init %empty_tuple.type = call %DeducedAnyS.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_deduced_pointer_param_as_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DeducedPointerS: type = class_type @DeducedPointerS [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.36d: type = pattern_type %DeducedPointerS [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.36d = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.36d = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %DeducedPointerS.val: %DeducedPointerS = struct_value () [concrete]
// CHECK:STDOUT:   %Direct.cpp_overload_set.type: type = cpp_overload_set_type @Direct.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Direct.cpp_overload_set.value: %Direct.cpp_overload_set.type = cpp_overload_set_value @Direct.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.6cb: type = ptr_type %DeducedPointerS [concrete]
// CHECK:STDOUT:   %Optional.type: type = generic_class_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.781: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.e46: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.781) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.ddb: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.e46) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.71e: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.caa: %ptr.as.OptionalStorage.impl.Some.type.71e = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.1: <witness> = custom_witness (%Destroy.Op.1a2547.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.25b: %Destroy.type = facet_value %ptr.6cb, (%custom_witness.df9cc1.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.dcc: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.25b) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.5d7: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%DeducedPointerS) [concrete]
// CHECK:STDOUT:   %.fb1: type = maybe_unformed_type %ptr.6cb [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.822: %OptionalStorage.type = facet_value %ptr.6cb, (%OptionalStorage.impl_witness.5d7) [concrete]
// CHECK:STDOUT:   %Optional.1ac: type = class_type @Optional, @Optional(%OptionalStorage.facet.822) [concrete]
// CHECK:STDOUT:   %pattern_type.548: type = pattern_type %Optional.1ac [concrete]
// CHECK:STDOUT:   %Direct.type: type = fn_type @Direct [concrete]
// CHECK:STDOUT:   %Direct: %Direct.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a33: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.1ac)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.3b8: type = facet_type <@OptionalAs, @OptionalAs(%T.220)> [symbolic]
// CHECK:STDOUT:   %U.042: %OptionalAs.type.3b8 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.ad5: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.167(%T.220, %U.042) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.331: %U.as_type.as.ImplicitAs.impl.Convert.type.ad5 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.345: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.822)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.774: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.220) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.5f9: %T.as_type.as.OptionalAs.impl.Convert.type.774 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.fd1: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.2e9, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.822) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.345 = facet_value %ptr.6cb, (%OptionalAs.impl_witness.fd1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.913: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bb5, @U.as_type.as.ImplicitAs.impl.167(%OptionalStorage.facet.822, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.abb: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.167(%OptionalStorage.facet.822, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.8c6: %U.as_type.as.ImplicitAs.impl.Convert.type.abb = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.032: %ImplicitAs.type.a33 = facet_value %ptr.6cb, (%ImplicitAs.impl_witness.913) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8a9: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.1ac, %ImplicitAs.facet.032) [concrete]
// CHECK:STDOUT:   %.96f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8a9, %ImplicitAs.facet.032 [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.as_type.as.ImplicitAs.impl.Convert.8c6, @U.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.822, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc11_14.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.548 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %Indirect.cpp_overload_set.type: type = cpp_overload_set_type @Indirect.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Indirect.cpp_overload_set.value: %Indirect.cpp_overload_set.type = cpp_overload_set_value @Indirect.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Indirect__carbon_thunk.type: type = fn_type @Indirect__carbon_thunk [concrete]
// CHECK:STDOUT:   %Indirect__carbon_thunk: %Indirect__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.dc4: ref %DeducedPointerS = temporary invalid, %DeducedPointerS.val [concrete]
// CHECK:STDOUT:   %DeducedPointerS.cpp_destructor.type: type = fn_type @DeducedPointerS.cpp_destructor [concrete]
// CHECK:STDOUT:   %DeducedPointerS.cpp_destructor: %DeducedPointerS.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DeducedPointerS = %DeducedPointerS.decl
// CHECK:STDOUT:     .Direct = %Direct.cpp_overload_set.value
// CHECK:STDOUT:     .Indirect = %Indirect.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeducedPointerS.decl: type = class_decl @DeducedPointerS [concrete = constants.%DeducedPointerS] {} {}
// CHECK:STDOUT:   %Direct.cpp_overload_set.value: %Direct.cpp_overload_set.type = cpp_overload_set_value @Direct.cpp_overload_set [concrete = constants.%Direct.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.import_ref.509: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.ddb)]
// CHECK:STDOUT:   %Core.import_ref.105 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.fa7: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.71e) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.caa)]
// CHECK:STDOUT:   %Core.import_ref.a90 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.306 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3e3 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.509, %Core.import_ref.105, %Core.import_ref.fa7, %Core.import_ref.a90, %Core.import_ref.306, %Core.import_ref.3e3), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Direct.decl: %Direct.type = fn_decl @Direct [concrete = constants.%Direct] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %OptionalStorage.facet.loc11_16.1: %OptionalStorage.type = facet_value constants.%ptr.6cb, (constants.%OptionalStorage.impl_witness.5d7) [concrete = constants.%OptionalStorage.facet.822]
// CHECK:STDOUT:     %.loc11_16.1: %OptionalStorage.type = converted constants.%ptr.6cb, %OptionalStorage.facet.loc11_16.1 [concrete = constants.%OptionalStorage.facet.822]
// CHECK:STDOUT:     %Optional.loc11_16.1: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.822) [concrete = constants.%Optional.1ac]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_16.2: type = splice_block %Optional.loc11_16.2 [concrete = constants.%Optional.1ac] {
// CHECK:STDOUT:       %OptionalStorage.facet.loc11_16.2: %OptionalStorage.type = facet_value constants.%ptr.6cb, (constants.%OptionalStorage.impl_witness.5d7) [concrete = constants.%OptionalStorage.facet.822]
// CHECK:STDOUT:       %.loc11_16.3: %OptionalStorage.type = converted constants.%ptr.6cb, %OptionalStorage.facet.loc11_16.2 [concrete = constants.%OptionalStorage.facet.822]
// CHECK:STDOUT:       %Optional.loc11_16.2: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.822) [concrete = constants.%Optional.1ac]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.eb9: @U.as_type.as.ImplicitAs.impl.167.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.ad5) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.167.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.331)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bb5 = impl_witness_table (%Core.import_ref.eb9), @U.as_type.as.ImplicitAs.impl.167 [concrete]
// CHECK:STDOUT:   %Core.import_ref.017d8: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.774) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.5f9)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.2e9 = impl_witness_table (%Core.import_ref.017d8), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Indirect.cpp_overload_set.value: %Indirect.cpp_overload_set.type = cpp_overload_set_value @Indirect.cpp_overload_set [concrete = constants.%Indirect.cpp_overload_set.value]
// CHECK:STDOUT:   %Indirect__carbon_thunk.decl: %Indirect__carbon_thunk.type = fn_decl @Indirect__carbon_thunk [concrete = constants.%Indirect__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.6cb, (constants.%OptionalStorage.impl_witness.5d7) [concrete = constants.%OptionalStorage.facet.822]
// CHECK:STDOUT:     %.loc13: %OptionalStorage.type = converted constants.%ptr.6cb, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.822]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.822) [concrete = constants.%Optional.1ac]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref %DeducedPointerS = var_storage %s.var_patt
// CHECK:STDOUT:   %.loc10_33.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_33.2: init %DeducedPointerS to %s.var = class_init () [concrete = constants.%DeducedPointerS.val]
// CHECK:STDOUT:   %.loc10_3: init %DeducedPointerS = converted %.loc10_33.1, %.loc10_33.2 [concrete = constants.%DeducedPointerS.val]
// CHECK:STDOUT:   assign %s.var, %.loc10_3
// CHECK:STDOUT:   %.loc10_13: type = splice_block %DeducedPointerS.ref.loc10 [concrete = constants.%DeducedPointerS] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DeducedPointerS.ref.loc10: type = name_ref DeducedPointerS, imports.%DeducedPointerS.decl [concrete = constants.%DeducedPointerS]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %DeducedPointerS = wrapper_binding s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.36d = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.36d = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Direct.ref: %Direct.cpp_overload_set.type = name_ref Direct, imports.%Direct.cpp_overload_set.value [concrete = constants.%Direct.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %DeducedPointerS = name_ref s, %s
// CHECK:STDOUT:   %addr.loc11: %ptr.6cb = addr_of %s.ref
// CHECK:STDOUT:   %impl.elem0: %.96f = impl_witness_access constants.%ImplicitAs.impl_witness.913, element0 [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.8c6]
// CHECK:STDOUT:   %bound_method.loc11_14.1: <bound method> = bound_method %addr.loc11, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @U.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.822, constants.%OptionalAs.facet) [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_14.2: <bound method> = bound_method %addr.loc11, %specific_fn
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.1ac = call %bound_method.loc11_14.2(%addr.loc11)
// CHECK:STDOUT:   %.loc11_14.1: init %Optional.1ac = converted %addr.loc11, %U.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_14.2: ref %Optional.1ac = temporary_storage
// CHECK:STDOUT:   %.loc11_14.3: ref %Optional.1ac = temporary %.loc11_14.2, %.loc11_14.1
// CHECK:STDOUT:   %.loc11_14.4: %Optional.1ac = acquire_value %.loc11_14.3
// CHECK:STDOUT:   %Direct.call: init %Optional.1ac = call imports.%Direct.decl(%.loc11_14.4)
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %.loc11_14.3, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%.loc11_14.3)
// CHECK:STDOUT:   %Cpp.ref.loc13_55: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Indirect.ref: %Indirect.cpp_overload_set.type = name_ref Indirect, imports.%Indirect.cpp_overload_set.value [concrete = constants.%Indirect.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc13_69.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Cpp.ref.loc13_74: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %DeducedPointerS.ref.loc13_77: type = name_ref DeducedPointerS, imports.%DeducedPointerS.decl [concrete = constants.%DeducedPointerS]
// CHECK:STDOUT:   %.loc13_69.2: ref %DeducedPointerS = temporary_storage
// CHECK:STDOUT:   %.loc13_69.3: init %DeducedPointerS to %.loc13_69.2 = class_init () [concrete = constants.%DeducedPointerS.val]
// CHECK:STDOUT:   %.loc13_71.1: init %DeducedPointerS = converted %.loc13_69.1, %.loc13_69.3 [concrete = constants.%DeducedPointerS.val]
// CHECK:STDOUT:   %.loc13_71.2: ref %DeducedPointerS = temporary %.loc13_69.2, %.loc13_71.1 [concrete = constants.%.dc4]
// CHECK:STDOUT:   %.loc13_71.3: %DeducedPointerS = acquire_value %.loc13_71.2 [concrete = constants.%DeducedPointerS.val]
// CHECK:STDOUT:   %.loc13_71.4: ref %DeducedPointerS = value_as_ref %.loc13_71.3
// CHECK:STDOUT:   %addr.loc13: %ptr.6cb = addr_of %.loc13_71.4
// CHECK:STDOUT:   %Indirect__carbon_thunk.call: init %Optional.1ac = call imports.%Indirect__carbon_thunk.decl(%addr.loc13)
// CHECK:STDOUT:   %.loc13_93.1: ref %Optional.1ac = temporary_storage
// CHECK:STDOUT:   %.loc13_93.2: ref %Optional.1ac = temporary %.loc13_93.1, %Indirect__carbon_thunk.call
// CHECK:STDOUT:   %.loc13_93.3: %Optional.1ac = acquire_value %.loc13_93.2
// CHECK:STDOUT:   %.loc13_51.1: type = splice_block %Optional [concrete = constants.%Optional.1ac] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Cpp.ref.loc13_31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DeducedPointerS.ref.loc13_34: type = name_ref DeducedPointerS, imports.%DeducedPointerS.decl [concrete = constants.%DeducedPointerS]
// CHECK:STDOUT:     %ptr: type = ptr_type %DeducedPointerS.ref.loc13_34 [concrete = constants.%ptr.6cb]
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.5d7) [concrete = constants.%OptionalStorage.facet.822]
// CHECK:STDOUT:     %.loc13_51.2: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.822]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.822) [concrete = constants.%Optional.1ac]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %Optional.1ac = wrapper_binding a, %.loc13_93.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.548 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.Op.bound.loc13: <bound method> = bound_method %.loc13_93.2, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call.loc13: init %empty_tuple.type = call %Destroy.Op.bound.loc13(%.loc13_93.2)
// CHECK:STDOUT:   %DeducedPointerS.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%DeducedPointerS.cpp_destructor
// CHECK:STDOUT:   %DeducedPointerS.cpp_destructor.call: init %empty_tuple.type = call %DeducedPointerS.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_14.1(%self.param: ref %.fb1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_14.2(%self.param: ref %MaybeUnformed.dcc) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_14.3(%self.param: ref %Optional.1ac) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
