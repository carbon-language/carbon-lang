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
  // CHECK:STDERR: fail_import_non_nullable_pointer_param_using_const_value.carbon:[[@LINE+5]]:3-28: error: no matching function for call to 'non_nullable_param_foo' [CppInteropParseError]
  // CHECK:STDERR: included from: fail_import_non_nullable_pointer_param_using_const_value.carbon:[[@LINE-7]]:10 [InCppInclude]
  // CHECK:STDERR: ./non_nullable_pointer_param.h:4:6: note: candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR: ./non_nullable_pointer_param.h:4:29-55: note: 1st argument ('const NonNullableParamS * _Nonnull') would lose const qualifier [CppInteropParseNoteReason]
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
  // These pass or return an Optional(DeducedPointerS*), not a DeducedPointerS*.

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
  // CHECK:STDERR: fail_import_deduced_pointer_param_as_optional_pointer.carbon:[[@LINE+3]]:40-78: error: cannot implicitly convert expression of type `Core.Optional(Cpp.DeducedPointerS* as Core.OptionalStorage)` to `Cpp.DeducedPointerS*` [ConversionFailure]
  // CHECK:STDERR: fail_import_deduced_pointer_param_as_optional_pointer.carbon:[[@LINE+2]]:40-78: note: type `Core.Optional(Cpp.DeducedPointerS* as Core.OptionalStorage)` does not implement interface `Core.ImplicitAs(Cpp.DeducedPointerS*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  let unused a: Cpp.DeducedPointerS* = Cpp.Indirect({} as Cpp.DeducedPointerS);
}

// CHECK:STDOUT: --- import_non_nullable_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.170: %SubobjectDestroy.type = facet_value %NonNullableParamS, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.dbc: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.170) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %NonNullableParamS, (%DestroyImpl.impl_witness.dbc, %custom_witness.b80) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref %NonNullableParamS = var_storage %s.var_patt
// CHECK:STDOUT:   %.loc8_35.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_35.2: init %NonNullableParamS to %s.var = class_init () [concrete = constants.%NonNullableParamS.val]
// CHECK:STDOUT:   %.loc8_3.1: init %NonNullableParamS = converted %.loc8_35.1, %.loc8_35.2 [concrete = constants.%NonNullableParamS.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3.1
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
// CHECK:STDOUT:   %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%NonNullableParamS, (constants.%DestroyImpl.impl_witness.dbc, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_3.2: %Destroy.type = converted constants.%NonNullableParamS, %Destroy.facet.loc8_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%NonNullableParamS, (constants.%DestroyImpl.impl_witness.dbc, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_3.3: %Destroy.type = converted constants.%NonNullableParamS, %Destroy.facet.loc8_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %NonNullableParamS) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_nullable_pointer_param_using_const_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.32c: %SubobjectDestroy.type = facet_value %const, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.fd5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.32c) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %const, (%DestroyImpl.impl_witness.fd5, %custom_witness.b80ef6.2) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.var: ref %const = var_storage %p.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %const = call %G.ref()
// CHECK:STDOUT:   assign %p.var, %G.call
// CHECK:STDOUT:   %.loc10_10: type = splice_block %const [concrete = constants.%const] {
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
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%const, (constants.%DestroyImpl.impl_witness.fd5, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_3.1: %Destroy.type = converted constants.%const, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%const, (constants.%DestroyImpl.impl_witness.fd5, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%const, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%p.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.1(%self.param: ref %ptr) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.2(%self.param: ref %const) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_double_non_nullable_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.20f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.f12: %ptr.as.Copy.impl.Op.type.20f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f50: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%DoubleNonNullableParamS) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.5b1: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%DoubleNonNullableParamS) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.43a: %ptr.as.Copy.impl.Op.type.5b1 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.40d, (%Copy.impl_witness.f50) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.104: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.38b: type = fn_type_with_self_type %Copy.WithSelf.Op.type.104, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.43a, @ptr.as.Copy.impl.Op(%DoubleNonNullableParamS) [concrete]
// CHECK:STDOUT:   %double_non_nullable_param_foo.cpp_overload_set.type: type = cpp_overload_set_type @double_non_nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %double_non_nullable_param_foo.cpp_overload_set.value: %double_non_nullable_param_foo.cpp_overload_set.type = cpp_overload_set_value @double_non_nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.2a5: type = ptr_type %ptr.40d [concrete]
// CHECK:STDOUT:   %double_non_nullable_param_foo.type: type = fn_type @double_non_nullable_param_foo [concrete]
// CHECK:STDOUT:   %double_non_nullable_param_foo: %double_non_nullable_param_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.07a: %SubobjectDestroy.type = facet_value %ptr.40d, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.aa5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.07a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.16e: %Destroy.type = facet_value %ptr.40d, (%DestroyImpl.impl_witness.aa5, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.49e: %SubobjectDestroy.type = facet_value %DoubleNonNullableParamS, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.320: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.49e) [concrete]
// CHECK:STDOUT:   %Destroy.facet.044: %Destroy.type = facet_value %DoubleNonNullableParamS, (%DestroyImpl.impl_witness.320, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DoubleNonNullableParamS = %DoubleNonNullableParamS.decl
// CHECK:STDOUT:     .double_non_nullable_param_foo = %double_non_nullable_param_foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DoubleNonNullableParamS.decl: type = class_decl @DoubleNonNullableParamS [concrete = constants.%DoubleNonNullableParamS] {} {}
// CHECK:STDOUT:   %Core.import_ref.06a: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.20f) = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.f12)]
// CHECK:STDOUT:   %Copy.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.06a), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %double_non_nullable_param_foo.cpp_overload_set.value: %double_non_nullable_param_foo.cpp_overload_set.type = cpp_overload_set_value @double_non_nullable_param_foo.cpp_overload_set [concrete = constants.%double_non_nullable_param_foo.cpp_overload_set.value]
// CHECK:STDOUT:   %double_non_nullable_param_foo.decl: %double_non_nullable_param_foo.type = fn_decl @double_non_nullable_param_foo [concrete = constants.%double_non_nullable_param_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref %DoubleNonNullableParamS = var_storage %s.var_patt
// CHECK:STDOUT:   %.loc8_41.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_41.2: init %DoubleNonNullableParamS to %s.var = class_init () [concrete = constants.%DoubleNonNullableParamS.val]
// CHECK:STDOUT:   %.loc8_3.1: init %DoubleNonNullableParamS = converted %.loc8_41.1, %.loc8_41.2 [concrete = constants.%DoubleNonNullableParamS.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3.1
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
// CHECK:STDOUT:   %impl.elem0: %.38b = impl_witness_access constants.%Copy.impl_witness.f50, element0 [concrete = constants.%ptr.as.Copy.impl.Op.43a]
// CHECK:STDOUT:   %bound_method.loc9_41.1: <bound method> = bound_method %addr.loc9, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%DoubleNonNullableParamS) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_41.2: <bound method> = bound_method %addr.loc9, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.40d = call %bound_method.loc9_41.2(%addr.loc9)
// CHECK:STDOUT:   assign %p.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9_37: type = splice_block %ptr [concrete = constants.%ptr.40d] {
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
// CHECK:STDOUT:   %Destroy.facet.loc9_3.1: %Destroy.type = facet_value constants.%ptr.40d, (constants.%DestroyImpl.impl_witness.aa5, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.16e]
// CHECK:STDOUT:   %.loc9_3.1: %Destroy.type = converted constants.%ptr.40d, %Destroy.facet.loc9_3.1 [concrete = constants.%Destroy.facet.16e]
// CHECK:STDOUT:   %Destroy.facet.loc9_3.2: %Destroy.type = facet_value constants.%ptr.40d, (constants.%DestroyImpl.impl_witness.aa5, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.16e]
// CHECK:STDOUT:   %.loc9_3.2: %Destroy.type = converted constants.%ptr.40d, %Destroy.facet.loc9_3.2 [concrete = constants.%Destroy.facet.16e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%p.var)
// CHECK:STDOUT:   %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%DoubleNonNullableParamS, (constants.%DestroyImpl.impl_witness.320, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.044]
// CHECK:STDOUT:   %.loc8_3.2: %Destroy.type = converted constants.%DoubleNonNullableParamS, %Destroy.facet.loc8_3.1 [concrete = constants.%Destroy.facet.044]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%DoubleNonNullableParamS, (constants.%DestroyImpl.impl_witness.320, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.044]
// CHECK:STDOUT:   %.loc8_3.3: %Destroy.type = converted constants.%DoubleNonNullableParamS, %Destroy.facet.loc8_3.2 [concrete = constants.%Destroy.facet.044]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc8: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9(%self.param: ref %ptr.40d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %DoubleNonNullableParamS) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_nullable_pointer_to_const_param_using_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6e8: %SubobjectDestroy.type = facet_value %const, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.229: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6e8) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %const, (%DestroyImpl.impl_witness.229, %custom_witness.b80ef6.2) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref %const = var_storage %s.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc10_3.1: ref %const = splice_block %s.var {}
// CHECK:STDOUT:   %G.call: init %const to %.loc10_3.1 = call %G.ref()
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
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%const, (constants.%DestroyImpl.impl_witness.229, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%const, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%const, (constants.%DestroyImpl.impl_witness.229, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_3.3: %Destroy.type = converted constants.%const, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.1(%self.param: ref %NonNullablePointerToConstParamS) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.2(%self.param: ref %const) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_nullable_pointer_to_const_param_using_non_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.795: %SubobjectDestroy.type = facet_value %NonNullablePointerToConstParamS, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.0f2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.795) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %NonNullablePointerToConstParamS, (%DestroyImpl.impl_witness.0f2, %custom_witness.b80) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref %NonNullablePointerToConstParamS = var_storage %s.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc8_3.1: ref %NonNullablePointerToConstParamS = splice_block %s.var {}
// CHECK:STDOUT:   %G.call: init %NonNullablePointerToConstParamS to %.loc8_3.1 = call %G.ref()
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
// CHECK:STDOUT:   %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%NonNullablePointerToConstParamS, (constants.%DestroyImpl.impl_witness.0f2, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_3.2: %Destroy.type = converted constants.%NonNullablePointerToConstParamS, %Destroy.facet.loc8_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%NonNullablePointerToConstParamS, (constants.%DestroyImpl.impl_witness.0f2, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_3.3: %Destroy.type = converted constants.%NonNullablePointerToConstParamS, %Destroy.facet.loc8_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %NonNullablePointerToConstParamS) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_const_non_nullable_pointer_param_using_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5e9: %SubobjectDestroy.type = facet_value %const, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a57: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5e9) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %const, (%DestroyImpl.impl_witness.a57, %custom_witness.b80ef6.2) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.var: ref %const = var_storage %p.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %const = call %G.ref()
// CHECK:STDOUT:   assign %p.var, %G.call
// CHECK:STDOUT:   %.loc10_10: type = splice_block %const [concrete = constants.%const] {
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
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%const, (constants.%DestroyImpl.impl_witness.a57, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_3.1: %Destroy.type = converted constants.%const, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%const, (constants.%DestroyImpl.impl_witness.a57, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%const, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%p.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.1(%self.param: ref %ptr) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.2(%self.param: ref %const) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_const_non_nullable_pointer_param_using_non_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.0d1: %SubobjectDestroy.type = facet_value %ptr, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e0c: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.0d1) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ptr, (%DestroyImpl.impl_witness.e0c, %custom_witness.b80) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.var: ref %ptr = var_storage %p.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %ptr = call %G.ref()
// CHECK:STDOUT:   assign %p.var, %G.call
// CHECK:STDOUT:   %.loc10_36: type = splice_block %ptr [concrete = constants.%ptr] {
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
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%ptr, (constants.%DestroyImpl.impl_witness.e0c, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_3.1: %Destroy.type = converted constants.%ptr, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%ptr, (constants.%DestroyImpl.impl_witness.e0c, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%ptr, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%p.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %ptr) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_const_nullable_pointer_param_using_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %T.220e40.1: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.61e: <witness> = lookup_impl_witness %ptr.e8f, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.18b: <witness> = lookup_impl_witness %ptr.e8f, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.5f3: %Destroy.type = facet_value %ptr.e8f, (%SubobjectDestroy.lookup_impl_witness.61e, %DestroyImpl.lookup_impl_witness.18b) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.dac: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.5f3) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.b0c: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.bf1: %ptr.as.OptionalStorage.impl.Some.type.b0c = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f99: %SubobjectDestroy.type = facet_value %ptr.13b, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c03: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f99) [concrete]
// CHECK:STDOUT:   %Destroy.facet.0a4: %Destroy.type = facet_value %ptr.13b, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.c03) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.97c: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.0a4) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.361: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%ConstNullableParamS) [concrete]
// CHECK:STDOUT:   %.c5e: type = maybe_unformed_type %ptr.13b [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.9ab: %OptionalStorage.type = facet_value %ptr.13b, (%OptionalStorage.impl_witness.361) [concrete]
// CHECK:STDOUT:   %Optional.4b8: type = class_type @Optional, @Optional(%OptionalStorage.facet.9ab) [concrete]
// CHECK:STDOUT:   %const_nullable_foo.type: type = fn_type @const_nullable_foo [concrete]
// CHECK:STDOUT:   %const_nullable_foo: %const_nullable_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.19b: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.4b8)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b02b67.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T.67d)> [symbolic]
// CHECK:STDOUT:   %U.862: %ImplicitAs.type.b02b67.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.type.dba: type = fn_type @const.as.ImplicitAs.impl.Convert, @const.as.ImplicitAs.impl(%T.67d, %U.862) [symbolic]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.de3: %const.as.ImplicitAs.impl.Convert.type.dba = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.5bd: type = facet_type <@OptionalAs, @OptionalAs(%T.220e40.1)> [symbolic]
// CHECK:STDOUT:   %U.c66: %OptionalAs.type.5bd = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.c47: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.0c0(%T.220e40.1, %U.c66) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.f83: %U.as_type.as.ImplicitAs.impl.Convert.type.c47 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.168: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.9ab)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.b2a: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.220e40.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.c36: %T.as_type.as.OptionalAs.impl.Convert.type.b2a = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.e49: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.98f, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.9ab) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.168 = facet_value %ptr.13b, (%OptionalAs.impl_witness.e49) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3ab: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.653, @U.as_type.as.ImplicitAs.impl.0c0(%OptionalStorage.facet.9ab, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f93: %ImplicitAs.type.19b = facet_value %ptr.13b, (%ImplicitAs.impl_witness.3ab) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.818: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c21, @const.as.ImplicitAs.impl(%Optional.4b8, %ImplicitAs.facet.f93) [concrete]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.type.c95: type = fn_type @const.as.ImplicitAs.impl.Convert, @const.as.ImplicitAs.impl(%Optional.4b8, %ImplicitAs.facet.f93) [concrete]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.f0b: %const.as.ImplicitAs.impl.Convert.type.c95 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.362: %ImplicitAs.type.19b = facet_value %const.2f4, (%ImplicitAs.impl_witness.818) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.9d7: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.4b8, %ImplicitAs.facet.362) [concrete]
// CHECK:STDOUT:   %.8ee: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.9d7, %ImplicitAs.facet.362 [concrete]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %const.as.ImplicitAs.impl.Convert.f0b, @const.as.ImplicitAs.impl.Convert(%Optional.4b8, %ImplicitAs.facet.f93) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_26.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f00: %SubobjectDestroy.type = facet_value %Optional.4b8, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.629: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f00) [concrete]
// CHECK:STDOUT:   %Destroy.facet.6e2: %Destroy.type = facet_value %Optional.4b8, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.629) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.7e8: %SubobjectDestroy.type = facet_value %const.2f4, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b37: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.7e8) [concrete]
// CHECK:STDOUT:   %Destroy.facet.a52: %Destroy.type = facet_value %const.2f4, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.b37) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.30b: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.dac)]
// CHECK:STDOUT:   %Core.import_ref.814 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.470: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.b0c) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.bf1)]
// CHECK:STDOUT:   %Core.import_ref.322 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0e8 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.713 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.30b, %Core.import_ref.814, %Core.import_ref.470, %Core.import_ref.322, %Core.import_ref.0e8, %Core.import_ref.713), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b53: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b53), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %const_nullable_foo.decl: %const_nullable_foo.type = fn_decl @const_nullable_foo [concrete = constants.%const_nullable_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_27.1: type = splice_block %Optional [concrete = constants.%Optional.4b8] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.13b, (constants.%OptionalStorage.impl_witness.361) [concrete = constants.%OptionalStorage.facet.9ab]
// CHECK:STDOUT:       %.loc11_27.2: %OptionalStorage.type = converted constants.%ptr.13b, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.9ab]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.9ab) [concrete = constants.%Optional.4b8]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.caf: @const.as.ImplicitAs.impl.%const.as.ImplicitAs.impl.Convert.type (%const.as.ImplicitAs.impl.Convert.type.dba) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @const.as.ImplicitAs.impl.%const.as.ImplicitAs.impl.Convert (constants.%const.as.ImplicitAs.impl.Convert.de3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c21 = impl_witness_table (%Core.import_ref.caf), @const.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.098: @U.as_type.as.ImplicitAs.impl.0c0.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.c47) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.0c0.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.f83)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.653 = impl_witness_table (%Core.import_ref.098), @U.as_type.as.ImplicitAs.impl.0c0 [concrete]
// CHECK:STDOUT:   %Core.import_ref.49b: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.b2a) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.c36)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.98f = impl_witness_table (%Core.import_ref.49b), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.var: ref %const.2f4 = var_storage %p.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %const.2f4 = call %G.ref()
// CHECK:STDOUT:   assign %p.var, %G.call
// CHECK:STDOUT:   %.loc10_10: type = splice_block %const [concrete = constants.%const.2f4] {
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
// CHECK:STDOUT:   %impl.elem0: %.8ee = impl_witness_access constants.%ImplicitAs.impl_witness.818, element0 [concrete = constants.%const.as.ImplicitAs.impl.Convert.f0b]
// CHECK:STDOUT:   %bound_method.loc11_26.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @const.as.ImplicitAs.impl.Convert(constants.%Optional.4b8, constants.%ImplicitAs.facet.f93) [concrete = constants.%const.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_26.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %.loc11_26.1: %const.2f4 = acquire_value %p.ref
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.call: init %Optional.4b8 = call %bound_method.loc11_26.2(%.loc11_26.1)
// CHECK:STDOUT:   %.loc11_26.2: init %Optional.4b8 = converted %p.ref, %const.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_26.3: ref %Optional.4b8 = temporary_storage
// CHECK:STDOUT:   %.loc11_26.4: ref %Optional.4b8 = temporary %.loc11_26.3, %.loc11_26.2
// CHECK:STDOUT:   %.loc11_26.5: %Optional.4b8 = acquire_value %.loc11_26.4
// CHECK:STDOUT:   %const_nullable_foo.call: init %empty_tuple.type = call imports.%const_nullable_foo.decl(%.loc11_26.5)
// CHECK:STDOUT:   %Destroy.facet.loc11_26.1: %Destroy.type = facet_value constants.%Optional.4b8, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.629) [concrete = constants.%Destroy.facet.6e2]
// CHECK:STDOUT:   %.loc11_26.6: %Destroy.type = converted constants.%Optional.4b8, %Destroy.facet.loc11_26.1 [concrete = constants.%Destroy.facet.6e2]
// CHECK:STDOUT:   %Destroy.facet.loc11_26.2: %Destroy.type = facet_value constants.%Optional.4b8, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.629) [concrete = constants.%Destroy.facet.6e2]
// CHECK:STDOUT:   %.loc11_26.7: %Destroy.type = converted constants.%Optional.4b8, %Destroy.facet.loc11_26.2 [concrete = constants.%Destroy.facet.6e2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc11_26.4)
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%const.2f4, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.b37) [concrete = constants.%Destroy.facet.a52]
// CHECK:STDOUT:   %.loc10_3.1: %Destroy.type = converted constants.%const.2f4, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet.a52]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%const.2f4, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.b37) [concrete = constants.%Destroy.facet.a52]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%const.2f4, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet.a52]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%p.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_26.1(%self.param: ref %.c5e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_26.2(%self.param: ref %MaybeUnformed.97c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_26.3(%self.param: ref %Optional.4b8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %const.2f4) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_const_nullable_pointer_param_using_non_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %T.220e40.1: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.61e: <witness> = lookup_impl_witness %ptr.e8f, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.18b: <witness> = lookup_impl_witness %ptr.e8f, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.5f3: %Destroy.type = facet_value %ptr.e8f, (%SubobjectDestroy.lookup_impl_witness.61e, %DestroyImpl.lookup_impl_witness.18b) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.dac: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.5f3) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.b0c: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.bf1: %ptr.as.OptionalStorage.impl.Some.type.b0c = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f99: %SubobjectDestroy.type = facet_value %ptr.13b, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c03: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f99) [concrete]
// CHECK:STDOUT:   %Destroy.facet.0a4: %Destroy.type = facet_value %ptr.13b, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.c03) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.97c: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.0a4) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.361: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%ConstNullableParamS) [concrete]
// CHECK:STDOUT:   %.c5e: type = maybe_unformed_type %ptr.13b [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.9ab: %OptionalStorage.type = facet_value %ptr.13b, (%OptionalStorage.impl_witness.361) [concrete]
// CHECK:STDOUT:   %Optional.4b8: type = class_type @Optional, @Optional(%OptionalStorage.facet.9ab) [concrete]
// CHECK:STDOUT:   %const_nullable_foo.type: type = fn_type @const_nullable_foo [concrete]
// CHECK:STDOUT:   %const_nullable_foo: %const_nullable_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.19b: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.4b8)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.5bd: type = facet_type <@OptionalAs, @OptionalAs(%T.220e40.1)> [symbolic]
// CHECK:STDOUT:   %U.c66: %OptionalAs.type.5bd = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.c47: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.0c0(%T.220e40.1, %U.c66) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.f83: %U.as_type.as.ImplicitAs.impl.Convert.type.c47 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.168: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.9ab)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.b2a: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.220e40.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.c36: %T.as_type.as.OptionalAs.impl.Convert.type.b2a = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.e49: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.98f, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.9ab) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.168 = facet_value %ptr.13b, (%OptionalAs.impl_witness.e49) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3ab: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.653, @U.as_type.as.ImplicitAs.impl.0c0(%OptionalStorage.facet.9ab, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.d7e: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.0c0(%OptionalStorage.facet.9ab, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.311: %U.as_type.as.ImplicitAs.impl.Convert.type.d7e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f93: %ImplicitAs.type.19b = facet_value %ptr.13b, (%ImplicitAs.impl_witness.3ab) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a91: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.4b8, %ImplicitAs.facet.f93) [concrete]
// CHECK:STDOUT:   %.56e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a91, %ImplicitAs.facet.f93 [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.as_type.as.ImplicitAs.impl.Convert.311, @U.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.9ab, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_26.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f00: %SubobjectDestroy.type = facet_value %Optional.4b8, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.629: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f00) [concrete]
// CHECK:STDOUT:   %Destroy.facet.6e2: %Destroy.type = facet_value %Optional.4b8, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.629) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.30b: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.dac)]
// CHECK:STDOUT:   %Core.import_ref.814 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.470: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.b0c) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.bf1)]
// CHECK:STDOUT:   %Core.import_ref.322 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0e8 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.713 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.30b, %Core.import_ref.814, %Core.import_ref.470, %Core.import_ref.322, %Core.import_ref.0e8, %Core.import_ref.713), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b53: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b53), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %const_nullable_foo.decl: %const_nullable_foo.type = fn_decl @const_nullable_foo [concrete = constants.%const_nullable_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_27.1: type = splice_block %Optional [concrete = constants.%Optional.4b8] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.13b, (constants.%OptionalStorage.impl_witness.361) [concrete = constants.%OptionalStorage.facet.9ab]
// CHECK:STDOUT:       %.loc11_27.2: %OptionalStorage.type = converted constants.%ptr.13b, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.9ab]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.9ab) [concrete = constants.%Optional.4b8]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.098: @U.as_type.as.ImplicitAs.impl.0c0.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.c47) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.0c0.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.f83)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.653 = impl_witness_table (%Core.import_ref.098), @U.as_type.as.ImplicitAs.impl.0c0 [concrete]
// CHECK:STDOUT:   %Core.import_ref.49b: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.b2a) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.c36)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.98f = impl_witness_table (%Core.import_ref.49b), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.var: ref %ptr.13b = var_storage %p.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %ptr.13b = call %G.ref()
// CHECK:STDOUT:   assign %p.var, %G.call
// CHECK:STDOUT:   %.loc10_33: type = splice_block %ptr [concrete = constants.%ptr.13b] {
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
// CHECK:STDOUT:   %impl.elem0: %.56e = impl_witness_access constants.%ImplicitAs.impl_witness.3ab, element0 [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.311]
// CHECK:STDOUT:   %bound_method.loc11_26.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @U.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.9ab, constants.%OptionalAs.facet) [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_26.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %.loc11_26.1: %ptr.13b = acquire_value %p.ref
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.4b8 = call %bound_method.loc11_26.2(%.loc11_26.1)
// CHECK:STDOUT:   %.loc11_26.2: init %Optional.4b8 = converted %p.ref, %U.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_26.3: ref %Optional.4b8 = temporary_storage
// CHECK:STDOUT:   %.loc11_26.4: ref %Optional.4b8 = temporary %.loc11_26.3, %.loc11_26.2
// CHECK:STDOUT:   %.loc11_26.5: %Optional.4b8 = acquire_value %.loc11_26.4
// CHECK:STDOUT:   %const_nullable_foo.call: init %empty_tuple.type = call imports.%const_nullable_foo.decl(%.loc11_26.5)
// CHECK:STDOUT:   %Destroy.facet.loc11_26.1: %Destroy.type = facet_value constants.%Optional.4b8, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.629) [concrete = constants.%Destroy.facet.6e2]
// CHECK:STDOUT:   %.loc11_26.6: %Destroy.type = converted constants.%Optional.4b8, %Destroy.facet.loc11_26.1 [concrete = constants.%Destroy.facet.6e2]
// CHECK:STDOUT:   %Destroy.facet.loc11_26.2: %Destroy.type = facet_value constants.%Optional.4b8, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.629) [concrete = constants.%Destroy.facet.6e2]
// CHECK:STDOUT:   %.loc11_26.7: %Destroy.type = converted constants.%Optional.4b8, %Destroy.facet.loc11_26.2 [concrete = constants.%Destroy.facet.6e2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc11_26.4)
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%ptr.13b, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.c03) [concrete = constants.%Destroy.facet.0a4]
// CHECK:STDOUT:   %.loc10_3.1: %Destroy.type = converted constants.%ptr.13b, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet.0a4]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%ptr.13b, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.c03) [concrete = constants.%Destroy.facet.0a4]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%ptr.13b, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet.0a4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%p.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_26.1(%self.param: ref %.c5e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_26.2(%self.param: ref %MaybeUnformed.97c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_26.3(%self.param: ref %Optional.4b8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_nullable_pointer_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %T.220e40.1: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.61e: <witness> = lookup_impl_witness %ptr.e8f, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.18b: <witness> = lookup_impl_witness %ptr.e8f, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.5f3: %Destroy.type = facet_value %ptr.e8f, (%SubobjectDestroy.lookup_impl_witness.61e, %DestroyImpl.lookup_impl_witness.18b) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.dac: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.5f3) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.b0c: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.bf1: %ptr.as.OptionalStorage.impl.Some.type.b0c = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a1d: %SubobjectDestroy.type = facet_value %ptr.1f0, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.82b: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a1d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.6e0: %Destroy.type = facet_value %ptr.1f0, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.82b) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.b5a: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.6e0) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.b77: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%NullableParamS) [concrete]
// CHECK:STDOUT:   %.280: type = maybe_unformed_type %ptr.1f0 [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.9ba: %OptionalStorage.type = facet_value %ptr.1f0, (%OptionalStorage.impl_witness.b77) [concrete]
// CHECK:STDOUT:   %Optional.37f: type = class_type @Optional, @Optional(%OptionalStorage.facet.9ba) [concrete]
// CHECK:STDOUT:   %nullable_param_foo.type: type = fn_type @nullable_param_foo [concrete]
// CHECK:STDOUT:   %nullable_param_foo: %nullable_param_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7b3: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.37f)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.5bd: type = facet_type <@OptionalAs, @OptionalAs(%T.220e40.1)> [symbolic]
// CHECK:STDOUT:   %U.c66: %OptionalAs.type.5bd = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.c47: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.0c0(%T.220e40.1, %U.c66) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.f83: %U.as_type.as.ImplicitAs.impl.Convert.type.c47 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.300: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.9ba)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.b2a: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.220e40.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.c36: %T.as_type.as.OptionalAs.impl.Convert.type.b2a = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.0b4: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.98f, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.9ba) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.300 = facet_value %ptr.1f0, (%OptionalAs.impl_witness.0b4) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.940: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.653, @U.as_type.as.ImplicitAs.impl.0c0(%OptionalStorage.facet.9ba, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.ce0: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.0c0(%OptionalStorage.facet.9ba, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.d9f: %U.as_type.as.ImplicitAs.impl.Convert.type.ce0 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.162: %ImplicitAs.type.7b3 = facet_value %ptr.1f0, (%ImplicitAs.impl_witness.940) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a1f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.37f, %ImplicitAs.facet.162) [concrete]
// CHECK:STDOUT:   %.94f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a1f, %ImplicitAs.facet.162 [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.as_type.as.ImplicitAs.impl.Convert.d9f, @U.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.9ba, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_26.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6c5: %SubobjectDestroy.type = facet_value %Optional.37f, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.22a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6c5) [concrete]
// CHECK:STDOUT:   %Destroy.facet.9ec: %Destroy.type = facet_value %Optional.37f, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.22a) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.935: %SubobjectDestroy.type = facet_value %NullableParamS, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.5da: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.935) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f94: %Destroy.type = facet_value %NullableParamS, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.5da) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.30b: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.dac)]
// CHECK:STDOUT:   %Core.import_ref.814 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.470: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.b0c) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.bf1)]
// CHECK:STDOUT:   %Core.import_ref.322 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0e8 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.713 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.30b, %Core.import_ref.814, %Core.import_ref.470, %Core.import_ref.322, %Core.import_ref.0e8, %Core.import_ref.713), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b53: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b53), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %nullable_param_foo.decl: %nullable_param_foo.type = fn_decl @nullable_param_foo [concrete = constants.%nullable_param_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc9_28.1: type = splice_block %Optional [concrete = constants.%Optional.37f] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.1f0, (constants.%OptionalStorage.impl_witness.b77) [concrete = constants.%OptionalStorage.facet.9ba]
// CHECK:STDOUT:       %.loc9_28.2: %OptionalStorage.type = converted constants.%ptr.1f0, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.9ba]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.9ba) [concrete = constants.%Optional.37f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.098: @U.as_type.as.ImplicitAs.impl.0c0.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.c47) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.0c0.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.f83)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.653 = impl_witness_table (%Core.import_ref.098), @U.as_type.as.ImplicitAs.impl.0c0 [concrete]
// CHECK:STDOUT:   %Core.import_ref.49b: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.b2a) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.c36)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.98f = impl_witness_table (%Core.import_ref.49b), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref %NullableParamS = var_storage %s.var_patt
// CHECK:STDOUT:   %.loc8_32.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_32.2: init %NullableParamS to %s.var = class_init () [concrete = constants.%NullableParamS.val]
// CHECK:STDOUT:   %.loc8_3.1: init %NullableParamS = converted %.loc8_32.1, %.loc8_32.2 [concrete = constants.%NullableParamS.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3.1
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
// CHECK:STDOUT:   %impl.elem0: %.94f = impl_witness_access constants.%ImplicitAs.impl_witness.940, element0 [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.d9f]
// CHECK:STDOUT:   %bound_method.loc9_26.1: <bound method> = bound_method %addr, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @U.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.9ba, constants.%OptionalAs.facet) [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_26.2: <bound method> = bound_method %addr, %specific_fn
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.37f = call %bound_method.loc9_26.2(%addr)
// CHECK:STDOUT:   %.loc9_26.1: init %Optional.37f = converted %addr, %U.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc9_26.2: ref %Optional.37f = temporary_storage
// CHECK:STDOUT:   %.loc9_26.3: ref %Optional.37f = temporary %.loc9_26.2, %.loc9_26.1
// CHECK:STDOUT:   %.loc9_26.4: %Optional.37f = acquire_value %.loc9_26.3
// CHECK:STDOUT:   %nullable_param_foo.call: init %empty_tuple.type = call imports.%nullable_param_foo.decl(%.loc9_26.4)
// CHECK:STDOUT:   %Destroy.facet.loc9_26.1: %Destroy.type = facet_value constants.%Optional.37f, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.22a) [concrete = constants.%Destroy.facet.9ec]
// CHECK:STDOUT:   %.loc9_26.5: %Destroy.type = converted constants.%Optional.37f, %Destroy.facet.loc9_26.1 [concrete = constants.%Destroy.facet.9ec]
// CHECK:STDOUT:   %Destroy.facet.loc9_26.2: %Destroy.type = facet_value constants.%Optional.37f, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.22a) [concrete = constants.%Destroy.facet.9ec]
// CHECK:STDOUT:   %.loc9_26.6: %Destroy.type = converted constants.%Optional.37f, %Destroy.facet.loc9_26.2 [concrete = constants.%Destroy.facet.9ec]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc9_26.3)
// CHECK:STDOUT:   %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%NullableParamS, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.5da) [concrete = constants.%Destroy.facet.f94]
// CHECK:STDOUT:   %.loc8_3.2: %Destroy.type = converted constants.%NullableParamS, %Destroy.facet.loc8_3.1 [concrete = constants.%Destroy.facet.f94]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%NullableParamS, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.5da) [concrete = constants.%Destroy.facet.f94]
// CHECK:STDOUT:   %.loc8_3.3: %Destroy.type = converted constants.%NullableParamS, %Destroy.facet.loc8_3.2 [concrete = constants.%Destroy.facet.f94]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc8: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_26.1(%self.param: ref %.280) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_26.2(%self.param: ref %MaybeUnformed.b5a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_26.3(%self.param: ref %Optional.37f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %NullableParamS) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- null_pointer_arg_to_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %nullable_param_foo.cpp_overload_set.type: type = cpp_overload_set_type @nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %nullable_param_foo.cpp_overload_set.value: %nullable_param_foo.cpp_overload_set.type = cpp_overload_set_value @nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Optional.type: type = parameterized_class_name_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.None.type.626: type = fn_type @Optional.None, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.None.3ad: %Optional.None.type.626 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.61e: <witness> = lookup_impl_witness %ptr.e8f, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.18b: <witness> = lookup_impl_witness %ptr.e8f, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.5f3: %Destroy.type = facet_value %ptr.e8f, (%SubobjectDestroy.lookup_impl_witness.61e, %DestroyImpl.lookup_impl_witness.18b) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.dac: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.5f3) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.type.629: type = fn_type @ptr.as.OptionalStorage.impl.None, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.f84: %ptr.as.OptionalStorage.impl.None.type.629 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %NullableParamS: type = class_type @NullableParamS [concrete]
// CHECK:STDOUT:   %ptr.1f0: type = ptr_type %NullableParamS [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a1d: %SubobjectDestroy.type = facet_value %ptr.1f0, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.82b: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a1d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.6e0: %Destroy.type = facet_value %ptr.1f0, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.82b) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.b5a: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.6e0) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.b77: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%NullableParamS) [concrete]
// CHECK:STDOUT:   %.280: type = maybe_unformed_type %ptr.1f0 [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.1f0, (%OptionalStorage.impl_witness.b77) [concrete]
// CHECK:STDOUT:   %Optional.37f: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.type.2bc: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.7f6: %Optional.None.type.2bc = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %Optional.None.7f6, @Optional.None(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %nullable_param_foo.type: type = fn_type @nullable_param_foo [concrete]
// CHECK:STDOUT:   %nullable_param_foo: %nullable_param_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8_66.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6c5: %SubobjectDestroy.type = facet_value %Optional.37f, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.22a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6c5) [concrete]
// CHECK:STDOUT:   %Destroy.facet.9ec: %Destroy.type = facet_value %Optional.37f, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.22a) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
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
// CHECK:STDOUT:   %Core.import_ref.30b: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.dac)]
// CHECK:STDOUT:   %Core.import_ref.9ff: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None.type (%ptr.as.OptionalStorage.impl.None.type.629) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None (constants.%ptr.as.OptionalStorage.impl.None.f84)]
// CHECK:STDOUT:   %Core.import_ref.a1a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.322 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0e8 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.713 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.30b, %Core.import_ref.9ff, %Core.import_ref.a1a, %Core.import_ref.322, %Core.import_ref.0e8, %Core.import_ref.713), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b5: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b5), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d6d: @Optional.%Optional.None.type (%Optional.None.type.626) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @Optional.%Optional.None (constants.%Optional.None.3ad)]
// CHECK:STDOUT:   %NullableParamS.decl: type = class_decl @NullableParamS [concrete = constants.%NullableParamS] {} {}
// CHECK:STDOUT:   %nullable_param_foo.decl: %nullable_param_foo.type = fn_decl @nullable_param_foo [concrete = constants.%nullable_param_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc8_67.1: type = splice_block %Optional [concrete = constants.%Optional.37f] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.1f0, (constants.%OptionalStorage.impl_witness.b77) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %.loc8_67.2: %OptionalStorage.type = converted constants.%ptr.1f0, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.37f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
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
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.b77) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc8_59: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.37f]
// CHECK:STDOUT:   %.loc8_60: %Optional.None.type.2bc = specific_constant imports.%Core.import_ref.d6d, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.7f6]
// CHECK:STDOUT:   %None.ref: %Optional.None.type.2bc = name_ref None, %.loc8_60 [concrete = constants.%Optional.None.7f6]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %None.ref, @Optional.None(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.specific_fn]
// CHECK:STDOUT:   %Optional.None.call: init %Optional.37f = call %Optional.None.specific_fn()
// CHECK:STDOUT:   %.loc8_66.1: ref %Optional.37f = temporary_storage
// CHECK:STDOUT:   %.loc8_66.2: ref %Optional.37f = temporary %.loc8_66.1, %Optional.None.call
// CHECK:STDOUT:   %.loc8_66.3: %Optional.37f = acquire_value %.loc8_66.2
// CHECK:STDOUT:   %nullable_param_foo.call: init %empty_tuple.type = call imports.%nullable_param_foo.decl(%.loc8_66.3)
// CHECK:STDOUT:   %Destroy.facet.loc8_66.1: %Destroy.type = facet_value constants.%Optional.37f, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.22a) [concrete = constants.%Destroy.facet.9ec]
// CHECK:STDOUT:   %.loc8_66.4: %Destroy.type = converted constants.%Optional.37f, %Destroy.facet.loc8_66.1 [concrete = constants.%Destroy.facet.9ec]
// CHECK:STDOUT:   %Destroy.facet.loc8_66.2: %Destroy.type = facet_value constants.%Optional.37f, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.22a) [concrete = constants.%Destroy.facet.9ec]
// CHECK:STDOUT:   %.loc8_66.5: %Destroy.type = converted constants.%Optional.37f, %Destroy.facet.loc8_66.2 [concrete = constants.%Destroy.facet.9ec]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc8_66.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_66.1(%self.param: ref %.280) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_66.2(%self.param: ref %MaybeUnformed.b5a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_66.3(%self.param: ref %Optional.37f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- nonnull_pointer_arg_to_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %Optional.type: type = parameterized_class_name_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.63b: type = fn_type @Optional.Some, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Some.792: %Optional.Some.type.63b = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.61e: <witness> = lookup_impl_witness %ptr.e8f, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.18b: <witness> = lookup_impl_witness %ptr.e8f, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.5f3: %Destroy.type = facet_value %ptr.e8f, (%SubobjectDestroy.lookup_impl_witness.61e, %DestroyImpl.lookup_impl_witness.18b) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.dac: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.5f3) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.b0c: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.bf1: %ptr.as.OptionalStorage.impl.Some.type.b0c = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.1f0: type = ptr_type %NullableParamS [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a1d: %SubobjectDestroy.type = facet_value %ptr.1f0, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.82b: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a1d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.6e0: %Destroy.type = facet_value %ptr.1f0, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.82b) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.b5a: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.6e0) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.b77: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%NullableParamS) [concrete]
// CHECK:STDOUT:   %.280: type = maybe_unformed_type %ptr.1f0 [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.1f0, (%OptionalStorage.impl_witness.b77) [concrete]
// CHECK:STDOUT:   %Optional.37f: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.Some.type.683: type = fn_type @Optional.Some, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.Some.fff: %Optional.Some.type.683 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Some.specific_fn: <specific function> = specific_function %Optional.Some.fff, @Optional.Some(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %nullable_param_foo.type: type = fn_type @nullable_param_foo [concrete]
// CHECK:STDOUT:   %nullable_param_foo: %nullable_param_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_68.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6c5: %SubobjectDestroy.type = facet_value %Optional.37f, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.22a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6c5) [concrete]
// CHECK:STDOUT:   %Destroy.facet.9ec: %Destroy.type = facet_value %Optional.37f, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.22a) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.935: %SubobjectDestroy.type = facet_value %NullableParamS, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.5da: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.935) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f94: %Destroy.type = facet_value %NullableParamS, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.5da) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
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
// CHECK:STDOUT:   %Core.import_ref.30b: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.dac)]
// CHECK:STDOUT:   %Core.import_ref.814 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.470: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.b0c) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.bf1)]
// CHECK:STDOUT:   %Core.import_ref.322 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0e8 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.713 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.30b, %Core.import_ref.814, %Core.import_ref.470, %Core.import_ref.322, %Core.import_ref.0e8, %Core.import_ref.713), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b5: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b5), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.9b4: @Optional.%Optional.Some.type (%Optional.Some.type.63b) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @Optional.%Optional.Some (constants.%Optional.Some.792)]
// CHECK:STDOUT:   %nullable_param_foo.decl: %nullable_param_foo.type = fn_decl @nullable_param_foo [concrete = constants.%nullable_param_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc9_69.1: type = splice_block %Optional [concrete = constants.%Optional.37f] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.1f0, (constants.%OptionalStorage.impl_witness.b77) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %.loc9_69.2: %OptionalStorage.type = converted constants.%ptr.1f0, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.37f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref %NullableParamS = var_storage %s.var_patt
// CHECK:STDOUT:   %.loc8_32.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_32.2: init %NullableParamS to %s.var = class_init () [concrete = constants.%NullableParamS.val]
// CHECK:STDOUT:   %.loc8_3.1: init %NullableParamS = converted %.loc8_32.1, %.loc8_32.2 [concrete = constants.%NullableParamS.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3.1
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
// CHECK:STDOUT:   %OptionalStorage.facet.loc9_59: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.b77) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc9_59: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet.loc9_59 [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.37f]
// CHECK:STDOUT:   %.loc9_60: %Optional.Some.type.683 = specific_constant imports.%Core.import_ref.9b4, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.Some.fff]
// CHECK:STDOUT:   %Some.ref: %Optional.Some.type.683 = name_ref Some, %.loc9_60 [concrete = constants.%Optional.Some.fff]
// CHECK:STDOUT:   %s.ref: ref %NullableParamS = name_ref s, %s
// CHECK:STDOUT:   %addr: %ptr.1f0 = addr_of %s.ref
// CHECK:STDOUT:   %OptionalStorage.facet.loc9_68.1: %OptionalStorage.type = facet_value constants.%ptr.1f0, (constants.%OptionalStorage.impl_witness.b77) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc9_68.1: %OptionalStorage.type = converted constants.%ptr.1f0, %OptionalStorage.facet.loc9_68.1 [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %OptionalStorage.facet.loc9_68.2: %OptionalStorage.type = facet_value constants.%ptr.1f0, (constants.%OptionalStorage.impl_witness.b77) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc9_68.2: %OptionalStorage.type = converted constants.%ptr.1f0, %OptionalStorage.facet.loc9_68.2 [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %Optional.Some.specific_fn: <specific function> = specific_function %Some.ref, @Optional.Some(constants.%OptionalStorage.facet) [concrete = constants.%Optional.Some.specific_fn]
// CHECK:STDOUT:   %Optional.Some.call: init %Optional.37f = call %Optional.Some.specific_fn(%addr)
// CHECK:STDOUT:   %.loc9_68.3: ref %Optional.37f = temporary_storage
// CHECK:STDOUT:   %.loc9_68.4: ref %Optional.37f = temporary %.loc9_68.3, %Optional.Some.call
// CHECK:STDOUT:   %.loc9_68.5: %Optional.37f = acquire_value %.loc9_68.4
// CHECK:STDOUT:   %nullable_param_foo.call: init %empty_tuple.type = call imports.%nullable_param_foo.decl(%.loc9_68.5)
// CHECK:STDOUT:   %Destroy.facet.loc9_68.1: %Destroy.type = facet_value constants.%Optional.37f, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.22a) [concrete = constants.%Destroy.facet.9ec]
// CHECK:STDOUT:   %.loc9_68.6: %Destroy.type = converted constants.%Optional.37f, %Destroy.facet.loc9_68.1 [concrete = constants.%Destroy.facet.9ec]
// CHECK:STDOUT:   %Destroy.facet.loc9_68.2: %Destroy.type = facet_value constants.%Optional.37f, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.22a) [concrete = constants.%Destroy.facet.9ec]
// CHECK:STDOUT:   %.loc9_68.7: %Destroy.type = converted constants.%Optional.37f, %Destroy.facet.loc9_68.2 [concrete = constants.%Destroy.facet.9ec]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc9_68.4)
// CHECK:STDOUT:   %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%NullableParamS, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.5da) [concrete = constants.%Destroy.facet.f94]
// CHECK:STDOUT:   %.loc8_3.2: %Destroy.type = converted constants.%NullableParamS, %Destroy.facet.loc8_3.1 [concrete = constants.%Destroy.facet.f94]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%NullableParamS, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.5da) [concrete = constants.%Destroy.facet.f94]
// CHECK:STDOUT:   %.loc8_3.3: %Destroy.type = converted constants.%NullableParamS, %Destroy.facet.loc8_3.2 [concrete = constants.%Destroy.facet.f94]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc8: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_68.1(%self.param: ref %.280) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_68.2(%self.param: ref %MaybeUnformed.b5a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_68.3(%self.param: ref %Optional.37f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %NullableParamS) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- forward_nullable_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %nullable_param_foo.cpp_overload_set.type: type = cpp_overload_set_type @nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %nullable_param_foo.cpp_overload_set.value: %nullable_param_foo.cpp_overload_set.type = cpp_overload_set_value @nullable_param_foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %nullable_param_get.cpp_overload_set.type: type = cpp_overload_set_type @nullable_param_get.cpp_overload_set [concrete]
// CHECK:STDOUT:   %nullable_param_get.cpp_overload_set.value: %nullable_param_get.cpp_overload_set.type = cpp_overload_set_value @nullable_param_get.cpp_overload_set [concrete]
// CHECK:STDOUT:   %NullableParamS: type = class_type @NullableParamS [concrete]
// CHECK:STDOUT:   %ptr.1f0: type = ptr_type %NullableParamS [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.61e: <witness> = lookup_impl_witness %ptr.e8f, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.18b: <witness> = lookup_impl_witness %ptr.e8f, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.5f3: %Destroy.type = facet_value %ptr.e8f, (%SubobjectDestroy.lookup_impl_witness.61e, %DestroyImpl.lookup_impl_witness.18b) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.dac: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.5f3) [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a1d: %SubobjectDestroy.type = facet_value %ptr.1f0, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.82b: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a1d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.6e0: %Destroy.type = facet_value %ptr.1f0, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.82b) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.b5a: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.6e0) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.b77: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%NullableParamS) [concrete]
// CHECK:STDOUT:   %.280: type = maybe_unformed_type %ptr.1f0 [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.1f0, (%OptionalStorage.impl_witness.b77) [concrete]
// CHECK:STDOUT:   %Optional.37f: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %nullable_param_get.type: type = fn_type @nullable_param_get [concrete]
// CHECK:STDOUT:   %nullable_param_get: %nullable_param_get.type = struct_value () [concrete]
// CHECK:STDOUT:   %nullable_param_foo.type: type = fn_type @nullable_param_foo [concrete]
// CHECK:STDOUT:   %nullable_param_foo: %nullable_param_foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8_49.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6c5: %SubobjectDestroy.type = facet_value %Optional.37f, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.22a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6c5) [concrete]
// CHECK:STDOUT:   %Destroy.facet.9ec: %Destroy.type = facet_value %Optional.37f, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.22a) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.30b: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.dac)]
// CHECK:STDOUT:   %Core.import_ref.814 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.a1a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.322 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0e8 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.713 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.30b, %Core.import_ref.814, %Core.import_ref.a1a, %Core.import_ref.322, %Core.import_ref.0e8, %Core.import_ref.713), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b5: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b5), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %nullable_param_get.decl: %nullable_param_get.type = fn_decl @nullable_param_get [concrete = constants.%nullable_param_get] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.1f0, (constants.%OptionalStorage.impl_witness.b77) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %.loc8: %OptionalStorage.type = converted constants.%ptr.1f0, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.37f]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nullable_param_foo.decl: %nullable_param_foo.type = fn_decl @nullable_param_foo [concrete = constants.%nullable_param_foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc8_50.1: type = splice_block %Optional [concrete = constants.%Optional.37f] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.1f0, (constants.%OptionalStorage.impl_witness.b77) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %.loc8_50.2: %OptionalStorage.type = converted constants.%ptr.1f0, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.37f]
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
// CHECK:STDOUT:   %nullable_param_get.call: init %Optional.37f = call imports.%nullable_param_get.decl()
// CHECK:STDOUT:   %.loc8_49.1: ref %Optional.37f = temporary_storage
// CHECK:STDOUT:   %.loc8_49.2: ref %Optional.37f = temporary %.loc8_49.1, %nullable_param_get.call
// CHECK:STDOUT:   %.loc8_49.3: %Optional.37f = acquire_value %.loc8_49.2
// CHECK:STDOUT:   %nullable_param_foo.call: init %empty_tuple.type = call imports.%nullable_param_foo.decl(%.loc8_49.3)
// CHECK:STDOUT:   %Destroy.facet.loc8_49.1: %Destroy.type = facet_value constants.%Optional.37f, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.22a) [concrete = constants.%Destroy.facet.9ec]
// CHECK:STDOUT:   %.loc8_49.4: %Destroy.type = converted constants.%Optional.37f, %Destroy.facet.loc8_49.1 [concrete = constants.%Destroy.facet.9ec]
// CHECK:STDOUT:   %Destroy.facet.loc8_49.2: %Destroy.type = facet_value constants.%Optional.37f, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.22a) [concrete = constants.%Destroy.facet.9ec]
// CHECK:STDOUT:   %.loc8_49.5: %Destroy.type = converted constants.%Optional.37f, %Destroy.facet.loc8_49.2 [concrete = constants.%Destroy.facet.9ec]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc8_49.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_49.1(%self.param: ref %.280) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_49.2(%self.param: ref %MaybeUnformed.b5a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_49.3(%self.param: ref %Optional.37f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_deduced_any_param_as_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c9e: %SubobjectDestroy.type = facet_value %DeducedAnyS, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.296: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c9e) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %DeducedAnyS, (%DestroyImpl.impl_witness.296, %custom_witness.b80) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref %DeducedAnyS = var_storage %s.var_patt
// CHECK:STDOUT:   %.loc8_29.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_29.2: init %DeducedAnyS to %s.var = class_init () [concrete = constants.%DeducedAnyS.val]
// CHECK:STDOUT:   %.loc8_3.1: init %DeducedAnyS = converted %.loc8_29.1, %.loc8_29.2 [concrete = constants.%DeducedAnyS.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3.1
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
// CHECK:STDOUT:   %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%DeducedAnyS, (constants.%DestroyImpl.impl_witness.296, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_3.2: %Destroy.type = converted constants.%DeducedAnyS, %Destroy.facet.loc8_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%DeducedAnyS, (constants.%DestroyImpl.impl_witness.296, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_3.3: %Destroy.type = converted constants.%DeducedAnyS, %Destroy.facet.loc8_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %DeducedAnyS) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_deduced_pointer_param_as_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %Optional.type: type = parameterized_class_name_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.220e40.1: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.61e: <witness> = lookup_impl_witness %ptr.e8f, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.18b: <witness> = lookup_impl_witness %ptr.e8f, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.5f3: %Destroy.type = facet_value %ptr.e8f, (%SubobjectDestroy.lookup_impl_witness.61e, %DestroyImpl.lookup_impl_witness.18b) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.dac: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.5f3) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.b0c: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.bf1: %ptr.as.OptionalStorage.impl.Some.type.b0c = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.cbc: %SubobjectDestroy.type = facet_value %ptr.6cb, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a4f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.cbc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.bf2: %Destroy.type = facet_value %ptr.6cb, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.a4f) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.fc9: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.bf2) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.9df: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%DeducedPointerS) [concrete]
// CHECK:STDOUT:   %.fb1: type = maybe_unformed_type %ptr.6cb [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.b3a: %OptionalStorage.type = facet_value %ptr.6cb, (%OptionalStorage.impl_witness.9df) [concrete]
// CHECK:STDOUT:   %Optional.1e6: type = class_type @Optional, @Optional(%OptionalStorage.facet.b3a) [concrete]
// CHECK:STDOUT:   %pattern_type.7e1: type = pattern_type %Optional.1e6 [concrete]
// CHECK:STDOUT:   %Direct.type: type = fn_type @Direct [concrete]
// CHECK:STDOUT:   %Direct: %Direct.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b63: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.1e6)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.5bd: type = facet_type <@OptionalAs, @OptionalAs(%T.220e40.1)> [symbolic]
// CHECK:STDOUT:   %U.c66: %OptionalAs.type.5bd = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.c47: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.0c0(%T.220e40.1, %U.c66) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.f83: %U.as_type.as.ImplicitAs.impl.Convert.type.c47 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.51c: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.b3a)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.b2a: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.220e40.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.c36: %T.as_type.as.OptionalAs.impl.Convert.type.b2a = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.957: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.98f, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.b3a) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.51c = facet_value %ptr.6cb, (%OptionalAs.impl_witness.957) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f71: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.653, @U.as_type.as.ImplicitAs.impl.0c0(%OptionalStorage.facet.b3a, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.5ca: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.0c0(%OptionalStorage.facet.b3a, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.c66: %U.as_type.as.ImplicitAs.impl.Convert.type.5ca = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.25a: %ImplicitAs.type.b63 = facet_value %ptr.6cb, (%ImplicitAs.impl_witness.f71) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c06: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.1e6, %ImplicitAs.facet.25a) [concrete]
// CHECK:STDOUT:   %.d55: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.c06, %ImplicitAs.facet.25a [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.as_type.as.ImplicitAs.impl.Convert.c66, @U.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.b3a, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_14.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.975: %SubobjectDestroy.type = facet_value %Optional.1e6, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.462: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.975) [concrete]
// CHECK:STDOUT:   %Destroy.facet.e27: %Destroy.type = facet_value %Optional.1e6, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.462) [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.7e1 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %Indirect.cpp_overload_set.type: type = cpp_overload_set_type @Indirect.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Indirect.cpp_overload_set.value: %Indirect.cpp_overload_set.type = cpp_overload_set_value @Indirect.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Indirect__carbon_thunk.type: type = fn_type @Indirect__carbon_thunk [concrete]
// CHECK:STDOUT:   %Indirect__carbon_thunk: %Indirect__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.dc4: ref %DeducedPointerS = temporary invalid, %DeducedPointerS.val [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.be3: %SubobjectDestroy.type = facet_value %DeducedPointerS, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.0ba: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.be3) [concrete]
// CHECK:STDOUT:   %Destroy.facet.5d5: %Destroy.type = facet_value %DeducedPointerS, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.0ba) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
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
// CHECK:STDOUT:   %Core.import_ref.30b: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.dac)]
// CHECK:STDOUT:   %Core.import_ref.814 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.470: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.b0c) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.bf1)]
// CHECK:STDOUT:   %Core.import_ref.322 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0e8 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.713 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.30b, %Core.import_ref.814, %Core.import_ref.470, %Core.import_ref.322, %Core.import_ref.0e8, %Core.import_ref.713), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b53: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b53), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Direct.decl: %Direct.type = fn_decl @Direct [concrete = constants.%Direct] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %OptionalStorage.facet.loc11_16.1: %OptionalStorage.type = facet_value constants.%ptr.6cb, (constants.%OptionalStorage.impl_witness.9df) [concrete = constants.%OptionalStorage.facet.b3a]
// CHECK:STDOUT:     %.loc11_16.1: %OptionalStorage.type = converted constants.%ptr.6cb, %OptionalStorage.facet.loc11_16.1 [concrete = constants.%OptionalStorage.facet.b3a]
// CHECK:STDOUT:     %Optional.loc11_16.1: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.b3a) [concrete = constants.%Optional.1e6]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_16.2: type = splice_block %Optional.loc11_16.2 [concrete = constants.%Optional.1e6] {
// CHECK:STDOUT:       %OptionalStorage.facet.loc11_16.2: %OptionalStorage.type = facet_value constants.%ptr.6cb, (constants.%OptionalStorage.impl_witness.9df) [concrete = constants.%OptionalStorage.facet.b3a]
// CHECK:STDOUT:       %.loc11_16.3: %OptionalStorage.type = converted constants.%ptr.6cb, %OptionalStorage.facet.loc11_16.2 [concrete = constants.%OptionalStorage.facet.b3a]
// CHECK:STDOUT:       %Optional.loc11_16.2: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.b3a) [concrete = constants.%Optional.1e6]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.098: @U.as_type.as.ImplicitAs.impl.0c0.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.c47) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.0c0.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.f83)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.653 = impl_witness_table (%Core.import_ref.098), @U.as_type.as.ImplicitAs.impl.0c0 [concrete]
// CHECK:STDOUT:   %Core.import_ref.49b: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.b2a) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.c36)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.98f = impl_witness_table (%Core.import_ref.49b), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Indirect.cpp_overload_set.value: %Indirect.cpp_overload_set.type = cpp_overload_set_value @Indirect.cpp_overload_set [concrete = constants.%Indirect.cpp_overload_set.value]
// CHECK:STDOUT:   %Indirect__carbon_thunk.decl: %Indirect__carbon_thunk.type = fn_decl @Indirect__carbon_thunk [concrete = constants.%Indirect__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.6cb, (constants.%OptionalStorage.impl_witness.9df) [concrete = constants.%OptionalStorage.facet.b3a]
// CHECK:STDOUT:     %.loc13: %OptionalStorage.type = converted constants.%ptr.6cb, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.b3a]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.b3a) [concrete = constants.%Optional.1e6]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.var: ref %DeducedPointerS = var_storage %s.var_patt
// CHECK:STDOUT:   %.loc10_33.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_33.2: init %DeducedPointerS to %s.var = class_init () [concrete = constants.%DeducedPointerS.val]
// CHECK:STDOUT:   %.loc10_3.1: init %DeducedPointerS = converted %.loc10_33.1, %.loc10_33.2 [concrete = constants.%DeducedPointerS.val]
// CHECK:STDOUT:   assign %s.var, %.loc10_3.1
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
// CHECK:STDOUT:   %impl.elem0: %.d55 = impl_witness_access constants.%ImplicitAs.impl_witness.f71, element0 [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.c66]
// CHECK:STDOUT:   %bound_method.loc11_14.1: <bound method> = bound_method %addr.loc11, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @U.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.b3a, constants.%OptionalAs.facet) [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_14.2: <bound method> = bound_method %addr.loc11, %specific_fn
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.1e6 = call %bound_method.loc11_14.2(%addr.loc11)
// CHECK:STDOUT:   %.loc11_14.1: init %Optional.1e6 = converted %addr.loc11, %U.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_14.2: ref %Optional.1e6 = temporary_storage
// CHECK:STDOUT:   %.loc11_14.3: ref %Optional.1e6 = temporary %.loc11_14.2, %.loc11_14.1
// CHECK:STDOUT:   %.loc11_14.4: %Optional.1e6 = acquire_value %.loc11_14.3
// CHECK:STDOUT:   %Direct.call: init %Optional.1e6 = call imports.%Direct.decl(%.loc11_14.4)
// CHECK:STDOUT:   %Destroy.facet.loc11_14.1: %Destroy.type = facet_value constants.%Optional.1e6, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.462) [concrete = constants.%Destroy.facet.e27]
// CHECK:STDOUT:   %.loc11_14.5: %Destroy.type = converted constants.%Optional.1e6, %Destroy.facet.loc11_14.1 [concrete = constants.%Destroy.facet.e27]
// CHECK:STDOUT:   %Destroy.facet.loc11_14.2: %Destroy.type = facet_value constants.%Optional.1e6, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.462) [concrete = constants.%Destroy.facet.e27]
// CHECK:STDOUT:   %.loc11_14.6: %Destroy.type = converted constants.%Optional.1e6, %Destroy.facet.loc11_14.2 [concrete = constants.%Destroy.facet.e27]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc11_14.3)
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
// CHECK:STDOUT:   %Indirect__carbon_thunk.call: init %Optional.1e6 = call imports.%Indirect__carbon_thunk.decl(%addr.loc13)
// CHECK:STDOUT:   %.loc13_93.1: ref %Optional.1e6 = temporary_storage
// CHECK:STDOUT:   %.loc13_93.2: ref %Optional.1e6 = temporary %.loc13_93.1, %Indirect__carbon_thunk.call
// CHECK:STDOUT:   %.loc13_93.3: %Optional.1e6 = acquire_value %.loc13_93.2
// CHECK:STDOUT:   %.loc13_51.1: type = splice_block %Optional [concrete = constants.%Optional.1e6] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Cpp.ref.loc13_31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DeducedPointerS.ref.loc13_34: type = name_ref DeducedPointerS, imports.%DeducedPointerS.decl [concrete = constants.%DeducedPointerS]
// CHECK:STDOUT:     %ptr: type = ptr_type %DeducedPointerS.ref.loc13_34 [concrete = constants.%ptr.6cb]
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.9df) [concrete = constants.%OptionalStorage.facet.b3a]
// CHECK:STDOUT:     %.loc13_51.2: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.b3a]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.b3a) [concrete = constants.%Optional.1e6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %Optional.1e6 = wrapper_binding a, %.loc13_93.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7e1 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_93.1: %Destroy.type = facet_value constants.%Optional.1e6, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.462) [concrete = constants.%Destroy.facet.e27]
// CHECK:STDOUT:   %.loc13_93.4: %Destroy.type = converted constants.%Optional.1e6, %Destroy.facet.loc13_93.1 [concrete = constants.%Destroy.facet.e27]
// CHECK:STDOUT:   %Destroy.facet.loc13_93.2: %Destroy.type = facet_value constants.%Optional.1e6, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.462) [concrete = constants.%Destroy.facet.e27]
// CHECK:STDOUT:   %.loc13_93.5: %Destroy.type = converted constants.%Optional.1e6, %Destroy.facet.loc13_93.2 [concrete = constants.%Destroy.facet.e27]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13_93: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc13_93.2)
// CHECK:STDOUT:   %Destroy.facet.loc13_71.1: %Destroy.type = facet_value constants.%DeducedPointerS, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.0ba) [concrete = constants.%Destroy.facet.5d5]
// CHECK:STDOUT:   %.loc13_71.5: %Destroy.type = converted constants.%DeducedPointerS, %Destroy.facet.loc13_71.1 [concrete = constants.%Destroy.facet.5d5]
// CHECK:STDOUT:   %Destroy.facet.loc13_71.2: %Destroy.type = facet_value constants.%DeducedPointerS, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.0ba) [concrete = constants.%Destroy.facet.5d5]
// CHECK:STDOUT:   %.loc13_71.6: %Destroy.type = converted constants.%DeducedPointerS, %Destroy.facet.loc13_71.2 [concrete = constants.%Destroy.facet.5d5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13_71: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc13_71.2)
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%DeducedPointerS, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.0ba) [concrete = constants.%Destroy.facet.5d5]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%DeducedPointerS, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet.5d5]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%DeducedPointerS, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.0ba) [concrete = constants.%Destroy.facet.5d5]
// CHECK:STDOUT:   %.loc10_3.3: %Destroy.type = converted constants.%DeducedPointerS, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet.5d5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_14.1(%self.param: ref %.fb1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_14.2(%self.param: ref %MaybeUnformed.fc9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_14.3(%self.param: ref %Optional.1e6) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13(%self.param: ref %DeducedPointerS) = "no_op";
// CHECK:STDOUT:
