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

struct S {};

auto foo(S* _Nonnull) -> void;

// --- import_non_nullable_pointer_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_nullable_pointer_param.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.S = {};
  Cpp.foo(&s);
  //@dump-sem-ir-end
}

// --- fail_import_non_nullable_pointer_param_using_const_value.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_nullable_pointer_param.h";

fn G() -> const Cpp.S;

fn F() {
  var s: const Cpp.S = G();
  // CHECK:STDERR: fail_import_non_nullable_pointer_param_using_const_value.carbon:[[@LINE+8]]:13: error: no matching function for call to 'foo' [CppInteropParseError]
  // CHECK:STDERR:    18 |   Cpp.foo(&s);
  // CHECK:STDERR:       |             ^
  // CHECK:STDERR: fail_import_non_nullable_pointer_param_using_const_value.carbon:[[@LINE-9]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./non_nullable_pointer_param.h:4:6: note: candidate function not viable: 1st argument ('const S * _Nonnull') would lose const qualifier [CppInteropParseNote]
  // CHECK:STDERR:     4 | auto foo(S* _Nonnull) -> void;
  // CHECK:STDERR:       |      ^   ~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo(&s);
}

// --- import_non_nullable_pointer_param_using_const_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_nullable_pointer_param.h";

fn G() -> const (Cpp.S*);

fn F() {
  //@dump-sem-ir-begin
  var p: const (Cpp.S*) = G();
  Cpp.foo(p);
  //@dump-sem-ir-end
}

// ============================================================================
// Double non-nullable pointer as a parameter type
// ============================================================================

// --- double_non_nullable_pointer_param.h

struct S {};

auto foo(S* _Nonnull * _Nonnull) -> void;

// --- import_double_non_nullable_pointer_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "double_non_nullable_pointer_param.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.S = {};
  var p: Cpp.S* = &s;
  Cpp.foo(&p);
  //@dump-sem-ir-end
}

// ============================================================================
// Non-nullable pointer to const as a parameter type
// ============================================================================

// --- non_nullable_pointer_to_const_param.h

struct S {};

auto foo(const S* _Nonnull) -> void;

// --- import_non_nullable_pointer_to_const_param_using_const.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_nullable_pointer_to_const_param.h";

fn G() -> const Cpp.S;

fn F() {
  //@dump-sem-ir-begin
  var s: const Cpp.S = G();
  Cpp.foo(&s);
  //@dump-sem-ir-end
}

// --- import_non_nullable_pointer_to_const_param_using_non_const.carbon

import Cpp library "non_nullable_pointer_to_const_param.h";

fn G() -> Cpp.S;

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.S = G();
  Cpp.foo(&s);
  //@dump-sem-ir-end
}

// ============================================================================
// Const non-nullable pointer as a parameter type
// ============================================================================

// --- const_non_nullable_pointer_param.h

struct S {};

auto foo(S* _Nonnull const) -> void;

// --- import_const_non_nullable_pointer_param_using_const.carbon

library "[[@TEST_NAME]]";

import Cpp library "const_non_nullable_pointer_param.h";

fn G() -> const (Cpp.S*);

fn F() {
  //@dump-sem-ir-begin
  var p: const (Cpp.S*) = G();
  Cpp.foo(p);
  //@dump-sem-ir-end
}

// --- import_const_non_nullable_pointer_param_using_non_const.carbon

library "[[@TEST_NAME]]";

import Cpp library "const_non_nullable_pointer_param.h";

fn G() -> Cpp.S*;

fn F() {
  //@dump-sem-ir-begin
  var p: Cpp.S* = G();
  Cpp.foo(p);
  //@dump-sem-ir-end
}

// ============================================================================
// Const nullable pointer as a parameter type
// ============================================================================

// --- const_nullable_pointer_param.h

struct S {};

auto foo(S* const) -> void;

// --- import_const_nullable_pointer_param_using_const.carbon

library "[[@TEST_NAME]]";

import Cpp library "const_nullable_pointer_param.h";

fn G() -> const (Cpp.S*);

fn F() {
  //@dump-sem-ir-begin
  var p: const (Cpp.S*) = G();
  Cpp.foo(p);
  //@dump-sem-ir-end
}

// --- import_const_nullable_pointer_param_using_non_const.carbon

library "[[@TEST_NAME]]";

import Cpp library "const_nullable_pointer_param.h";

fn G() -> Cpp.S*;

fn F() {
  //@dump-sem-ir-begin
  var p: Cpp.S* = G();
  Cpp.foo(p);
  //@dump-sem-ir-end
}

// ============================================================================
// Non-nullable pointer as a return value
// ============================================================================

// --- non_nullable_pointer_return.h

struct S {};

auto foo() -> S* _Nonnull;

// --- import_non_nullable_pointer_return.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_nullable_pointer_return.h";

fn IngestDoublePointer(s: Cpp.S*);

fn F() {
  //@dump-sem-ir-begin
  IngestDoublePointer(Cpp.foo());
  Cpp.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Double non-nullable pointer as a return value
// ============================================================================

// --- double_non_nullable_pointer_return.h

struct S {};

auto foo() -> S* _Nonnull * _Nonnull;

// --- import_double_pointer_return.carbon

library "[[@TEST_NAME]]";

import Cpp library "double_non_nullable_pointer_return.h";

fn IngestDoublePointer(s: Cpp.S**);

fn F() {
  //@dump-sem-ir-begin
  IngestDoublePointer(Cpp.foo());
  Cpp.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Const non-nullable pointer as a return value
// ============================================================================

// --- const_non_nullable_pointer_return.h

struct S {};

auto foo() -> const S* _Nonnull;

// --- import_const_non_nullable_pointer_return.carbon

library "[[@TEST_NAME]]";

import Cpp library "const_non_nullable_pointer_return.h";

fn IngestConstPointer(s: const Cpp.S*);

fn F() {
  //@dump-sem-ir-begin
  IngestConstPointer(Cpp.foo());
  Cpp.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Nullable pointer
// ============================================================================

// --- nullable_pointer_param.h

struct S {};

auto foo(S*) -> void;

auto get() -> S*;

// --- non_nullable_pointer_arg_to_pointer_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "nullable_pointer_param.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.S = {};
  Cpp.foo(&s);
  //@dump-sem-ir-end
}

// --- null_pointer_arg_to_pointer_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "nullable_pointer_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(Core.Optional(Cpp.S*).None());
  //@dump-sem-ir-end
}

// --- nonnull_pointer_arg_to_pointer_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "nullable_pointer_param.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.S = {};
  Cpp.foo(Core.Optional(Cpp.S*).Some(&s));
  //@dump-sem-ir-end
}

// --- forward_nullable_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp library "nullable_pointer_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(Cpp.get());
  //@dump-sem-ir-end
}

// ============================================================================
// Deduced pointer type as template argument
// ============================================================================

// --- deduced_any_param.h

struct S {};

template<typename T>
auto foo(T) -> T;

// --- import_deduced_any_param_as_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp library "deduced_any_param.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.S = {};

  // We assume that this cannot return a null pointer, because the function is
  // generic in the type `T`.
  // TODO: This isn't guaranteed; is this the behavior we want?
  let unused p: Cpp.S* = Cpp.foo(&s);
  //@dump-sem-ir-end
}

// --- deduced_pointer_param.h

struct S {};

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
  var s: Cpp.S = {};
  Cpp.Direct(&s);

  let unused a: Core.Optional(Cpp.S*) = Cpp.Indirect({} as Cpp.S);
  //@dump-sem-ir-end
}

// --- fail_import_deduced_pointer_param_as_optional_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp library "deduced_pointer_param.h";

fn F() {
  // This could return a null pointer, so should not be given the type `Cpp.S*`.
  // CHECK:STDERR: fail_import_deduced_pointer_param_as_optional_pointer.carbon:[[@LINE+7]]:26: error: cannot implicitly convert expression of type `Core.Optional(Cpp.S* as Core.OptionalStorage)` to `Cpp.S*` [ConversionFailure]
  // CHECK:STDERR:   let unused a: Cpp.S* = Cpp.Indirect({} as Cpp.S);
  // CHECK:STDERR:                          ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_deduced_pointer_param_as_optional_pointer.carbon:[[@LINE+4]]:26: note: type `Core.Optional(Cpp.S* as Core.OptionalStorage)` does not implement interface `Core.ImplicitAs(Cpp.S*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused a: Cpp.S* = Cpp.Indirect({} as Cpp.S);
  // CHECK:STDERR:                          ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused a: Cpp.S* = Cpp.Indirect({} as Cpp.S);
}

// CHECK:STDOUT: --- import_non_nullable_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.0f5: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.0f5 = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.0f5 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.037: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %S.cpp_destructor.type: type = fn_type @S.cpp_destructor [concrete]
// CHECK:STDOUT:   %S.cpp_destructor: %S.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.0f5 = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.0f5 = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc8_19.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_19.2: init %S to %s.var = class_init () [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_3: init %S = converted %.loc8_19.1, %.loc8_19.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = ref_binding s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %addr: %ptr.037 = addr_of %s.ref
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%addr)
// CHECK:STDOUT:   %S.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_nullable_pointer_param_using_const_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %const: type = const_type %ptr [concrete]
// CHECK:STDOUT:   %pattern_type.282: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.282 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:   %p.var_patt: %pattern_type.282 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc10_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.282 = ref_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.282 = var_pattern %p.patt [concrete = constants.%p.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %const = var %p.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %const = call %G.ref()
// CHECK:STDOUT:   assign %p.var, %G.call
// CHECK:STDOUT:   %.loc10: type = splice_block %const [concrete = constants.%const] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr: type = ptr_type %S.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     %const: type = const_type %ptr [concrete = constants.%const]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %const = ref_binding p, %p.var
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %p.ref: ref %const = name_ref p, %p
// CHECK:STDOUT:   %.loc11_11.1: ref %ptr = as_compatible %p.ref
// CHECK:STDOUT:   %.loc11_11.2: ref %ptr = converted %p.ref, %.loc11_11.1
// CHECK:STDOUT:   %.loc11_11.3: %ptr = acquire_value %.loc11_11.2
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc11_11.3)
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
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.0f5: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.0f5 = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.0f5 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.037: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.de6: type = pattern_type %ptr.037 [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.de6 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:   %p.var_patt: %pattern_type.de6 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.478: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%S) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.048: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%S) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.def: %ptr.as.Copy.impl.Op.type.048 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.037, (%Copy.impl_witness.478) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.eff: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.166: type = fn_type_with_self_type %Copy.WithSelf.Op.type.eff, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.def, @ptr.as.Copy.impl.Op(%S) [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.65c: type = ptr_type %ptr.037 [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %S.cpp_destructor.type: type = fn_type @S.cpp_destructor [concrete]
// CHECK:STDOUT:   %S.cpp_destructor: %S.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4f1 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.0f5 = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.0f5 = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc8_19.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_19.2: init %S to %s.var = class_init () [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_3: init %S = converted %.loc8_19.1, %.loc8_19.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref.loc8 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = ref_binding s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.de6 = ref_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.de6 = var_pattern %p.patt [concrete = constants.%p.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.037 = var %p.var_patt
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %addr.loc9: %ptr.037 = addr_of %s.ref
// CHECK:STDOUT:   %impl.elem0: %.166 = impl_witness_access constants.%Copy.impl_witness.478, element0 [concrete = constants.%ptr.as.Copy.impl.Op.def]
// CHECK:STDOUT:   %bound_method.loc9_19.1: <bound method> = bound_method %addr.loc9, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%S) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_19.2: <bound method> = bound_method %addr.loc9, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.037 = call %bound_method.loc9_19.2(%addr.loc9)
// CHECK:STDOUT:   assign %p.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9: type = splice_block %ptr [concrete = constants.%ptr.037] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc9: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr: type = ptr_type %S.ref.loc9 [concrete = constants.%ptr.037]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.037 = ref_binding p, %p.var
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %p.ref: ref %ptr.037 = name_ref p, %p
// CHECK:STDOUT:   %addr.loc10: %ptr.65c = addr_of %p.ref
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%addr.loc10)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %p.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%p.var)
// CHECK:STDOUT:   %S.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %ptr.037) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_nullable_pointer_to_const_param_using_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.914: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.914 = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.914 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.711: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.914 = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.914 = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %const = var %s.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc10_3: ref %const = splice_block %s.var {}
// CHECK:STDOUT:   %G.call: init %const to %.loc10_3 = call %G.ref()
// CHECK:STDOUT:   assign %s.var, %G.call
// CHECK:STDOUT:   %.loc10_10: type = splice_block %const [concrete = constants.%const] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %const: type = const_type %S.ref [concrete = constants.%const]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %const = ref_binding s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %const = name_ref s, %s
// CHECK:STDOUT:   %addr: %ptr.711 = addr_of %s.ref
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%addr)
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
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %pattern_type.0f5: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.0f5 = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.0f5 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.037: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %ptr.711: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %S.cpp_destructor.type: type = fn_type @S.cpp_destructor [concrete]
// CHECK:STDOUT:   %S.cpp_destructor: %S.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.0f5 = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.0f5 = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc8_3: ref %S = splice_block %s.var {}
// CHECK:STDOUT:   %G.call: init %S to %.loc8_3 = call %G.ref()
// CHECK:STDOUT:   assign %s.var, %G.call
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = ref_binding s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %addr: %ptr.037 = addr_of %s.ref
// CHECK:STDOUT:   %.loc9_11.1: %ptr.711 = as_compatible %addr
// CHECK:STDOUT:   %.loc9_11.2: %ptr.711 = converted %addr, %.loc9_11.1
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc9_11.2)
// CHECK:STDOUT:   %S.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_const_non_nullable_pointer_param_using_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %const: type = const_type %ptr [concrete]
// CHECK:STDOUT:   %pattern_type.282: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.282 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:   %p.var_patt: %pattern_type.282 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc10_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.282 = ref_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.282 = var_pattern %p.patt [concrete = constants.%p.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %const = var %p.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %const = call %G.ref()
// CHECK:STDOUT:   assign %p.var, %G.call
// CHECK:STDOUT:   %.loc10: type = splice_block %const [concrete = constants.%const] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr: type = ptr_type %S.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     %const: type = const_type %ptr [concrete = constants.%const]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %const = ref_binding p, %p.var
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %p.ref: ref %const = name_ref p, %p
// CHECK:STDOUT:   %.loc11_11.1: ref %ptr = as_compatible %p.ref
// CHECK:STDOUT:   %.loc11_11.2: ref %ptr = converted %p.ref, %.loc11_11.1
// CHECK:STDOUT:   %.loc11_11.3: %ptr = acquire_value %.loc11_11.2
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc11_11.3)
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
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.de6: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.de6 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:   %p.var_patt: %pattern_type.de6 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.de6 = ref_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.de6 = var_pattern %p.patt [concrete = constants.%p.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr = var %p.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %ptr = call %G.ref()
// CHECK:STDOUT:   assign %p.var, %G.call
// CHECK:STDOUT:   %.loc10: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr: type = ptr_type %S.ref [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr = ref_binding p, %p.var
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %p.ref: ref %ptr = name_ref p, %p
// CHECK:STDOUT:   %.loc11: %ptr = acquire_value %p.ref
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc11)
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
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.037: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %const.f96: type = const_type %ptr.037 [concrete]
// CHECK:STDOUT:   %pattern_type.282: type = pattern_type %const.f96 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.282 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:   %p.var_patt: %pattern_type.282 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.781: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.e46: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.781) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.ddb: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.e46) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.d38: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.191: %ptr.as.OptionalStorage.impl.Some.type.d38 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.1: <witness> = custom_witness (%Destroy.Op.1a2547.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.5b4: %Destroy.type = facet_value %ptr.037, (%custom_witness.df9cc1.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.9f2: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.5b4) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.582: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.7bc, @ptr.as.OptionalStorage.impl(%S) [concrete]
// CHECK:STDOUT:   %.08a0: type = maybe_unformed_type %ptr.037 [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.0ce: %OptionalStorage.type = facet_value %ptr.037, (%OptionalStorage.impl_witness.582) [concrete]
// CHECK:STDOUT:   %Optional.659: type = class_type @Optional, @Optional(%OptionalStorage.facet.0ce) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1c1: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.659)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3aa489.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T.67d)> [symbolic]
// CHECK:STDOUT:   %U.294: %ImplicitAs.type.3aa489.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.type.f7e: type = fn_type @const.as.ImplicitAs.impl.Convert, @const.as.ImplicitAs.impl(%T.67d, %U.294) [symbolic]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.2f7: %const.as.ImplicitAs.impl.Convert.type.f7e = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.3b8: type = facet_type <@OptionalAs, @OptionalAs(%T.220)> [symbolic]
// CHECK:STDOUT:   %U.042: %OptionalAs.type.3b8 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.ad5: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.167(%T.220, %U.042) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.331: %U.as_type.as.ImplicitAs.impl.Convert.type.ad5 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.561: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.0ce)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.774: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.220) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.5f9: %T.as_type.as.OptionalAs.impl.Convert.type.774 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.72e: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.efa, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.0ce) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.561 = facet_value %ptr.037, (%OptionalAs.impl_witness.72e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d0f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.cf7, @U.as_type.as.ImplicitAs.impl.167(%OptionalStorage.facet.0ce, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.21e: %ImplicitAs.type.1c1 = facet_value %ptr.037, (%ImplicitAs.impl_witness.d0f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.326: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b0d, @const.as.ImplicitAs.impl(%Optional.659, %ImplicitAs.facet.21e) [concrete]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.type.9f5: type = fn_type @const.as.ImplicitAs.impl.Convert, @const.as.ImplicitAs.impl(%Optional.659, %ImplicitAs.facet.21e) [concrete]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.1e5: %const.as.ImplicitAs.impl.Convert.type.9f5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9d6: %ImplicitAs.type.1c1 = facet_value %const.f96, (%ImplicitAs.impl_witness.326) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.9f9: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.659, %ImplicitAs.facet.9d6) [concrete]
// CHECK:STDOUT:   %.ba9: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.9f9, %ImplicitAs.facet.9d6 [concrete]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %const.as.ImplicitAs.impl.Convert.1e5, @const.as.ImplicitAs.impl.Convert(%Optional.659, %ImplicitAs.facet.21e) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc11_11.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.5: type = fn_type @Destroy.Op.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.5: %Destroy.Op.type.1d8f74.5 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.509: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.ddb)]
// CHECK:STDOUT:   %Core.import_ref.26e = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.d86: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.d38) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.191)]
// CHECK:STDOUT:   %Core.import_ref.274 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.358 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.db3 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.7bc = impl_witness_table (%Core.import_ref.509, %Core.import_ref.26e, %Core.import_ref.d86, %Core.import_ref.274, %Core.import_ref.358, %Core.import_ref.db3), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_12.1: type = splice_block %Optional [concrete = constants.%Optional.659] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.037, (constants.%OptionalStorage.impl_witness.582) [concrete = constants.%OptionalStorage.facet.0ce]
// CHECK:STDOUT:       %.loc11_12.2: %OptionalStorage.type = converted constants.%ptr.037, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.0ce]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.0ce) [concrete = constants.%Optional.659]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.a74: @const.as.ImplicitAs.impl.%const.as.ImplicitAs.impl.Convert.type (%const.as.ImplicitAs.impl.Convert.type.f7e) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @const.as.ImplicitAs.impl.%const.as.ImplicitAs.impl.Convert (constants.%const.as.ImplicitAs.impl.Convert.2f7)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b0d = impl_witness_table (%Core.import_ref.a74), @const.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.eb9: @U.as_type.as.ImplicitAs.impl.167.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.ad5) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.167.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.331)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cf7 = impl_witness_table (%Core.import_ref.eb9), @U.as_type.as.ImplicitAs.impl.167 [concrete]
// CHECK:STDOUT:   %Core.import_ref.017d8: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.774) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.5f9)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.efa = impl_witness_table (%Core.import_ref.017d8), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.282 = ref_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.282 = var_pattern %p.patt [concrete = constants.%p.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %const.f96 = var %p.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %const.f96 = call %G.ref()
// CHECK:STDOUT:   assign %p.var, %G.call
// CHECK:STDOUT:   %.loc10: type = splice_block %const [concrete = constants.%const.f96] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr: type = ptr_type %S.ref [concrete = constants.%ptr.037]
// CHECK:STDOUT:     %const: type = const_type %ptr [concrete = constants.%const.f96]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %const.f96 = ref_binding p, %p.var
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %p.ref: ref %const.f96 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.ba9 = impl_witness_access constants.%ImplicitAs.impl_witness.326, element0 [concrete = constants.%const.as.ImplicitAs.impl.Convert.1e5]
// CHECK:STDOUT:   %bound_method.loc11_11.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @const.as.ImplicitAs.impl.Convert(constants.%Optional.659, constants.%ImplicitAs.facet.21e) [concrete = constants.%const.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_11.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %.loc11_11.1: %const.f96 = acquire_value %p.ref
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.call: init %Optional.659 = call %bound_method.loc11_11.2(%.loc11_11.1)
// CHECK:STDOUT:   %.loc11_11.2: init %Optional.659 = converted %p.ref, %const.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_11.3: ref %Optional.659 = temporary_storage
// CHECK:STDOUT:   %.loc11_11.4: ref %Optional.659 = temporary %.loc11_11.3, %.loc11_11.2
// CHECK:STDOUT:   %.loc11_11.5: %Optional.659 = acquire_value %.loc11_11.4
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc11_11.5)
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %.loc11_11.4, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%.loc11_11.4)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10: <bound method> = bound_method %p.var, constants.%Destroy.Op.1a2547.5
// CHECK:STDOUT:   %Destroy.Op.call.loc10: init %empty_tuple.type = call %Destroy.Op.bound.loc10(%p.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_11.1(%self.param: ref %.08a0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_11.2(%self.param: ref %MaybeUnformed.9f2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_11.3(%self.param: ref %Optional.659) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10(%self.param: ref %const.f96) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_const_nullable_pointer_param_using_non_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.037: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.de6: type = pattern_type %ptr.037 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.de6 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:   %p.var_patt: %pattern_type.de6 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.781: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.e46: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.781) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.ddb: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.e46) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.d38: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.191: %ptr.as.OptionalStorage.impl.Some.type.d38 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.1: <witness> = custom_witness (%Destroy.Op.1a2547.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.5b4: %Destroy.type = facet_value %ptr.037, (%custom_witness.df9cc1.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.9f2: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.5b4) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.582: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.7bc, @ptr.as.OptionalStorage.impl(%S) [concrete]
// CHECK:STDOUT:   %.08a0: type = maybe_unformed_type %ptr.037 [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.0ce: %OptionalStorage.type = facet_value %ptr.037, (%OptionalStorage.impl_witness.582) [concrete]
// CHECK:STDOUT:   %Optional.659: type = class_type @Optional, @Optional(%OptionalStorage.facet.0ce) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1c1: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.659)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.3b8: type = facet_type <@OptionalAs, @OptionalAs(%T.220)> [symbolic]
// CHECK:STDOUT:   %U.042: %OptionalAs.type.3b8 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.ad5: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.167(%T.220, %U.042) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.331: %U.as_type.as.ImplicitAs.impl.Convert.type.ad5 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.561: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.0ce)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.774: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.220) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.5f9: %T.as_type.as.OptionalAs.impl.Convert.type.774 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.72e: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.efa, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.0ce) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.561 = facet_value %ptr.037, (%OptionalAs.impl_witness.72e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d0f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.cf7, @U.as_type.as.ImplicitAs.impl.167(%OptionalStorage.facet.0ce, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.cfa: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.167(%OptionalStorage.facet.0ce, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.8f4: %U.as_type.as.ImplicitAs.impl.Convert.type.cfa = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.21e: %ImplicitAs.type.1c1 = facet_value %ptr.037, (%ImplicitAs.impl_witness.d0f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c92: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.659, %ImplicitAs.facet.21e) [concrete]
// CHECK:STDOUT:   %.c09: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.c92, %ImplicitAs.facet.21e [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.as_type.as.ImplicitAs.impl.Convert.8f4, @U.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.0ce, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc11_11.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.509: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.ddb)]
// CHECK:STDOUT:   %Core.import_ref.26e = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.d86: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.d38) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.191)]
// CHECK:STDOUT:   %Core.import_ref.274 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.358 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.db3 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.7bc = impl_witness_table (%Core.import_ref.509, %Core.import_ref.26e, %Core.import_ref.d86, %Core.import_ref.274, %Core.import_ref.358, %Core.import_ref.db3), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_12.1: type = splice_block %Optional [concrete = constants.%Optional.659] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.037, (constants.%OptionalStorage.impl_witness.582) [concrete = constants.%OptionalStorage.facet.0ce]
// CHECK:STDOUT:       %.loc11_12.2: %OptionalStorage.type = converted constants.%ptr.037, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.0ce]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.0ce) [concrete = constants.%Optional.659]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.eb9: @U.as_type.as.ImplicitAs.impl.167.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.ad5) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.167.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.331)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cf7 = impl_witness_table (%Core.import_ref.eb9), @U.as_type.as.ImplicitAs.impl.167 [concrete]
// CHECK:STDOUT:   %Core.import_ref.017d8: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.774) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.5f9)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.efa = impl_witness_table (%Core.import_ref.017d8), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.de6 = ref_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.de6 = var_pattern %p.patt [concrete = constants.%p.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.037 = var %p.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %ptr.037 = call %G.ref()
// CHECK:STDOUT:   assign %p.var, %G.call
// CHECK:STDOUT:   %.loc10: type = splice_block %ptr [concrete = constants.%ptr.037] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr: type = ptr_type %S.ref [concrete = constants.%ptr.037]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.037 = ref_binding p, %p.var
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %p.ref: ref %ptr.037 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.c09 = impl_witness_access constants.%ImplicitAs.impl_witness.d0f, element0 [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.8f4]
// CHECK:STDOUT:   %bound_method.loc11_11.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @U.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.0ce, constants.%OptionalAs.facet) [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_11.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %.loc11_11.1: %ptr.037 = acquire_value %p.ref
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.659 = call %bound_method.loc11_11.2(%.loc11_11.1)
// CHECK:STDOUT:   %.loc11_11.2: init %Optional.659 = converted %p.ref, %U.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_11.3: ref %Optional.659 = temporary_storage
// CHECK:STDOUT:   %.loc11_11.4: ref %Optional.659 = temporary %.loc11_11.3, %.loc11_11.2
// CHECK:STDOUT:   %.loc11_11.5: %Optional.659 = acquire_value %.loc11_11.4
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc11_11.5)
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %.loc11_11.4, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%.loc11_11.4)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10: <bound method> = bound_method %p.var, constants.%Destroy.Op.1a2547.1
// CHECK:STDOUT:   %Destroy.Op.call.loc10: init %empty_tuple.type = call %Destroy.Op.bound.loc10(%p.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_11.1(%self.param: ref %.08a0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_11.2(%self.param: ref %MaybeUnformed.9f2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_11.3(%self.param: ref %Optional.659) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_nullable_pointer_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %IngestDoublePointer.type: type = fn_type @IngestDoublePointer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IngestDoublePointer: %IngestDoublePointer.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
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
// CHECK:STDOUT:   %foo.ref.loc10: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call.loc10: init %ptr = call imports.%foo.decl()
// CHECK:STDOUT:   %.loc10_31.1: %ptr = value_of_initializer %foo.call.loc10
// CHECK:STDOUT:   %.loc10_31.2: %ptr = converted %foo.call.loc10, %.loc10_31.1
// CHECK:STDOUT:   %IngestDoublePointer.call: init %empty_tuple.type = call %IngestDoublePointer.ref(%.loc10_31.2)
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc11: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call.loc11: init %ptr = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_double_pointer_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.037: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %ptr.65c: type = ptr_type %ptr.037 [concrete]
// CHECK:STDOUT:   %IngestDoublePointer.type: type = fn_type @IngestDoublePointer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IngestDoublePointer: %IngestDoublePointer.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
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
// CHECK:STDOUT:   %foo.ref.loc10: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call.loc10: init %ptr.65c = call imports.%foo.decl()
// CHECK:STDOUT:   %.loc10_31.1: %ptr.65c = value_of_initializer %foo.call.loc10
// CHECK:STDOUT:   %.loc10_31.2: %ptr.65c = converted %foo.call.loc10, %.loc10_31.1
// CHECK:STDOUT:   %IngestDoublePointer.call: init %empty_tuple.type = call %IngestDoublePointer.ref(%.loc10_31.2)
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc11: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call.loc11: init %ptr.65c = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_const_non_nullable_pointer_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %IngestConstPointer.type: type = fn_type @IngestConstPointer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IngestConstPointer: %IngestConstPointer.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
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
// CHECK:STDOUT:   %foo.ref.loc10: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call.loc10: init %ptr = call imports.%foo.decl()
// CHECK:STDOUT:   %.loc10_30.1: %ptr = value_of_initializer %foo.call.loc10
// CHECK:STDOUT:   %.loc10_30.2: %ptr = converted %foo.call.loc10, %.loc10_30.1
// CHECK:STDOUT:   %IngestConstPointer.call: init %empty_tuple.type = call %IngestConstPointer.ref(%.loc10_30.2)
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc11: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call.loc11: init %ptr = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_nullable_pointer_arg_to_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.0f5: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.0f5 = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.0f5 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.037: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.781: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.e46: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.781) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.ddb: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.e46) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.d38: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.191: %ptr.as.OptionalStorage.impl.Some.type.d38 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.1: <witness> = custom_witness (%Destroy.Op.1a2547.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.5b4: %Destroy.type = facet_value %ptr.037, (%custom_witness.df9cc1.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.9f2: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.5b4) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.582: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.7bc, @ptr.as.OptionalStorage.impl(%S) [concrete]
// CHECK:STDOUT:   %.08a0: type = maybe_unformed_type %ptr.037 [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.0ce: %OptionalStorage.type = facet_value %ptr.037, (%OptionalStorage.impl_witness.582) [concrete]
// CHECK:STDOUT:   %Optional.659: type = class_type @Optional, @Optional(%OptionalStorage.facet.0ce) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1c1: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.659)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.3b8: type = facet_type <@OptionalAs, @OptionalAs(%T.220)> [symbolic]
// CHECK:STDOUT:   %U.042: %OptionalAs.type.3b8 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.ad5: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.167(%T.220, %U.042) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.331: %U.as_type.as.ImplicitAs.impl.Convert.type.ad5 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.561: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.0ce)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.774: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.220) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.5f9: %T.as_type.as.OptionalAs.impl.Convert.type.774 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.72e: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.efa, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.0ce) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.561 = facet_value %ptr.037, (%OptionalAs.impl_witness.72e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d0f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.cf7, @U.as_type.as.ImplicitAs.impl.167(%OptionalStorage.facet.0ce, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.cfa: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.167(%OptionalStorage.facet.0ce, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.8f4: %U.as_type.as.ImplicitAs.impl.Convert.type.cfa = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.21e: %ImplicitAs.type.1c1 = facet_value %ptr.037, (%ImplicitAs.impl_witness.d0f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c92: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.659, %ImplicitAs.facet.21e) [concrete]
// CHECK:STDOUT:   %.c09: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.c92, %ImplicitAs.facet.21e [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.as_type.as.ImplicitAs.impl.Convert.8f4, @U.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.0ce, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc9_11.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT:   %S.cpp_destructor.type: type = fn_type @S.cpp_destructor [concrete]
// CHECK:STDOUT:   %S.cpp_destructor: %S.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.509: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.ddb)]
// CHECK:STDOUT:   %Core.import_ref.26e = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.d86: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.d38) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.191)]
// CHECK:STDOUT:   %Core.import_ref.274 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.358 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.db3 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.7bc = impl_witness_table (%Core.import_ref.509, %Core.import_ref.26e, %Core.import_ref.d86, %Core.import_ref.274, %Core.import_ref.358, %Core.import_ref.db3), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc9_13.1: type = splice_block %Optional [concrete = constants.%Optional.659] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.037, (constants.%OptionalStorage.impl_witness.582) [concrete = constants.%OptionalStorage.facet.0ce]
// CHECK:STDOUT:       %.loc9_13.2: %OptionalStorage.type = converted constants.%ptr.037, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.0ce]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.0ce) [concrete = constants.%Optional.659]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.eb9: @U.as_type.as.ImplicitAs.impl.167.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.ad5) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.167.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.331)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cf7 = impl_witness_table (%Core.import_ref.eb9), @U.as_type.as.ImplicitAs.impl.167 [concrete]
// CHECK:STDOUT:   %Core.import_ref.017d8: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.774) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.5f9)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.efa = impl_witness_table (%Core.import_ref.017d8), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.0f5 = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.0f5 = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc8_19.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_19.2: init %S to %s.var = class_init () [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_3: init %S = converted %.loc8_19.1, %.loc8_19.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = ref_binding s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %addr: %ptr.037 = addr_of %s.ref
// CHECK:STDOUT:   %impl.elem0: %.c09 = impl_witness_access constants.%ImplicitAs.impl_witness.d0f, element0 [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.8f4]
// CHECK:STDOUT:   %bound_method.loc9_11.1: <bound method> = bound_method %addr, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @U.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.0ce, constants.%OptionalAs.facet) [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_11.2: <bound method> = bound_method %addr, %specific_fn
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.659 = call %bound_method.loc9_11.2(%addr)
// CHECK:STDOUT:   %.loc9_11.1: init %Optional.659 = converted %addr, %U.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc9_11.2: ref %Optional.659 = temporary_storage
// CHECK:STDOUT:   %.loc9_11.3: ref %Optional.659 = temporary %.loc9_11.2, %.loc9_11.1
// CHECK:STDOUT:   %.loc9_11.4: %Optional.659 = acquire_value %.loc9_11.3
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc9_11.4)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc9_11.3, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc9_11.3)
// CHECK:STDOUT:   %S.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_11.1(%self.param: ref %.08a0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_11.2(%self.param: ref %MaybeUnformed.9f2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_11.3(%self.param: ref %Optional.659) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- null_pointer_arg_to_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Optional.type: type = generic_class_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.None.type.bdc: type = fn_type @Optional.None, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.None.01e: %Optional.None.type.bdc = struct_value () [symbolic]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.037: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.781: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.e46: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.781) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.ddb: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.e46) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.type.703: type = fn_type @ptr.as.OptionalStorage.impl.None, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.467: %ptr.as.OptionalStorage.impl.None.type.703 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.1: <witness> = custom_witness (%Destroy.Op.1a2547.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.5b4: %Destroy.type = facet_value %ptr.037, (%custom_witness.df9cc1.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.9f2: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.5b4) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.725: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.aa4, @ptr.as.OptionalStorage.impl(%S) [concrete]
// CHECK:STDOUT:   %.08a: type = maybe_unformed_type %ptr.037 [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.037, (%OptionalStorage.impl_witness.725) [concrete]
// CHECK:STDOUT:   %Optional.db1: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.type.fc8: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.9d4: %Optional.None.type.fc8 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %Optional.None.9d4, @Optional.None(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc8_38.3 [concrete]
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
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.import_ref.e9f: @Optional.%Optional.None.type (%Optional.None.type.bdc) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @Optional.%Optional.None (constants.%Optional.None.01e)]
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.509: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.ddb)]
// CHECK:STDOUT:   %Core.import_ref.df2: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None.type (%ptr.as.OptionalStorage.impl.None.type.703) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None (constants.%ptr.as.OptionalStorage.impl.None.467)]
// CHECK:STDOUT:   %Core.import_ref.a6e = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.274 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.358 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.db3 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.aa4 = impl_witness_table (%Core.import_ref.509, %Core.import_ref.df2, %Core.import_ref.a6e, %Core.import_ref.274, %Core.import_ref.358, %Core.import_ref.db3), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc8_39.1: type = splice_block %Optional [concrete = constants.%Optional.db1] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.037, (constants.%OptionalStorage.impl_witness.725) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %.loc8_39.2: %OptionalStorage.type = converted constants.%ptr.037, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.db1]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Optional.ref: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Cpp.ref.loc8_25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %ptr: type = ptr_type %S.ref [concrete = constants.%ptr.037]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.725) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc8_31: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.db1]
// CHECK:STDOUT:   %.loc8_32: %Optional.None.type.fc8 = specific_constant imports.%Core.import_ref.e9f, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.9d4]
// CHECK:STDOUT:   %None.ref: %Optional.None.type.fc8 = name_ref None, %.loc8_32 [concrete = constants.%Optional.None.9d4]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %None.ref, @Optional.None(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.specific_fn]
// CHECK:STDOUT:   %Optional.None.call: init %Optional.db1 = call %Optional.None.specific_fn()
// CHECK:STDOUT:   %.loc8_38.1: ref %Optional.db1 = temporary_storage
// CHECK:STDOUT:   %.loc8_38.2: ref %Optional.db1 = temporary %.loc8_38.1, %Optional.None.call
// CHECK:STDOUT:   %.loc8_38.3: %Optional.db1 = acquire_value %.loc8_38.2
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc8_38.3)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc8_38.2, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc8_38.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_38.1(%self.param: ref %.08a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_38.2(%self.param: ref %MaybeUnformed.9f2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_38.3(%self.param: ref %Optional.db1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nonnull_pointer_arg_to_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.0f5: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.0f5 = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.0f5 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Optional.type: type = generic_class_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.304: type = fn_type @Optional.Some, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Some.2a0: %Optional.Some.type.304 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.037: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.781: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.e46: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.781) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.ddb: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.e46) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.d38: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.191: %ptr.as.OptionalStorage.impl.Some.type.d38 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.1: <witness> = custom_witness (%Destroy.Op.1a2547.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.5b4: %Destroy.type = facet_value %ptr.037, (%custom_witness.df9cc1.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.9f2: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.5b4) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.582: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.7bc, @ptr.as.OptionalStorage.impl(%S) [concrete]
// CHECK:STDOUT:   %.08a0: type = maybe_unformed_type %ptr.037 [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.037, (%OptionalStorage.impl_witness.582) [concrete]
// CHECK:STDOUT:   %Optional.659: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.Some.type.004: type = fn_type @Optional.Some, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.Some.9ba: %Optional.Some.type.004 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Some.specific_fn: <specific function> = specific_function %Optional.Some.9ba, @Optional.Some(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc9_40.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT:   %S.cpp_destructor.type: type = fn_type @S.cpp_destructor [concrete]
// CHECK:STDOUT:   %S.cpp_destructor: %S.cpp_destructor.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.import_ref.48e: @Optional.%Optional.Some.type (%Optional.Some.type.304) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @Optional.%Optional.Some (constants.%Optional.Some.2a0)]
// CHECK:STDOUT:   %Core.import_ref.509: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.ddb)]
// CHECK:STDOUT:   %Core.import_ref.26e = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.d86: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.d38) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.191)]
// CHECK:STDOUT:   %Core.import_ref.274 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.358 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.db3 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.7bc = impl_witness_table (%Core.import_ref.509, %Core.import_ref.26e, %Core.import_ref.d86, %Core.import_ref.274, %Core.import_ref.358, %Core.import_ref.db3), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc9_41.1: type = splice_block %Optional [concrete = constants.%Optional.659] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.037, (constants.%OptionalStorage.impl_witness.582) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %.loc9_41.2: %OptionalStorage.type = converted constants.%ptr.037, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.659]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.0f5 = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.0f5 = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc8_19.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_19.2: init %S to %s.var = class_init () [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_3: init %S = converted %.loc8_19.1, %.loc8_19.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref.loc8 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = ref_binding s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc9_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Optional.ref: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Cpp.ref.loc9_25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc9: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %ptr: type = ptr_type %S.ref.loc9 [concrete = constants.%ptr.037]
// CHECK:STDOUT:   %OptionalStorage.facet.loc9_31: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.582) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc9_31: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet.loc9_31 [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.659]
// CHECK:STDOUT:   %.loc9_32: %Optional.Some.type.004 = specific_constant imports.%Core.import_ref.48e, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.Some.9ba]
// CHECK:STDOUT:   %Some.ref: %Optional.Some.type.004 = name_ref Some, %.loc9_32 [concrete = constants.%Optional.Some.9ba]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %addr: %ptr.037 = addr_of %s.ref
// CHECK:STDOUT:   %OptionalStorage.facet.loc9_40.1: %OptionalStorage.type = facet_value constants.%ptr.037, (constants.%OptionalStorage.impl_witness.582) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc9_40.1: %OptionalStorage.type = converted constants.%ptr.037, %OptionalStorage.facet.loc9_40.1 [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %OptionalStorage.facet.loc9_40.2: %OptionalStorage.type = facet_value constants.%ptr.037, (constants.%OptionalStorage.impl_witness.582) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc9_40.2: %OptionalStorage.type = converted constants.%ptr.037, %OptionalStorage.facet.loc9_40.2 [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %Optional.Some.specific_fn: <specific function> = specific_function %Some.ref, @Optional.Some(constants.%OptionalStorage.facet) [concrete = constants.%Optional.Some.specific_fn]
// CHECK:STDOUT:   %Optional.Some.call: init %Optional.659 = call %Optional.Some.specific_fn(%addr)
// CHECK:STDOUT:   %.loc9_40.3: ref %Optional.659 = temporary_storage
// CHECK:STDOUT:   %.loc9_40.4: ref %Optional.659 = temporary %.loc9_40.3, %Optional.Some.call
// CHECK:STDOUT:   %.loc9_40.5: %Optional.659 = acquire_value %.loc9_40.4
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc9_40.5)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc9_40.4, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc9_40.4)
// CHECK:STDOUT:   %S.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_40.1(%self.param: ref %.08a0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_40.2(%self.param: ref %MaybeUnformed.9f2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_40.3(%self.param: ref %Optional.659) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- forward_nullable_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %get.cpp_overload_set.type: type = cpp_overload_set_type @get.cpp_overload_set [concrete]
// CHECK:STDOUT:   %get.cpp_overload_set.value: %get.cpp_overload_set.type = cpp_overload_set_value @get.cpp_overload_set [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.037: type = ptr_type %S [concrete]
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
// CHECK:STDOUT:   %Destroy.facet.5b4: %Destroy.type = facet_value %ptr.037, (%custom_witness.df9cc1.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.9f2: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.5b4) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.392: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.76e, @ptr.as.OptionalStorage.impl(%S) [concrete]
// CHECK:STDOUT:   %.08a: type = maybe_unformed_type %ptr.037 [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.037, (%OptionalStorage.impl_witness.392) [concrete]
// CHECK:STDOUT:   %Optional.19d: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %get.type: type = fn_type @get [concrete]
// CHECK:STDOUT:   %get: %get.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc8_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     .get = %get.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %get.cpp_overload_set.value: %get.cpp_overload_set.type = cpp_overload_set_value @get.cpp_overload_set [concrete = constants.%get.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.509: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.ddb)]
// CHECK:STDOUT:   %Core.import_ref.26e = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.a6e = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.274 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.358 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.db3 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.76e = impl_witness_table (%Core.import_ref.509, %Core.import_ref.26e, %Core.import_ref.a6e, %Core.import_ref.274, %Core.import_ref.358, %Core.import_ref.db3), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %get.decl: %get.type = fn_decl @get [concrete = constants.%get] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.037, (constants.%OptionalStorage.impl_witness.392) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %.loc8: %OptionalStorage.type = converted constants.%ptr.037, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.19d]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc8_20.1: type = splice_block %Optional [concrete = constants.%Optional.19d] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.037, (constants.%OptionalStorage.impl_witness.392) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %.loc8_20.2: %OptionalStorage.type = converted constants.%ptr.037, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.19d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %get.ref: %get.cpp_overload_set.type = name_ref get, imports.%get.cpp_overload_set.value [concrete = constants.%get.cpp_overload_set.value]
// CHECK:STDOUT:   %get.call: init %Optional.19d = call imports.%get.decl()
// CHECK:STDOUT:   %.loc8_19.1: ref %Optional.19d = temporary_storage
// CHECK:STDOUT:   %.loc8_19.2: ref %Optional.19d = temporary %.loc8_19.1, %get.call
// CHECK:STDOUT:   %.loc8_19.3: %Optional.19d = acquire_value %.loc8_19.2
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc8_19.3)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc8_19.2, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc8_19.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_19.1(%self.param: ref %.08a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_19.2(%self.param: ref %MaybeUnformed.9f2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_19.3(%self.param: ref %Optional.19d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_deduced_any_param_as_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.0f5: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.0f5 = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.0f5 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.037: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.de6: type = pattern_type %ptr.037 [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.de6 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %S.cpp_destructor.type: type = fn_type @S.cpp_destructor [concrete]
// CHECK:STDOUT:   %S.cpp_destructor: %S.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.0f5 = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.0f5 = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc8_19.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_19.2: init %S to %s.var = class_init () [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_3: init %S = converted %.loc8_19.1, %.loc8_19.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref.loc8 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = ref_binding s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.de6 = value_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %addr: %ptr.037 = addr_of %s.ref
// CHECK:STDOUT:   %foo.call: init %ptr.037 = call imports.%foo.decl(%addr)
// CHECK:STDOUT:   %.loc13_36.1: %ptr.037 = value_of_initializer %foo.call
// CHECK:STDOUT:   %.loc13_36.2: %ptr.037 = converted %foo.call, %.loc13_36.1
// CHECK:STDOUT:   %.loc13_22: type = splice_block %ptr [concrete = constants.%ptr.037] {
// CHECK:STDOUT:     %Cpp.ref.loc13_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc13: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr: type = ptr_type %S.ref.loc13 [concrete = constants.%ptr.037]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.037 = value_binding p, %.loc13_36.2
// CHECK:STDOUT:   %S.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_deduced_pointer_param_as_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.0f5: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.0f5 = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.0f5 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %Direct.cpp_overload_set.type: type = cpp_overload_set_type @Direct.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Direct.cpp_overload_set.value: %Direct.cpp_overload_set.type = cpp_overload_set_value @Direct.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.037: type = ptr_type %S [concrete]
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
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.d38: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.191: %ptr.as.OptionalStorage.impl.Some.type.d38 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.1: <witness> = custom_witness (%Destroy.Op.1a2547.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.5b4: %Destroy.type = facet_value %ptr.037, (%custom_witness.df9cc1.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.9f2: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.5b4) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.582: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.7bc, @ptr.as.OptionalStorage.impl(%S) [concrete]
// CHECK:STDOUT:   %.08a0: type = maybe_unformed_type %ptr.037 [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.0ce: %OptionalStorage.type = facet_value %ptr.037, (%OptionalStorage.impl_witness.582) [concrete]
// CHECK:STDOUT:   %Optional.659: type = class_type @Optional, @Optional(%OptionalStorage.facet.0ce) [concrete]
// CHECK:STDOUT:   %pattern_type.9f4: type = pattern_type %Optional.659 [concrete]
// CHECK:STDOUT:   %Direct.type: type = fn_type @Direct [concrete]
// CHECK:STDOUT:   %Direct: %Direct.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1c1: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.659)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.3b8: type = facet_type <@OptionalAs, @OptionalAs(%T.220)> [symbolic]
// CHECK:STDOUT:   %U.042: %OptionalAs.type.3b8 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.ad5: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.167(%T.220, %U.042) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.331: %U.as_type.as.ImplicitAs.impl.Convert.type.ad5 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.561: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.0ce)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.774: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.220) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.5f9: %T.as_type.as.OptionalAs.impl.Convert.type.774 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.72e: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.efa, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.0ce) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.561 = facet_value %ptr.037, (%OptionalAs.impl_witness.72e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d0f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.cf7, @U.as_type.as.ImplicitAs.impl.167(%OptionalStorage.facet.0ce, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.cfa: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.167(%OptionalStorage.facet.0ce, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.8f4: %U.as_type.as.ImplicitAs.impl.Convert.type.cfa = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.21e: %ImplicitAs.type.1c1 = facet_value %ptr.037, (%ImplicitAs.impl_witness.d0f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c92: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.659, %ImplicitAs.facet.21e) [concrete]
// CHECK:STDOUT:   %.c09: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.c92, %ImplicitAs.facet.21e [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.as_type.as.ImplicitAs.impl.Convert.8f4, @U.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.0ce, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.9f4 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %Indirect.cpp_overload_set.type: type = cpp_overload_set_type @Indirect.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Indirect.cpp_overload_set.value: %Indirect.cpp_overload_set.type = cpp_overload_set_value @Indirect.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Indirect__carbon_thunk.type: type = fn_type @Indirect__carbon_thunk [concrete]
// CHECK:STDOUT:   %Indirect__carbon_thunk: %Indirect__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.a94: ref %S = temporary invalid, %S.val [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc13_65.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT:   %S.cpp_destructor.type: type = fn_type @S.cpp_destructor [concrete]
// CHECK:STDOUT:   %S.cpp_destructor: %S.cpp_destructor.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .Direct = %Direct.cpp_overload_set.value
// CHECK:STDOUT:     .Indirect = %Indirect.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Direct.cpp_overload_set.value: %Direct.cpp_overload_set.type = cpp_overload_set_value @Direct.cpp_overload_set [concrete = constants.%Direct.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.import_ref.509: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.ddb)]
// CHECK:STDOUT:   %Core.import_ref.26e = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.d86: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.d38) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.191)]
// CHECK:STDOUT:   %Core.import_ref.274 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.358 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.db3 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.7bc = impl_witness_table (%Core.import_ref.509, %Core.import_ref.26e, %Core.import_ref.d86, %Core.import_ref.274, %Core.import_ref.358, %Core.import_ref.db3), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Direct.decl: %Direct.type = fn_decl @Direct [concrete = constants.%Direct] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %OptionalStorage.facet.loc11_16.1: %OptionalStorage.type = facet_value constants.%ptr.037, (constants.%OptionalStorage.impl_witness.582) [concrete = constants.%OptionalStorage.facet.0ce]
// CHECK:STDOUT:     %.loc11_16.1: %OptionalStorage.type = converted constants.%ptr.037, %OptionalStorage.facet.loc11_16.1 [concrete = constants.%OptionalStorage.facet.0ce]
// CHECK:STDOUT:     %Optional.loc11_16.1: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.0ce) [concrete = constants.%Optional.659]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_16.2: type = splice_block %Optional.loc11_16.2 [concrete = constants.%Optional.659] {
// CHECK:STDOUT:       %OptionalStorage.facet.loc11_16.2: %OptionalStorage.type = facet_value constants.%ptr.037, (constants.%OptionalStorage.impl_witness.582) [concrete = constants.%OptionalStorage.facet.0ce]
// CHECK:STDOUT:       %.loc11_16.3: %OptionalStorage.type = converted constants.%ptr.037, %OptionalStorage.facet.loc11_16.2 [concrete = constants.%OptionalStorage.facet.0ce]
// CHECK:STDOUT:       %Optional.loc11_16.2: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.0ce) [concrete = constants.%Optional.659]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.eb9: @U.as_type.as.ImplicitAs.impl.167.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.ad5) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.167.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.331)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cf7 = impl_witness_table (%Core.import_ref.eb9), @U.as_type.as.ImplicitAs.impl.167 [concrete]
// CHECK:STDOUT:   %Core.import_ref.017d8: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.774) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.5f9)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.efa = impl_witness_table (%Core.import_ref.017d8), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT:   %Indirect.cpp_overload_set.value: %Indirect.cpp_overload_set.type = cpp_overload_set_value @Indirect.cpp_overload_set [concrete = constants.%Indirect.cpp_overload_set.value]
// CHECK:STDOUT:   %Indirect__carbon_thunk.decl: %Indirect__carbon_thunk.type = fn_decl @Indirect__carbon_thunk [concrete = constants.%Indirect__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.037, (constants.%OptionalStorage.impl_witness.582) [concrete = constants.%OptionalStorage.facet.0ce]
// CHECK:STDOUT:     %.loc13: %OptionalStorage.type = converted constants.%ptr.037, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.0ce]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.0ce) [concrete = constants.%Optional.659]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.0f5 = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.0f5 = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc10_19.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_19.2: init %S to %s.var = class_init () [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc10_3: init %S = converted %.loc10_19.1, %.loc10_19.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   assign %s.var, %.loc10_3
// CHECK:STDOUT:   %.loc10_13: type = splice_block %S.ref.loc10 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc10: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = ref_binding s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Direct.ref: %Direct.cpp_overload_set.type = name_ref Direct, imports.%Direct.cpp_overload_set.value [concrete = constants.%Direct.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %addr.loc11: %ptr.037 = addr_of %s.ref
// CHECK:STDOUT:   %impl.elem0: %.c09 = impl_witness_access constants.%ImplicitAs.impl_witness.d0f, element0 [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.8f4]
// CHECK:STDOUT:   %bound_method.loc11_14.1: <bound method> = bound_method %addr.loc11, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @U.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.0ce, constants.%OptionalAs.facet) [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_14.2: <bound method> = bound_method %addr.loc11, %specific_fn
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.659 = call %bound_method.loc11_14.2(%addr.loc11)
// CHECK:STDOUT:   %.loc11_14.1: init %Optional.659 = converted %addr.loc11, %U.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_14.2: ref %Optional.659 = temporary_storage
// CHECK:STDOUT:   %.loc11_14.3: ref %Optional.659 = temporary %.loc11_14.2, %.loc11_14.1
// CHECK:STDOUT:   %.loc11_14.4: %Optional.659 = acquire_value %.loc11_14.3
// CHECK:STDOUT:   %Direct.call: init %Optional.659 = call imports.%Direct.decl(%.loc11_14.4)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.9f4 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13_41: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Indirect.ref: %Indirect.cpp_overload_set.type = name_ref Indirect, imports.%Indirect.cpp_overload_set.value [concrete = constants.%Indirect.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc13_55.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Cpp.ref.loc13_60: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc13_63: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %.loc13_55.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc13_55.3: init %S to %.loc13_55.2 = class_init () [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc13_57.1: init %S = converted %.loc13_55.1, %.loc13_55.3 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc13_57.2: ref %S = temporary %.loc13_55.2, %.loc13_57.1 [concrete = constants.%.a94]
// CHECK:STDOUT:   %.loc13_57.3: %S = acquire_value %.loc13_57.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc13_57.4: ref %S = value_as_ref %.loc13_57.3
// CHECK:STDOUT:   %addr.loc13: %ptr.037 = addr_of %.loc13_57.4
// CHECK:STDOUT:   %Indirect__carbon_thunk.call: init %Optional.659 = call imports.%Indirect__carbon_thunk.decl(%addr.loc13)
// CHECK:STDOUT:   %.loc13_65.1: ref %Optional.659 = temporary_storage
// CHECK:STDOUT:   %.loc13_65.2: ref %Optional.659 = temporary %.loc13_65.1, %Indirect__carbon_thunk.call
// CHECK:STDOUT:   %.loc13_65.3: %Optional.659 = acquire_value %.loc13_65.2
// CHECK:STDOUT:   %.loc13_37.1: type = splice_block %Optional [concrete = constants.%Optional.659] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Cpp.ref.loc13_31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc13_34: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr: type = ptr_type %S.ref.loc13_34 [concrete = constants.%ptr.037]
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.582) [concrete = constants.%OptionalStorage.facet.0ce]
// CHECK:STDOUT:     %.loc13_37.2: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.0ce]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.0ce) [concrete = constants.%Optional.659]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %Optional.659 = value_binding a, %.loc13_65.3
// CHECK:STDOUT:   %Destroy.Op.bound.loc13: <bound method> = bound_method %.loc13_65.2, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call.loc13: init %empty_tuple.type = call %Destroy.Op.bound.loc13(%.loc13_65.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %.loc11_14.3, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%.loc11_14.3)
// CHECK:STDOUT:   %S.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_65.1(%self.param: ref %.08a0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_65.2(%self.param: ref %MaybeUnformed.9f2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_65.3(%self.param: ref %Optional.659) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
