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
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
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
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
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
// CHECK:STDOUT:   %addr: %ptr.5c7 = addr_of %s.ref
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
// CHECK:STDOUT:   %pattern_type.f25: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc10_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
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
// CHECK:STDOUT:     %p.patt: %pattern_type.f25 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.f25 = var_pattern %p.patt [concrete]
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
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %p.var, constants.%Destroy.Op.651ba6.2
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
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.259: type = pattern_type %ptr.5c7 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.26a: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%S) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.8cf: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%S) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.85d: %ptr.as.Copy.impl.Op.type.8cf = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.5c7, (%Copy.impl_witness.26a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c46: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.1c4: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c46, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.85d, @ptr.as.Copy.impl.Op(%S) [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.dfe: type = ptr_type %ptr.5c7 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
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
// CHECK:STDOUT:     %p.patt: %pattern_type.259 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.259 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.5c7 = var %p.var_patt
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %addr.loc9: %ptr.5c7 = addr_of %s.ref
// CHECK:STDOUT:   %impl.elem0: %.1c4 = impl_witness_access constants.%Copy.impl_witness.26a, element0 [concrete = constants.%ptr.as.Copy.impl.Op.85d]
// CHECK:STDOUT:   %bound_method.loc9_19.1: <bound method> = bound_method %addr.loc9, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%S) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_19.2: <bound method> = bound_method %addr.loc9, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.5c7 = call %bound_method.loc9_19.2(%addr.loc9)
// CHECK:STDOUT:   assign %p.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9: type = splice_block %ptr [concrete = constants.%ptr.5c7] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc9: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr: type = ptr_type %S.ref.loc9 [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.5c7 = ref_binding p, %p.var
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %p.ref: ref %ptr.5c7 = name_ref p, %p
// CHECK:STDOUT:   %addr.loc10: %ptr.dfe = addr_of %p.ref
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%addr.loc10)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %p.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%p.var)
// CHECK:STDOUT:   %S.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %ptr.5c7) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_nullable_pointer_to_const_param_using_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.9be: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.ff5: type = ptr_type %const [concrete]
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
// CHECK:STDOUT:     %s.patt: %pattern_type.9be = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.9be = var_pattern %s.patt [concrete]
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
// CHECK:STDOUT:   %addr: %ptr.ff5 = addr_of %s.ref
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
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %ptr.ff5: type = ptr_type %const [concrete]
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
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
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
// CHECK:STDOUT:   %addr: %ptr.5c7 = addr_of %s.ref
// CHECK:STDOUT:   %.loc9_11.1: %ptr.ff5 = as_compatible %addr
// CHECK:STDOUT:   %.loc9_11.2: %ptr.ff5 = converted %addr, %.loc9_11.1
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
// CHECK:STDOUT:   %pattern_type.f25: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc10_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
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
// CHECK:STDOUT:     %p.patt: %pattern_type.f25 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.f25 = var_pattern %p.patt [concrete]
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
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %p.var, constants.%Destroy.Op.651ba6.2
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
// CHECK:STDOUT:   %pattern_type.259: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
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
// CHECK:STDOUT:     %p.patt: %pattern_type.259 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.259 = var_pattern %p.patt [concrete]
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
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %const.b9a: type = const_type %ptr.5c7 [concrete]
// CHECK:STDOUT:   %pattern_type.f25: type = pattern_type %const.b9a [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.542: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.e1f: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.617: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.e1f) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2e9: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.617) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.c7a: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.38a: %ptr.as.OptionalStorage.impl.Some.type.c7a = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8d7fae.1: <witness> = custom_witness (%Destroy.Op.651ba6.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.021: %Destroy.type = facet_value %ptr.5c7, (%custom_witness.8d7fae.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.74a: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.021) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.ba1: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.bee, @ptr.as.OptionalStorage.impl(%S) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.872: %OptionalStorage.type = facet_value %ptr.5c7, (%OptionalStorage.impl_witness.ba1) [concrete]
// CHECK:STDOUT:   %Optional.065: type = class_type @Optional, @Optional(%OptionalStorage.facet.872) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1a0: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.065)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.031604.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T.67d)> [symbolic]
// CHECK:STDOUT:   %U.738: %ImplicitAs.type.031604.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.type.a49: type = fn_type @const.as.ImplicitAs.impl.Convert, @const.as.ImplicitAs.impl(%T.67d, %U.738) [symbolic]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.d5f: %const.as.ImplicitAs.impl.Convert.type.a49 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.ea5: type = facet_type <@OptionalAs, @OptionalAs(%T.542)> [symbolic]
// CHECK:STDOUT:   %U.2d2: %OptionalAs.type.ea5 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.79d: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.1a8(%T.542, %U.2d2) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.dd8: %U.as_type.as.ImplicitAs.impl.Convert.type.79d = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.8cb: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.872)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.072: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.542) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.6f5: %T.as_type.as.OptionalAs.impl.Convert.type.072 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.83b: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.45f, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.872) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.8cb = facet_value %ptr.5c7, (%OptionalAs.impl_witness.83b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.de6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.67f, @U.as_type.as.ImplicitAs.impl.1a8(%OptionalStorage.facet.872, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a29: %ImplicitAs.type.1a0 = facet_value %ptr.5c7, (%ImplicitAs.impl_witness.de6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2c2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c8f, @const.as.ImplicitAs.impl(%Optional.065, %ImplicitAs.facet.a29) [concrete]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.type.ee8: type = fn_type @const.as.ImplicitAs.impl.Convert, @const.as.ImplicitAs.impl(%Optional.065, %ImplicitAs.facet.a29) [concrete]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.4ce: %const.as.ImplicitAs.impl.Convert.type.ee8 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b46: %ImplicitAs.type.1a0 = facet_value %const.b9a, (%ImplicitAs.impl_witness.2c2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.9f5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.065, %ImplicitAs.facet.b46) [concrete]
// CHECK:STDOUT:   %.657: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.9f5, %ImplicitAs.facet.b46 [concrete]
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %const.as.ImplicitAs.impl.Convert.4ce, @const.as.ImplicitAs.impl.Convert(%Optional.065, %ImplicitAs.facet.a29) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc11_11.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.4: type = fn_type @Destroy.Op.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.4: %Destroy.Op.type.bae255.4 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.6af: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.2e9)]
// CHECK:STDOUT:   %Core.import_ref.33a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.bb5: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.c7a) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.38a)]
// CHECK:STDOUT:   %Core.import_ref.433 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6e2 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.bee = impl_witness_table (%Core.import_ref.6af, %Core.import_ref.33a, %Core.import_ref.bb5, %Core.import_ref.433, %Core.import_ref.6e2), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_12.1: type = splice_block %Optional [concrete = constants.%Optional.065] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.5c7, (constants.%OptionalStorage.impl_witness.ba1) [concrete = constants.%OptionalStorage.facet.872]
// CHECK:STDOUT:       %.loc11_12.2: %OptionalStorage.type = converted constants.%ptr.5c7, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.872]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.872) [concrete = constants.%Optional.065]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.a8a: @const.as.ImplicitAs.impl.%const.as.ImplicitAs.impl.Convert.type (%const.as.ImplicitAs.impl.Convert.type.a49) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @const.as.ImplicitAs.impl.%const.as.ImplicitAs.impl.Convert (constants.%const.as.ImplicitAs.impl.Convert.d5f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c8f = impl_witness_table (%Core.import_ref.a8a), @const.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.92e: @U.as_type.as.ImplicitAs.impl.1a8.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.79d) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.1a8.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.dd8)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.67f = impl_witness_table (%Core.import_ref.92e), @U.as_type.as.ImplicitAs.impl.1a8 [concrete]
// CHECK:STDOUT:   %Core.import_ref.33c: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.072) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.6f5)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.45f = impl_witness_table (%Core.import_ref.33c), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.f25 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.f25 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %const.b9a = var %p.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %const.b9a = call %G.ref()
// CHECK:STDOUT:   assign %p.var, %G.call
// CHECK:STDOUT:   %.loc10: type = splice_block %const [concrete = constants.%const.b9a] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr: type = ptr_type %S.ref [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:     %const: type = const_type %ptr [concrete = constants.%const.b9a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %const.b9a = ref_binding p, %p.var
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %p.ref: ref %const.b9a = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.657 = impl_witness_access constants.%ImplicitAs.impl_witness.2c2, element0 [concrete = constants.%const.as.ImplicitAs.impl.Convert.4ce]
// CHECK:STDOUT:   %bound_method.loc11_11.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @const.as.ImplicitAs.impl.Convert(constants.%Optional.065, constants.%ImplicitAs.facet.a29) [concrete = constants.%const.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_11.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %.loc11_11.1: %const.b9a = acquire_value %p.ref
// CHECK:STDOUT:   %const.as.ImplicitAs.impl.Convert.call: init %Optional.065 = call %bound_method.loc11_11.2(%.loc11_11.1)
// CHECK:STDOUT:   %.loc11_11.2: init %Optional.065 = converted %p.ref, %const.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_11.3: ref %Optional.065 = temporary_storage
// CHECK:STDOUT:   %.loc11_11.4: ref %Optional.065 = temporary %.loc11_11.3, %.loc11_11.2
// CHECK:STDOUT:   %.loc11_11.5: %Optional.065 = acquire_value %.loc11_11.4
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc11_11.5)
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %.loc11_11.4, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%.loc11_11.4)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10: <bound method> = bound_method %p.var, constants.%Destroy.Op.651ba6.4
// CHECK:STDOUT:   %Destroy.Op.call.loc10: init %empty_tuple.type = call %Destroy.Op.bound.loc10(%p.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_11.1(%self.param: ref %MaybeUnformed.74a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_11.2(%self.param: ref %Optional.065) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10(%self.param: ref %const.b9a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_const_nullable_pointer_param_using_non_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.259: type = pattern_type %ptr.5c7 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.542: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.e1f: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.617: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.e1f) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2e9: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.617) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.c7a: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.38a: %ptr.as.OptionalStorage.impl.Some.type.c7a = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8d7fae.1: <witness> = custom_witness (%Destroy.Op.651ba6.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.021: %Destroy.type = facet_value %ptr.5c7, (%custom_witness.8d7fae.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.74a: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.021) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.ba1: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.bee, @ptr.as.OptionalStorage.impl(%S) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.872: %OptionalStorage.type = facet_value %ptr.5c7, (%OptionalStorage.impl_witness.ba1) [concrete]
// CHECK:STDOUT:   %Optional.065: type = class_type @Optional, @Optional(%OptionalStorage.facet.872) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1a0: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.065)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.ea5: type = facet_type <@OptionalAs, @OptionalAs(%T.542)> [symbolic]
// CHECK:STDOUT:   %U.2d2: %OptionalAs.type.ea5 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.79d: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.1a8(%T.542, %U.2d2) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.dd8: %U.as_type.as.ImplicitAs.impl.Convert.type.79d = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.8cb: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.872)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.072: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.542) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.6f5: %T.as_type.as.OptionalAs.impl.Convert.type.072 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.83b: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.45f, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.872) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.8cb = facet_value %ptr.5c7, (%OptionalAs.impl_witness.83b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.de6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.67f, @U.as_type.as.ImplicitAs.impl.1a8(%OptionalStorage.facet.872, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.b4e: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.1a8(%OptionalStorage.facet.872, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.ad4: %U.as_type.as.ImplicitAs.impl.Convert.type.b4e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a29: %ImplicitAs.type.1a0 = facet_value %ptr.5c7, (%ImplicitAs.impl_witness.de6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.673: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.065, %ImplicitAs.facet.a29) [concrete]
// CHECK:STDOUT:   %.8ca: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.673, %ImplicitAs.facet.a29 [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.as_type.as.ImplicitAs.impl.Convert.ad4, @U.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.872, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc11_11.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.6af: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.2e9)]
// CHECK:STDOUT:   %Core.import_ref.33a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.bb5: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.c7a) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.38a)]
// CHECK:STDOUT:   %Core.import_ref.433 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6e2 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.bee = impl_witness_table (%Core.import_ref.6af, %Core.import_ref.33a, %Core.import_ref.bb5, %Core.import_ref.433, %Core.import_ref.6e2), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_12.1: type = splice_block %Optional [concrete = constants.%Optional.065] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.5c7, (constants.%OptionalStorage.impl_witness.ba1) [concrete = constants.%OptionalStorage.facet.872]
// CHECK:STDOUT:       %.loc11_12.2: %OptionalStorage.type = converted constants.%ptr.5c7, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.872]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.872) [concrete = constants.%Optional.065]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.92e: @U.as_type.as.ImplicitAs.impl.1a8.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.79d) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.1a8.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.dd8)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.67f = impl_witness_table (%Core.import_ref.92e), @U.as_type.as.ImplicitAs.impl.1a8 [concrete]
// CHECK:STDOUT:   %Core.import_ref.33c: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.072) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.6f5)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.45f = impl_witness_table (%Core.import_ref.33c), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.259 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.259 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.5c7 = var %p.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %ptr.5c7 = call %G.ref()
// CHECK:STDOUT:   assign %p.var, %G.call
// CHECK:STDOUT:   %.loc10: type = splice_block %ptr [concrete = constants.%ptr.5c7] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr: type = ptr_type %S.ref [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.5c7 = ref_binding p, %p.var
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %p.ref: ref %ptr.5c7 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.8ca = impl_witness_access constants.%ImplicitAs.impl_witness.de6, element0 [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.ad4]
// CHECK:STDOUT:   %bound_method.loc11_11.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @U.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.872, constants.%OptionalAs.facet) [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_11.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %.loc11_11.1: %ptr.5c7 = acquire_value %p.ref
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.065 = call %bound_method.loc11_11.2(%.loc11_11.1)
// CHECK:STDOUT:   %.loc11_11.2: init %Optional.065 = converted %p.ref, %U.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_11.3: ref %Optional.065 = temporary_storage
// CHECK:STDOUT:   %.loc11_11.4: ref %Optional.065 = temporary %.loc11_11.3, %.loc11_11.2
// CHECK:STDOUT:   %.loc11_11.5: %Optional.065 = acquire_value %.loc11_11.4
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc11_11.5)
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %.loc11_11.4, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%.loc11_11.4)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10: <bound method> = bound_method %p.var, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc10: init %empty_tuple.type = call %Destroy.Op.bound.loc10(%p.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_11.1(%self.param: ref %MaybeUnformed.74a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_11.2(%self.param: ref %Optional.065) {
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
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %ptr.dfe: type = ptr_type %ptr.5c7 [concrete]
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
// CHECK:STDOUT:   %foo.call.loc10: init %ptr.dfe = call imports.%foo.decl()
// CHECK:STDOUT:   %.loc10_31.1: %ptr.dfe = value_of_initializer %foo.call.loc10
// CHECK:STDOUT:   %.loc10_31.2: %ptr.dfe = converted %foo.call.loc10, %.loc10_31.1
// CHECK:STDOUT:   %IngestDoublePointer.call: init %empty_tuple.type = call %IngestDoublePointer.ref(%.loc10_31.2)
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc11: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call.loc11: init %ptr.dfe = call imports.%foo.decl()
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
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.542: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.e1f: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.617: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.e1f) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2e9: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.617) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.c7a: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.38a: %ptr.as.OptionalStorage.impl.Some.type.c7a = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8d7fae.1: <witness> = custom_witness (%Destroy.Op.651ba6.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.021: %Destroy.type = facet_value %ptr.5c7, (%custom_witness.8d7fae.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.74a: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.021) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.ba1: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.bee, @ptr.as.OptionalStorage.impl(%S) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.872: %OptionalStorage.type = facet_value %ptr.5c7, (%OptionalStorage.impl_witness.ba1) [concrete]
// CHECK:STDOUT:   %Optional.065: type = class_type @Optional, @Optional(%OptionalStorage.facet.872) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1a0: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.065)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.ea5: type = facet_type <@OptionalAs, @OptionalAs(%T.542)> [symbolic]
// CHECK:STDOUT:   %U.2d2: %OptionalAs.type.ea5 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.79d: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.1a8(%T.542, %U.2d2) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.dd8: %U.as_type.as.ImplicitAs.impl.Convert.type.79d = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.8cb: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.872)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.072: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.542) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.6f5: %T.as_type.as.OptionalAs.impl.Convert.type.072 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.83b: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.45f, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.872) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.8cb = facet_value %ptr.5c7, (%OptionalAs.impl_witness.83b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.de6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.67f, @U.as_type.as.ImplicitAs.impl.1a8(%OptionalStorage.facet.872, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.b4e: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.1a8(%OptionalStorage.facet.872, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.ad4: %U.as_type.as.ImplicitAs.impl.Convert.type.b4e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a29: %ImplicitAs.type.1a0 = facet_value %ptr.5c7, (%ImplicitAs.impl_witness.de6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.673: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.065, %ImplicitAs.facet.a29) [concrete]
// CHECK:STDOUT:   %.8ca: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.673, %ImplicitAs.facet.a29 [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.as_type.as.ImplicitAs.impl.Convert.ad4, @U.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.872, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc9_11.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.6af: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.2e9)]
// CHECK:STDOUT:   %Core.import_ref.33a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.bb5: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.c7a) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.38a)]
// CHECK:STDOUT:   %Core.import_ref.433 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6e2 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.bee = impl_witness_table (%Core.import_ref.6af, %Core.import_ref.33a, %Core.import_ref.bb5, %Core.import_ref.433, %Core.import_ref.6e2), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc9_13.1: type = splice_block %Optional [concrete = constants.%Optional.065] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.5c7, (constants.%OptionalStorage.impl_witness.ba1) [concrete = constants.%OptionalStorage.facet.872]
// CHECK:STDOUT:       %.loc9_13.2: %OptionalStorage.type = converted constants.%ptr.5c7, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.872]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.872) [concrete = constants.%Optional.065]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.92e: @U.as_type.as.ImplicitAs.impl.1a8.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.79d) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.1a8.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.dd8)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.67f = impl_witness_table (%Core.import_ref.92e), @U.as_type.as.ImplicitAs.impl.1a8 [concrete]
// CHECK:STDOUT:   %Core.import_ref.33c: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.072) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.6f5)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.45f = impl_witness_table (%Core.import_ref.33c), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
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
// CHECK:STDOUT:   %addr: %ptr.5c7 = addr_of %s.ref
// CHECK:STDOUT:   %impl.elem0: %.8ca = impl_witness_access constants.%ImplicitAs.impl_witness.de6, element0 [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.ad4]
// CHECK:STDOUT:   %bound_method.loc9_11.1: <bound method> = bound_method %addr, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @U.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.872, constants.%OptionalAs.facet) [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_11.2: <bound method> = bound_method %addr, %specific_fn
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.065 = call %bound_method.loc9_11.2(%addr)
// CHECK:STDOUT:   %.loc9_11.1: init %Optional.065 = converted %addr, %U.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc9_11.2: ref %Optional.065 = temporary_storage
// CHECK:STDOUT:   %.loc9_11.3: ref %Optional.065 = temporary %.loc9_11.2, %.loc9_11.1
// CHECK:STDOUT:   %.loc9_11.4: %Optional.065 = acquire_value %.loc9_11.3
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc9_11.4)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc9_11.3, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc9_11.3)
// CHECK:STDOUT:   %S.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_11.1(%self.param: ref %MaybeUnformed.74a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_11.2(%self.param: ref %Optional.065) {
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
// CHECK:STDOUT:   %T.542: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.None.type.fc5: type = fn_type @Optional.None, @Optional(%T.542) [symbolic]
// CHECK:STDOUT:   %Optional.None.fcb: %Optional.None.type.fc5 = struct_value () [symbolic]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.e1f: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.617: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.e1f) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2e9: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.617) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.type.03c: type = fn_type @ptr.as.OptionalStorage.impl.None, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.f37: %ptr.as.OptionalStorage.impl.None.type.03c = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8d7fae.1: <witness> = custom_witness (%Destroy.Op.651ba6.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.021: %Destroy.type = facet_value %ptr.5c7, (%custom_witness.8d7fae.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.74a: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.021) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.a7b: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.75b, @ptr.as.OptionalStorage.impl(%S) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.5c7, (%OptionalStorage.impl_witness.a7b) [concrete]
// CHECK:STDOUT:   %Optional.7f5: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.type.060: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.c13: %Optional.None.type.060 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %Optional.None.c13, @Optional.None(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc8_38.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.6af2: @Optional.%Optional.None.type (%Optional.None.type.fc5) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @Optional.%Optional.None (constants.%Optional.None.fcb)]
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.6af0: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.2e9)]
// CHECK:STDOUT:   %Core.import_ref.1cf: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None.type (%ptr.as.OptionalStorage.impl.None.type.03c) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None (constants.%ptr.as.OptionalStorage.impl.None.f37)]
// CHECK:STDOUT:   %Core.import_ref.7cb = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.433 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6e2 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.75b = impl_witness_table (%Core.import_ref.6af0, %Core.import_ref.1cf, %Core.import_ref.7cb, %Core.import_ref.433, %Core.import_ref.6e2), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc8_39.1: type = splice_block %Optional [concrete = constants.%Optional.7f5] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.5c7, (constants.%OptionalStorage.impl_witness.a7b) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %.loc8_39.2: %OptionalStorage.type = converted constants.%ptr.5c7, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.7f5]
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
// CHECK:STDOUT:   %ptr: type = ptr_type %S.ref [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.a7b) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc8_31: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.7f5]
// CHECK:STDOUT:   %.loc8_32: %Optional.None.type.060 = specific_constant imports.%Core.import_ref.6af2, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.c13]
// CHECK:STDOUT:   %None.ref: %Optional.None.type.060 = name_ref None, %.loc8_32 [concrete = constants.%Optional.None.c13]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %None.ref, @Optional.None(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.specific_fn]
// CHECK:STDOUT:   %Optional.None.call: init %Optional.7f5 = call %Optional.None.specific_fn()
// CHECK:STDOUT:   %.loc8_38.1: ref %Optional.7f5 = temporary_storage
// CHECK:STDOUT:   %.loc8_38.2: ref %Optional.7f5 = temporary %.loc8_38.1, %Optional.None.call
// CHECK:STDOUT:   %.loc8_38.3: %Optional.7f5 = acquire_value %.loc8_38.2
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc8_38.3)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc8_38.2, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc8_38.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_38.1(%self.param: ref %MaybeUnformed.74a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_38.2(%self.param: ref %Optional.7f5) {
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
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Optional.type: type = generic_class_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.542: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.eaa: type = fn_type @Optional.Some, @Optional(%T.542) [symbolic]
// CHECK:STDOUT:   %Optional.Some.6ca: %Optional.Some.type.eaa = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.e1f: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.617: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.e1f) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2e9: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.617) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.c7a: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.38a: %ptr.as.OptionalStorage.impl.Some.type.c7a = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8d7fae.1: <witness> = custom_witness (%Destroy.Op.651ba6.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.021: %Destroy.type = facet_value %ptr.5c7, (%custom_witness.8d7fae.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.74a: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.021) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.ba1: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.bee, @ptr.as.OptionalStorage.impl(%S) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.5c7, (%OptionalStorage.impl_witness.ba1) [concrete]
// CHECK:STDOUT:   %Optional.065: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.Some.type.031: type = fn_type @Optional.Some, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.Some.16d: %Optional.Some.type.031 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Some.specific_fn: <specific function> = specific_function %Optional.Some.16d, @Optional.Some(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc9_40.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.955: @Optional.%Optional.Some.type (%Optional.Some.type.eaa) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @Optional.%Optional.Some (constants.%Optional.Some.6ca)]
// CHECK:STDOUT:   %Core.import_ref.6af: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.2e9)]
// CHECK:STDOUT:   %Core.import_ref.33a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.bb5: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.c7a) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.38a)]
// CHECK:STDOUT:   %Core.import_ref.433 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6e2 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.bee = impl_witness_table (%Core.import_ref.6af, %Core.import_ref.33a, %Core.import_ref.bb5, %Core.import_ref.433, %Core.import_ref.6e2), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc9_41.1: type = splice_block %Optional [concrete = constants.%Optional.065] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.5c7, (constants.%OptionalStorage.impl_witness.ba1) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %.loc9_41.2: %OptionalStorage.type = converted constants.%ptr.5c7, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.065]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
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
// CHECK:STDOUT:   %ptr: type = ptr_type %S.ref.loc9 [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:   %OptionalStorage.facet.loc9_31: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.ba1) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc9_31: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet.loc9_31 [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.065]
// CHECK:STDOUT:   %.loc9_32: %Optional.Some.type.031 = specific_constant imports.%Core.import_ref.955, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.Some.16d]
// CHECK:STDOUT:   %Some.ref: %Optional.Some.type.031 = name_ref Some, %.loc9_32 [concrete = constants.%Optional.Some.16d]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %addr: %ptr.5c7 = addr_of %s.ref
// CHECK:STDOUT:   %OptionalStorage.facet.loc9_40: %OptionalStorage.type = facet_value constants.%ptr.5c7, (constants.%OptionalStorage.impl_witness.ba1) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc9_40.1: %OptionalStorage.type = converted constants.%ptr.5c7, %OptionalStorage.facet.loc9_40 [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %Optional.Some.specific_fn: <specific function> = specific_function %Some.ref, @Optional.Some(constants.%OptionalStorage.facet) [concrete = constants.%Optional.Some.specific_fn]
// CHECK:STDOUT:   %Optional.Some.call: init %Optional.065 = call %Optional.Some.specific_fn(%addr)
// CHECK:STDOUT:   %.loc9_40.2: ref %Optional.065 = temporary_storage
// CHECK:STDOUT:   %.loc9_40.3: ref %Optional.065 = temporary %.loc9_40.2, %Optional.Some.call
// CHECK:STDOUT:   %.loc9_40.4: %Optional.065 = acquire_value %.loc9_40.3
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc9_40.4)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc9_40.3, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc9_40.3)
// CHECK:STDOUT:   %S.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_40.1(%self.param: ref %MaybeUnformed.74a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_40.2(%self.param: ref %Optional.065) {
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
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.e1f: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.617: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.e1f) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2e9: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.617) [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8d7fae.1: <witness> = custom_witness (%Destroy.Op.651ba6.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.021: %Destroy.type = facet_value %ptr.5c7, (%custom_witness.8d7fae.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.74a: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.021) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.723: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.d23, @ptr.as.OptionalStorage.impl(%S) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.5c7, (%OptionalStorage.impl_witness.723) [concrete]
// CHECK:STDOUT:   %Optional.ece: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %get.type: type = fn_type @get [concrete]
// CHECK:STDOUT:   %get: %get.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc8_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.6af: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.2e9)]
// CHECK:STDOUT:   %Core.import_ref.33a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.7cb = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.433 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6e2 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.d23 = impl_witness_table (%Core.import_ref.6af, %Core.import_ref.33a, %Core.import_ref.7cb, %Core.import_ref.433, %Core.import_ref.6e2), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %get.decl: %get.type = fn_decl @get [concrete = constants.%get] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.5c7, (constants.%OptionalStorage.impl_witness.723) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %.loc8: %OptionalStorage.type = converted constants.%ptr.5c7, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.ece]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc8_20.1: type = splice_block %Optional [concrete = constants.%Optional.ece] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.5c7, (constants.%OptionalStorage.impl_witness.723) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %.loc8_20.2: %OptionalStorage.type = converted constants.%ptr.5c7, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.ece]
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
// CHECK:STDOUT:   %get.call: init %Optional.ece = call imports.%get.decl()
// CHECK:STDOUT:   %.loc8_19.1: ref %Optional.ece = temporary_storage
// CHECK:STDOUT:   %.loc8_19.2: ref %Optional.ece = temporary %.loc8_19.1, %get.call
// CHECK:STDOUT:   %.loc8_19.3: %Optional.ece = acquire_value %.loc8_19.2
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc8_19.3)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc8_19.2, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc8_19.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_19.1(%self.param: ref %MaybeUnformed.74a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_19.2(%self.param: ref %Optional.ece) {
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
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.259: type = pattern_type %ptr.5c7 [concrete]
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
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
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
// CHECK:STDOUT:     %p.patt: %pattern_type.259 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %addr: %ptr.5c7 = addr_of %s.ref
// CHECK:STDOUT:   %foo.call: init %ptr.5c7 = call imports.%foo.decl(%addr)
// CHECK:STDOUT:   %.loc13_22: type = splice_block %ptr [concrete = constants.%ptr.5c7] {
// CHECK:STDOUT:     %Cpp.ref.loc13_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc13: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr: type = ptr_type %S.ref.loc13 [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_36.1: %ptr.5c7 = value_of_initializer %foo.call
// CHECK:STDOUT:   %.loc13_36.2: %ptr.5c7 = converted %foo.call, %.loc13_36.1
// CHECK:STDOUT:   %p: %ptr.5c7 = value_binding p, %.loc13_36.2
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
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %Direct.cpp_overload_set.type: type = cpp_overload_set_type @Direct.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Direct.cpp_overload_set.value: %Direct.cpp_overload_set.type = cpp_overload_set_value @Direct.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %Optional.type: type = generic_class_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.542: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.e1f: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.617: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.e1f) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2e9: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.617) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.c7a: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.38a: %ptr.as.OptionalStorage.impl.Some.type.c7a = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8d7fae.1: <witness> = custom_witness (%Destroy.Op.651ba6.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.021: %Destroy.type = facet_value %ptr.5c7, (%custom_witness.8d7fae.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.74a: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.021) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.ba1: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.bee, @ptr.as.OptionalStorage.impl(%S) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.872: %OptionalStorage.type = facet_value %ptr.5c7, (%OptionalStorage.impl_witness.ba1) [concrete]
// CHECK:STDOUT:   %Optional.065: type = class_type @Optional, @Optional(%OptionalStorage.facet.872) [concrete]
// CHECK:STDOUT:   %pattern_type.70a: type = pattern_type %Optional.065 [concrete]
// CHECK:STDOUT:   %Direct.type: type = fn_type @Direct [concrete]
// CHECK:STDOUT:   %Direct: %Direct.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1a0: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.065)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.ea5: type = facet_type <@OptionalAs, @OptionalAs(%T.542)> [symbolic]
// CHECK:STDOUT:   %U.2d2: %OptionalAs.type.ea5 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.79d: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.1a8(%T.542, %U.2d2) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.dd8: %U.as_type.as.ImplicitAs.impl.Convert.type.79d = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.8cb: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.872)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.072: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.542) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.6f5: %T.as_type.as.OptionalAs.impl.Convert.type.072 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.83b: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.45f, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.872) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.8cb = facet_value %ptr.5c7, (%OptionalAs.impl_witness.83b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.de6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.67f, @U.as_type.as.ImplicitAs.impl.1a8(%OptionalStorage.facet.872, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.b4e: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.1a8(%OptionalStorage.facet.872, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.ad4: %U.as_type.as.ImplicitAs.impl.Convert.type.b4e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a29: %ImplicitAs.type.1a0 = facet_value %ptr.5c7, (%ImplicitAs.impl_witness.de6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.673: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.065, %ImplicitAs.facet.a29) [concrete]
// CHECK:STDOUT:   %.8ca: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.673, %ImplicitAs.facet.a29 [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.as_type.as.ImplicitAs.impl.Convert.ad4, @U.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.872, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %Indirect.cpp_overload_set.type: type = cpp_overload_set_type @Indirect.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Indirect.cpp_overload_set.value: %Indirect.cpp_overload_set.type = cpp_overload_set_value @Indirect.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Indirect__carbon_thunk.type: type = fn_type @Indirect__carbon_thunk [concrete]
// CHECK:STDOUT:   %Indirect__carbon_thunk: %Indirect__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.6ef: ref %S = temporary invalid, %S.val [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc13_65.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.6af: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.2e9)]
// CHECK:STDOUT:   %Core.import_ref.33a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.bb5: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.c7a) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.38a)]
// CHECK:STDOUT:   %Core.import_ref.433 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6e2 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.bee = impl_witness_table (%Core.import_ref.6af, %Core.import_ref.33a, %Core.import_ref.bb5, %Core.import_ref.433, %Core.import_ref.6e2), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Direct.decl: %Direct.type = fn_decl @Direct [concrete = constants.%Direct] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %OptionalStorage.facet.loc11_16.1: %OptionalStorage.type = facet_value constants.%ptr.5c7, (constants.%OptionalStorage.impl_witness.ba1) [concrete = constants.%OptionalStorage.facet.872]
// CHECK:STDOUT:     %.loc11_16.1: %OptionalStorage.type = converted constants.%ptr.5c7, %OptionalStorage.facet.loc11_16.1 [concrete = constants.%OptionalStorage.facet.872]
// CHECK:STDOUT:     %Optional.loc11_16.1: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.872) [concrete = constants.%Optional.065]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_16.2: type = splice_block %Optional.loc11_16.2 [concrete = constants.%Optional.065] {
// CHECK:STDOUT:       %OptionalStorage.facet.loc11_16.2: %OptionalStorage.type = facet_value constants.%ptr.5c7, (constants.%OptionalStorage.impl_witness.ba1) [concrete = constants.%OptionalStorage.facet.872]
// CHECK:STDOUT:       %.loc11_16.3: %OptionalStorage.type = converted constants.%ptr.5c7, %OptionalStorage.facet.loc11_16.2 [concrete = constants.%OptionalStorage.facet.872]
// CHECK:STDOUT:       %Optional.loc11_16.2: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.872) [concrete = constants.%Optional.065]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.92e: @U.as_type.as.ImplicitAs.impl.1a8.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.79d) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.1a8.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.dd8)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.67f = impl_witness_table (%Core.import_ref.92e), @U.as_type.as.ImplicitAs.impl.1a8 [concrete]
// CHECK:STDOUT:   %Core.import_ref.33c: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.072) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.6f5)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.45f = impl_witness_table (%Core.import_ref.33c), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT:   %Indirect.cpp_overload_set.value: %Indirect.cpp_overload_set.type = cpp_overload_set_value @Indirect.cpp_overload_set [concrete = constants.%Indirect.cpp_overload_set.value]
// CHECK:STDOUT:   %Indirect__carbon_thunk.decl: %Indirect__carbon_thunk.type = fn_decl @Indirect__carbon_thunk [concrete = constants.%Indirect__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.5c7, (constants.%OptionalStorage.impl_witness.ba1) [concrete = constants.%OptionalStorage.facet.872]
// CHECK:STDOUT:     %.loc13: %OptionalStorage.type = converted constants.%ptr.5c7, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.872]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.872) [concrete = constants.%Optional.065]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
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
// CHECK:STDOUT:   %addr.loc11: %ptr.5c7 = addr_of %s.ref
// CHECK:STDOUT:   %impl.elem0: %.8ca = impl_witness_access constants.%ImplicitAs.impl_witness.de6, element0 [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.ad4]
// CHECK:STDOUT:   %bound_method.loc11_14.1: <bound method> = bound_method %addr.loc11, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @U.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.872, constants.%OptionalAs.facet) [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_14.2: <bound method> = bound_method %addr.loc11, %specific_fn
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.065 = call %bound_method.loc11_14.2(%addr.loc11)
// CHECK:STDOUT:   %.loc11_14.1: init %Optional.065 = converted %addr.loc11, %U.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_14.2: ref %Optional.065 = temporary_storage
// CHECK:STDOUT:   %.loc11_14.3: ref %Optional.065 = temporary %.loc11_14.2, %.loc11_14.1
// CHECK:STDOUT:   %.loc11_14.4: %Optional.065 = acquire_value %.loc11_14.3
// CHECK:STDOUT:   %Direct.call: init %Optional.065 = call imports.%Direct.decl(%.loc11_14.4)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.70a = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13_41: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Indirect.ref: %Indirect.cpp_overload_set.type = name_ref Indirect, imports.%Indirect.cpp_overload_set.value [concrete = constants.%Indirect.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc13_55.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Cpp.ref.loc13_60: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc13_63: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %.loc13_55.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc13_55.3: init %S to %.loc13_55.2 = class_init () [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc13_57.1: init %S = converted %.loc13_55.1, %.loc13_55.3 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc13_57.2: ref %S = temporary %.loc13_55.2, %.loc13_57.1 [concrete = constants.%.6ef]
// CHECK:STDOUT:   %.loc13_57.3: %S = acquire_value %.loc13_57.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc13_57.4: ref %S = value_as_ref %.loc13_57.3
// CHECK:STDOUT:   %addr.loc13: %ptr.5c7 = addr_of %.loc13_57.4
// CHECK:STDOUT:   %Indirect__carbon_thunk.call: init %Optional.065 = call imports.%Indirect__carbon_thunk.decl(%addr.loc13)
// CHECK:STDOUT:   %.loc13_37.1: type = splice_block %Optional [concrete = constants.%Optional.065] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Cpp.ref.loc13_31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc13_34: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr: type = ptr_type %S.ref.loc13_34 [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.ba1) [concrete = constants.%OptionalStorage.facet.872]
// CHECK:STDOUT:     %.loc13_37.2: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.872]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.872) [concrete = constants.%Optional.065]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_65.1: ref %Optional.065 = temporary_storage
// CHECK:STDOUT:   %.loc13_65.2: ref %Optional.065 = temporary %.loc13_65.1, %Indirect__carbon_thunk.call
// CHECK:STDOUT:   %.loc13_65.3: %Optional.065 = acquire_value %.loc13_65.2
// CHECK:STDOUT:   %a: %Optional.065 = value_binding a, %.loc13_65.3
// CHECK:STDOUT:   %Destroy.Op.bound.loc13: <bound method> = bound_method %.loc13_65.2, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc13: init %empty_tuple.type = call %Destroy.Op.bound.loc13(%.loc13_65.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %.loc11_14.3, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%.loc11_14.3)
// CHECK:STDOUT:   %S.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_65.1(%self.param: ref %MaybeUnformed.74a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_65.2(%self.param: ref %Optional.065) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
