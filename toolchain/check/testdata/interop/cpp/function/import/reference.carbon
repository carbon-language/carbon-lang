// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/reference.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/reference.carbon

// ============================================================================
// Lvalue reference as a parameter type
// ============================================================================

// --- param_lvalue_ref.h

struct S {};
struct T {};

auto TakesLValue(S&) -> void;

// --- call_param_lvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "param_lvalue_ref.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.S = {};
  Cpp.TakesLValue(ref s);
  //@dump-sem-ir-end
}

// --- fail_param_lvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "param_lvalue_ref.h";

fn F() {
  //@dump-sem-ir-begin
  var v: Cpp.S;
  let s: Cpp.S = v;
  // CHECK:STDERR: fail_param_lvalue_ref.carbon:[[@LINE+8]]:20: error: no matching function for call to 'TakesLValue' [CppInteropParseError]
  // CHECK:STDERR:    18 |   Cpp.TakesLValue(s);
  // CHECK:STDERR:       |                    ^
  // CHECK:STDERR: fail_param_lvalue_ref.carbon:[[@LINE-9]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_lvalue_ref.h:5:6: note: candidate function not viable: 1st argument ('const S') would lose const qualifier [CppInteropParseNote]
  // CHECK:STDERR:     5 | auto TakesLValue(S&) -> void;
  // CHECK:STDERR:       |      ^           ~~
  // CHECK:STDERR:
  Cpp.TakesLValue(s);

  // CHECK:STDERR: fail_param_lvalue_ref.carbon:[[@LINE+8]]:35: error: no matching function for call to 'TakesLValue' [CppInteropParseError]
  // CHECK:STDERR:    28 |   Cpp.TakesLValue(s as const Cpp.S);
  // CHECK:STDERR:       |                                   ^
  // CHECK:STDERR: fail_param_lvalue_ref.carbon:[[@LINE-19]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_lvalue_ref.h:5:6: note: candidate function not viable: 1st argument ('const S') would lose const qualifier [CppInteropParseNote]
  // CHECK:STDERR:     5 | auto TakesLValue(S&) -> void;
  // CHECK:STDERR:       |      ^           ~~
  // CHECK:STDERR:
  Cpp.TakesLValue(s as const Cpp.S);

  var t: Cpp.T;
  // CHECK:STDERR: fail_param_lvalue_ref.carbon:[[@LINE+8]]:20: error: no matching function for call to 'TakesLValue' [CppInteropParseError]
  // CHECK:STDERR:    39 |   Cpp.TakesLValue(t);
  // CHECK:STDERR:       |                    ^
  // CHECK:STDERR: fail_param_lvalue_ref.carbon:[[@LINE-30]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_lvalue_ref.h:5:6: note: candidate function not viable: no known conversion from 'T' to 'S &' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | auto TakesLValue(S&) -> void;
  // CHECK:STDERR:       |      ^           ~~
  // CHECK:STDERR:
  Cpp.TakesLValue(t);

  var u: Cpp.S;
  // CHECK:STDERR: fail_param_lvalue_ref.carbon:[[@LINE+8]]:35: error: no matching function for call to 'TakesLValue' [CppInteropParseError]
  // CHECK:STDERR:    50 |   Cpp.TakesLValue(u as const Cpp.S);
  // CHECK:STDERR:       |                                   ^
  // CHECK:STDERR: fail_param_lvalue_ref.carbon:[[@LINE-41]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_lvalue_ref.h:5:6: note: candidate function not viable: 1st argument ('const S') would lose const qualifier [CppInteropParseNote]
  // CHECK:STDERR:     5 | auto TakesLValue(S&) -> void;
  // CHECK:STDERR:       |      ^           ~~
  // CHECK:STDERR:
  Cpp.TakesLValue(u as const Cpp.S);
  //@dump-sem-ir-end
}

// --- derived_to_base.carbon

library "[[@TEST_NAME]]";

import Cpp;

inline Cpp '''
struct Base { Base(const Base&) = delete; };
struct Derived : Base {};
void f(Base &b);
''';

fn F(ref r: Cpp.Derived) {
  Cpp.f(ref r);
}

// ============================================================================
// Rvalue reference as a parameter type
// ============================================================================

// --- param_rvalue_ref.h

struct S {};
struct T {};

auto TakesRValue(S&&) -> void;

// --- call_param_rvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "param_rvalue_ref.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.TakesRValue({} as Cpp.S);
  //@dump-sem-ir-end
}

// --- fail_param_value_arg_for_rvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "param_rvalue_ref.h";

fn F() {
  let s: Cpp.S = {};
  // CHECK:STDERR: fail_param_value_arg_for_rvalue_ref.carbon:[[@LINE+8]]:20: error: no matching function for call to 'TakesRValue' [CppInteropParseError]
  // CHECK:STDERR:    16 |   Cpp.TakesRValue(s);
  // CHECK:STDERR:       |                    ^
  // CHECK:STDERR: fail_param_value_arg_for_rvalue_ref.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_rvalue_ref.h:5:6: note: candidate function not viable: 1st argument ('const S') would lose const qualifier [CppInteropParseNote]
  // CHECK:STDERR:     5 | auto TakesRValue(S&&) -> void;
  // CHECK:STDERR:       |      ^           ~~~
  // CHECK:STDERR:
  Cpp.TakesRValue(s);
}

// --- fail_param_rvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "param_rvalue_ref.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.S;
  // CHECK:STDERR: fail_param_rvalue_ref.carbon:[[@LINE+8]]:20: error: no matching function for call to 'TakesRValue' [CppInteropParseError]
  // CHECK:STDERR:    17 |   Cpp.TakesRValue(s);
  // CHECK:STDERR:       |                    ^
  // CHECK:STDERR: fail_param_rvalue_ref.carbon:[[@LINE-8]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_rvalue_ref.h:5:6: note: candidate function not viable: expects an rvalue for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | auto TakesRValue(S&&) -> void;
  // CHECK:STDERR:       |      ^           ~~~
  // CHECK:STDERR:
  Cpp.TakesRValue(s);

  var t: Cpp.T;
  // CHECK:STDERR: fail_param_rvalue_ref.carbon:[[@LINE+8]]:20: error: no matching function for call to 'TakesRValue' [CppInteropParseError]
  // CHECK:STDERR:    28 |   Cpp.TakesRValue(t);
  // CHECK:STDERR:       |                    ^
  // CHECK:STDERR: fail_param_rvalue_ref.carbon:[[@LINE-19]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_rvalue_ref.h:5:6: note: candidate function not viable: no known conversion from 'T' to 'S' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | auto TakesRValue(S&&) -> void;
  // CHECK:STDERR:       |      ^           ~~~
  // CHECK:STDERR:
  Cpp.TakesRValue(t);

  // CHECK:STDERR: fail_param_rvalue_ref.carbon:[[@LINE+8]]:47: error: no matching function for call to 'TakesRValue' [CppInteropParseError]
  // CHECK:STDERR:    38 |   Cpp.TakesRValue(({} as Cpp.S) as const Cpp.S);
  // CHECK:STDERR:       |                                               ^
  // CHECK:STDERR: fail_param_rvalue_ref.carbon:[[@LINE-29]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_rvalue_ref.h:5:6: note: candidate function not viable: 1st argument ('const S') would lose const qualifier [CppInteropParseNote]
  // CHECK:STDERR:     5 | auto TakesRValue(S&&) -> void;
  // CHECK:STDERR:       |      ^           ~~~
  // CHECK:STDERR:
  Cpp.TakesRValue(({} as Cpp.S) as const Cpp.S);
  //@dump-sem-ir-end
}

// ============================================================================
// Const reference as a parameter type
// ============================================================================

// --- param_const_lvalue_ref.h

struct S {};
struct T {};

auto TakesConstLValue(const S&) -> void;

// --- call_param_const_lvalue_ref_with_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "param_const_lvalue_ref.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.S = {};
  Cpp.TakesConstLValue(s as const Cpp.S);

  Cpp.TakesConstLValue(s);
  //@dump-sem-ir-end
}

// --- call_param_const_lvalue_ref_with_value.carbon

library "[[@TEST_NAME]]";

import Cpp library "param_const_lvalue_ref.h";

fn F() {
  //@dump-sem-ir-begin
  let s: Cpp.S = {};
  Cpp.TakesConstLValue(s);

  Cpp.TakesConstLValue(s as const Cpp.S);
  //@dump-sem-ir-end
}

// --- fail_call_param_const_lvalue_ref_with_wrong_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "param_const_lvalue_ref.h";

fn F() {
  var t: Cpp.T;
  // CHECK:STDERR: fail_call_param_const_lvalue_ref_with_wrong_type.carbon:[[@LINE+8]]:25: error: no matching function for call to 'TakesConstLValue' [CppInteropParseError]
  // CHECK:STDERR:    16 |   Cpp.TakesConstLValue(t);
  // CHECK:STDERR:       |                         ^
  // CHECK:STDERR: fail_call_param_const_lvalue_ref_with_wrong_type.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_const_lvalue_ref.h:5:6: note: candidate function not viable: no known conversion from 'T' to 'const S' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | auto TakesConstLValue(const S&) -> void;
  // CHECK:STDERR:       |      ^                ~~~~~~~~
  // CHECK:STDERR:
  Cpp.TakesConstLValue(t);
}

// ============================================================================
// Const rvalue reference as a parameter type
// ============================================================================

// --- param_const_rvalue_ref.h

struct S {};
struct T {};

auto TakesConstRValue(const S&&) -> void;

// --- param_value_arg_for_const_rvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "param_const_rvalue_ref.h";

fn F(v: Cpp.S, cv: const Cpp.S) {
  //@dump-sem-ir-begin
  Cpp.TakesConstRValue(v);
  Cpp.TakesConstRValue(cv);
  //@dump-sem-ir-end
}

// --- param_init_arg_for_const_rvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "param_const_rvalue_ref.h";

fn Make() -> Cpp.S;

fn F() {
  Cpp.TakesConstRValue({});
  Cpp.TakesConstRValue({} as Cpp.S);
  Cpp.TakesConstRValue(({} as Cpp.S) as const Cpp.S);
  Cpp.TakesConstRValue(Make());
}

// --- fail_param_lvalue_ref_arg_const_rvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "param_const_rvalue_ref.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.S;
  // CHECK:STDERR: fail_param_lvalue_ref_arg_const_rvalue_ref.carbon:[[@LINE+8]]:25: error: no matching function for call to 'TakesConstRValue' [CppInteropParseError]
  // CHECK:STDERR:    17 |   Cpp.TakesConstRValue(s);
  // CHECK:STDERR:       |                         ^
  // CHECK:STDERR: fail_param_lvalue_ref_arg_const_rvalue_ref.carbon:[[@LINE-8]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_const_rvalue_ref.h:5:6: note: candidate function not viable: expects an rvalue for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | auto TakesConstRValue(const S&&) -> void;
  // CHECK:STDERR:       |      ^                ~~~~~~~~~
  // CHECK:STDERR:
  Cpp.TakesConstRValue(s);

  var t: Cpp.T;
  // CHECK:STDERR: fail_param_lvalue_ref_arg_const_rvalue_ref.carbon:[[@LINE+8]]:25: error: no matching function for call to 'TakesConstRValue' [CppInteropParseError]
  // CHECK:STDERR:    28 |   Cpp.TakesConstRValue(t);
  // CHECK:STDERR:       |                         ^
  // CHECK:STDERR: fail_param_lvalue_ref_arg_const_rvalue_ref.carbon:[[@LINE-19]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_const_rvalue_ref.h:5:6: note: candidate function not viable: no known conversion from 'T' to 'const S' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | auto TakesConstRValue(const S&&) -> void;
  // CHECK:STDERR:       |      ^                ~~~~~~~~~
  // CHECK:STDERR:
  Cpp.TakesConstRValue(t);
  //@dump-sem-ir-end
}

// ============================================================================
// Lvalue reference as return type
// ============================================================================

// --- return_lvalue_ref.h

struct S {};

auto ReturnsLValue() -> S&;

// --- call_return_lvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "return_lvalue_ref.h";

fn F() {
  //@dump-sem-ir-begin
  let unused ref s: Cpp.S = Cpp.ReturnsLValue();
  //@dump-sem-ir-end
}

// ============================================================================
// Rvalue reference as return type
// ============================================================================

// --- return_rvalue_ref.h

struct S {};

auto ReturnsRValue() -> S&&;

// --- call_return_rvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "return_rvalue_ref.h";

fn F() {
  //@dump-sem-ir-begin
  let unused ref s: Cpp.S = Cpp.ReturnsRValue();
  //@dump-sem-ir-end
}

// ============================================================================
// Const reference as return type
// ============================================================================

// --- return_const_lvalue_ref.h

struct S {};

auto ReturnConstLValue() -> const S&;

// --- call_return_const_lvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "return_const_lvalue_ref.h";

fn F() {
  //@dump-sem-ir-begin
  let unused ref s: const Cpp.S = Cpp.ReturnConstLValue();
  //@dump-sem-ir-end
}

// --- fail_call_return_const_lvalue_ref_const_correctness.carbon

library "[[@TEST_NAME]]";

import Cpp library "return_const_lvalue_ref.h";

fn F() {
  // CHECK:STDERR: fail_call_return_const_lvalue_ref_const_correctness.carbon:[[@LINE+7]]:29: error: cannot implicitly convert expression of type `const Cpp.S` to `Cpp.S` [ConversionFailure]
  // CHECK:STDERR:   let unused ref s: Cpp.S = Cpp.ReturnConstLValue();
  // CHECK:STDERR:                             ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_call_return_const_lvalue_ref_const_correctness.carbon:[[@LINE+4]]:29: note: type `const Cpp.S` does not implement interface `Core.ImplicitAs(Cpp.S)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused ref s: Cpp.S = Cpp.ReturnConstLValue();
  // CHECK:STDERR:                             ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused ref s: Cpp.S = Cpp.ReturnConstLValue();
}

// ============================================================================
// Forwarding references
// ============================================================================

// --- forwarding_reference.carbon

library "[[@TEST_NAME]]";

import Cpp;

class X {}

inline Cpp '''c++
template<typename T> void TakeAnything_ExpectLRef(T&&) {
  static_assert(__is_same(T, Carbon::X&));
}
template<typename T> void TakeAnything_ExpectConstLRef(T&&) {
  static_assert(__is_same(T, const Carbon::X&));
}
template<typename T> void TakeAnything_ExpectRRef(T&&) {
  static_assert(__is_same(T, Carbon::X));
}
template<typename T> void TakeAnything_ExpectConstRRef(T&&) {
  static_assert(__is_same(T, const Carbon::X));
}
''';

fn Make() -> X;

fn PassValue(x: X) {
  //@dump-sem-ir-begin
  // TODO: Is a `const X&&` the right choice here?
  Cpp.TakeAnything_ExpectConstRRef(x);
  //@dump-sem-ir-end
}

fn PassRef(ref x: X) {
  //@dump-sem-ir-begin
  Cpp.TakeAnything_ExpectLRef(ref x);
  //@dump-sem-ir-end
}

fn PassConstRef(ref x: const X) {
  //@dump-sem-ir-begin
  Cpp.TakeAnything_ExpectConstLRef(x);
  //@dump-sem-ir-end
}

fn PassInit() {
  //@dump-sem-ir-begin
  Cpp.TakeAnything_ExpectRRef(Make());
  //@dump-sem-ir-end
}

// --- fail_todo_forwarding_reference_by_ref.carbon

library "[[@TEST_NAME]]";

import Cpp;

class X {}

inline Cpp '''c++
template<typename T> void TakeAnything_ExpectConstLRef(T&&) {
  static_assert(__is_same(T, const Carbon::X&));
}
''';

fn PassConstRef(ref x: const X) {
  // CHECK:STDERR: fail_todo_forwarding_reference_by_ref.carbon:[[@LINE+7]]:36: error: `ref` tag is not an argument to a `ref` parameter [RefTagNoRefParam]
  // CHECK:STDERR:   Cpp.TakeAnything_ExpectConstLRef(ref x);
  // CHECK:STDERR:                                    ^~~~~
  // CHECK:STDERR: fail_todo_forwarding_reference_by_ref.carbon:[[@LINE-9]]:59: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: template<typename T> void TakeAnything_ExpectConstLRef(T&&) {
  // CHECK:STDERR:                                                           ^
  // CHECK:STDERR:
  Cpp.TakeAnything_ExpectConstLRef(ref x);
}


// CHECK:STDOUT: --- call_param_lvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.b2d: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %TakesLValue.cpp_overload_set.type: type = cpp_overload_set_type @TakesLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesLValue.cpp_overload_set.value: %TakesLValue.cpp_overload_set.type = cpp_overload_set_value @TakesLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesLValue.type: type = fn_type @TakesLValue [concrete]
// CHECK:STDOUT:   %TakesLValue: %TakesLValue.type = struct_value () [concrete]
// CHECK:STDOUT:   %S.cpp_destructor.type: type = fn_type @S.cpp_destructor [concrete]
// CHECK:STDOUT:   %S.cpp_destructor: %S.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .TakesLValue = %TakesLValue.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %TakesLValue.cpp_overload_set.value: %TakesLValue.cpp_overload_set.type = cpp_overload_set_value @TakesLValue.cpp_overload_set [concrete = constants.%TakesLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %TakesLValue.decl: %TakesLValue.type = fn_decl @TakesLValue [concrete = constants.%TakesLValue] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.b2d = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.b2d = var_pattern %s.patt [concrete]
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
// CHECK:STDOUT:   %TakesLValue.ref: %TakesLValue.cpp_overload_set.type = name_ref TakesLValue, imports.%TakesLValue.cpp_overload_set.value [concrete = constants.%TakesLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %.loc9: %S = ref_tag %s.ref
// CHECK:STDOUT:   %TakesLValue.call: init %empty_tuple.type = call imports.%TakesLValue.decl(%s.ref)
// CHECK:STDOUT:   %S.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_param_lvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %pattern_type.b2d: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.e5a: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.38d: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.e5a) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.eab: %T.as_type.as.DefaultOrUnformed.impl.Op.type.38d = struct_value () [symbolic]
// CHECK:STDOUT:   %S.Op.type: type = fn_type @S.Op [concrete]
// CHECK:STDOUT:   %S.Op: %S.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b26fff.1: <witness> = custom_witness (%S.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.229: %Default.type = facet_value %S, (%custom_witness.b26fff.1) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.f99: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.48a, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.229) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.9ab: %DefaultOrUnformed.type = facet_value %S, (%DefaultOrUnformed.impl_witness.f99) [concrete]
// CHECK:STDOUT:   %TakesLValue.cpp_overload_set.type: type = cpp_overload_set_type @TakesLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesLValue.cpp_overload_set.value: %TakesLValue.cpp_overload_set.type = cpp_overload_set_value @TakesLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %T.3b5: type = class_type @T [concrete]
// CHECK:STDOUT:   %pattern_type.471: type = pattern_type %T.3b5 [concrete]
// CHECK:STDOUT:   %T.Op.type: type = fn_type @T.Op [concrete]
// CHECK:STDOUT:   %T.Op: %T.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b26fff.2: <witness> = custom_witness (%T.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.dcd: %Default.type = facet_value %T.3b5, (%custom_witness.b26fff.2) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.dca: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.48a, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.dcd) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.fe5: %DefaultOrUnformed.type = facet_value %T.3b5, (%DefaultOrUnformed.impl_witness.dca) [concrete]
// CHECK:STDOUT:   %S.cpp_destructor.type: type = fn_type @S.cpp_destructor [concrete]
// CHECK:STDOUT:   %S.cpp_destructor: %S.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.cpp_destructor.type: type = fn_type @T.cpp_destructor [concrete]
// CHECK:STDOUT:   %T.cpp_destructor: %T.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .TakesLValue = %TakesLValue.cpp_overload_set.value
// CHECK:STDOUT:     .T = %T.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.a03: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.38d) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.eab)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.48a = impl_witness_table (%Core.import_ref.a03), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %TakesLValue.cpp_overload_set.value: %TakesLValue.cpp_overload_set.type = cpp_overload_set_value @TakesLValue.cpp_overload_set [concrete = constants.%TakesLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %T.decl: type = class_decl @T [concrete = constants.%T.3b5] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.b2d = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.b2d = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %S = var %v.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc8: %DefaultOrUnformed.type = facet_value constants.%S, (constants.%DefaultOrUnformed.impl_witness.f99) [concrete = constants.%DefaultOrUnformed.facet.9ab]
// CHECK:STDOUT:   %.loc8_15.1: %DefaultOrUnformed.type = converted constants.%S, %DefaultOrUnformed.facet.loc8 [concrete = constants.%DefaultOrUnformed.facet.9ab]
// CHECK:STDOUT:   %as_type.loc8: type = facet_access_type %.loc8_15.1 [concrete = constants.%S]
// CHECK:STDOUT:   %.loc8_15.2: type = converted %.loc8_15.1, %as_type.loc8 [concrete = constants.%S]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc8_3: ref %S = splice_block %v.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc8: init %S to %.loc8_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn.1()
// CHECK:STDOUT:   assign %v.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc8
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref.loc8 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %S = ref_binding v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.b2d = value_binding_pattern s [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.ref: ref %S = name_ref v, %v
// CHECK:STDOUT:   %.loc9_13: type = splice_block %S.ref.loc9 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc9: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_18: %S = acquire_value %v.ref
// CHECK:STDOUT:   %s: %S = value_binding s, %.loc9_18
// CHECK:STDOUT:   %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesLValue.ref.loc18: %TakesLValue.cpp_overload_set.type = name_ref TakesLValue, imports.%TakesLValue.cpp_overload_set.value [concrete = constants.%TakesLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref.loc18: %S = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc28_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesLValue.ref.loc28: %TakesLValue.cpp_overload_set.type = name_ref TakesLValue, imports.%TakesLValue.cpp_overload_set.value [concrete = constants.%TakesLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref.loc28: %S = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc28_30: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc28: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %const.loc28: type = const_type %S.ref.loc28 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc28_21.1: %const = as_compatible %s.ref.loc28
// CHECK:STDOUT:   %.loc28_21.2: %const = converted %s.ref.loc28, %.loc28_21.1
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type.471 = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.var_patt: %pattern_type.471 = var_pattern %t.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %T.3b5 = var %t.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc30: %DefaultOrUnformed.type = facet_value constants.%T.3b5, (constants.%DefaultOrUnformed.impl_witness.dca) [concrete = constants.%DefaultOrUnformed.facet.fe5]
// CHECK:STDOUT:   %.loc30_15.1: %DefaultOrUnformed.type = converted constants.%T.3b5, %DefaultOrUnformed.facet.loc30 [concrete = constants.%DefaultOrUnformed.facet.fe5]
// CHECK:STDOUT:   %as_type.loc30: type = facet_access_type %.loc30_15.1 [concrete = constants.%T.3b5]
// CHECK:STDOUT:   %.loc30_15.2: type = converted %.loc30_15.1, %as_type.loc30 [concrete = constants.%T.3b5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc30_3: ref %T.3b5 = splice_block %t.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc30: init %T.3b5 to %.loc30_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn.2()
// CHECK:STDOUT:   assign %t.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc30
// CHECK:STDOUT:   %.loc30_13: type = splice_block %T.ref [concrete = constants.%T.3b5] {
// CHECK:STDOUT:     %Cpp.ref.loc30: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %T.ref: type = name_ref T, imports.%T.decl [concrete = constants.%T.3b5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %T.3b5 = ref_binding t, %t.var
// CHECK:STDOUT:   %Cpp.ref.loc39: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesLValue.ref.loc39: %TakesLValue.cpp_overload_set.type = name_ref TakesLValue, imports.%TakesLValue.cpp_overload_set.value [concrete = constants.%TakesLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %t.ref: ref %T.3b5 = name_ref t, %t
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %u.patt: %pattern_type.b2d = ref_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.var_patt: %pattern_type.b2d = var_pattern %u.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %u.var: ref %S = var %u.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc41: %DefaultOrUnformed.type = facet_value constants.%S, (constants.%DefaultOrUnformed.impl_witness.f99) [concrete = constants.%DefaultOrUnformed.facet.9ab]
// CHECK:STDOUT:   %.loc41_15.1: %DefaultOrUnformed.type = converted constants.%S, %DefaultOrUnformed.facet.loc41 [concrete = constants.%DefaultOrUnformed.facet.9ab]
// CHECK:STDOUT:   %as_type.loc41: type = facet_access_type %.loc41_15.1 [concrete = constants.%S]
// CHECK:STDOUT:   %.loc41_15.2: type = converted %.loc41_15.1, %as_type.loc41 [concrete = constants.%S]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc41_3: ref %S = splice_block %u.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc41: init %S to %.loc41_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn.3()
// CHECK:STDOUT:   assign %u.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc41
// CHECK:STDOUT:   %.loc41_13: type = splice_block %S.ref.loc41 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc41: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc41: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %u: ref %S = ref_binding u, %u.var
// CHECK:STDOUT:   %Cpp.ref.loc50_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesLValue.ref.loc50: %TakesLValue.cpp_overload_set.type = name_ref TakesLValue, imports.%TakesLValue.cpp_overload_set.value [concrete = constants.%TakesLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: ref %S = name_ref u, %u
// CHECK:STDOUT:   %Cpp.ref.loc50_30: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc50: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %const.loc50: type = const_type %S.ref.loc50 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc50_21.1: ref %const = as_compatible %u.ref
// CHECK:STDOUT:   %.loc50_21.2: ref %const = converted %u.ref, %.loc50_21.1
// CHECK:STDOUT:   %S.cpp_destructor.bound.loc41: <bound method> = bound_method %u.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call.loc41: init %empty_tuple.type = call %S.cpp_destructor.bound.loc41(%u.var)
// CHECK:STDOUT:   %T.cpp_destructor.bound: <bound method> = bound_method %t.var, constants.%T.cpp_destructor
// CHECK:STDOUT:   %T.cpp_destructor.call: init %empty_tuple.type = call %T.cpp_destructor.bound(%t.var)
// CHECK:STDOUT:   %S.cpp_destructor.bound.loc8: <bound method> = bound_method %v.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call.loc8: init %empty_tuple.type = call %S.cpp_destructor.bound.loc8(%v.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_param_rvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TakesRValue.cpp_overload_set.type: type = cpp_overload_set_type @TakesRValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesRValue.cpp_overload_set.value: %TakesRValue.cpp_overload_set.type = cpp_overload_set_value @TakesRValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.055: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %TakesRValue__carbon_thunk.type: type = fn_type @TakesRValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesRValue__carbon_thunk: %TakesRValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TakesRValue = %TakesRValue.cpp_overload_set.value
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakesRValue.cpp_overload_set.value: %TakesRValue.cpp_overload_set.type = cpp_overload_set_value @TakesRValue.cpp_overload_set [concrete = constants.%TakesRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %TakesRValue__carbon_thunk.decl: %TakesRValue__carbon_thunk.type = fn_decl @TakesRValue__carbon_thunk [concrete = constants.%TakesRValue__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesRValue.ref: %TakesRValue.cpp_overload_set.type = name_ref TakesRValue, imports.%TakesRValue.cpp_overload_set.value [concrete = constants.%TakesRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_20.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Cpp.ref.loc8_25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc8_20.2: init %S to %_.var = class_init () [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_22: init %S = converted %.loc8_20.1, %.loc8_20.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr: %ptr.055 = addr_of %_.var
// CHECK:STDOUT:   %TakesRValue__carbon_thunk.call: init %empty_tuple.type = call imports.%TakesRValue__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_param_rvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.b2d: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.e5a: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.38d: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.e5a) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.eab: %T.as_type.as.DefaultOrUnformed.impl.Op.type.38d = struct_value () [symbolic]
// CHECK:STDOUT:   %S.Op.type: type = fn_type @S.Op [concrete]
// CHECK:STDOUT:   %S.Op: %S.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b26fff.1: <witness> = custom_witness (%S.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.229: %Default.type = facet_value %S, (%custom_witness.b26fff.1) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.f99: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.48a, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.229) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.9ab: %DefaultOrUnformed.type = facet_value %S, (%DefaultOrUnformed.impl_witness.f99) [concrete]
// CHECK:STDOUT:   %TakesRValue.cpp_overload_set.type: type = cpp_overload_set_type @TakesRValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesRValue.cpp_overload_set.value: %TakesRValue.cpp_overload_set.type = cpp_overload_set_value @TakesRValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %T.3b5: type = class_type @T [concrete]
// CHECK:STDOUT:   %pattern_type.471: type = pattern_type %T.3b5 [concrete]
// CHECK:STDOUT:   %T.Op.type: type = fn_type @T.Op [concrete]
// CHECK:STDOUT:   %T.Op: %T.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b26fff.2: <witness> = custom_witness (%T.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.dcd: %Default.type = facet_value %T.3b5, (%custom_witness.b26fff.2) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.dca: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.48a, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.dcd) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.fe5: %DefaultOrUnformed.type = facet_value %T.3b5, (%DefaultOrUnformed.impl_witness.dca) [concrete]
// CHECK:STDOUT:   %empty_struct.a40: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %empty_struct.418: %const = struct_value () [concrete]
// CHECK:STDOUT:   %T.cpp_destructor.type: type = fn_type @T.cpp_destructor [concrete]
// CHECK:STDOUT:   %T.cpp_destructor: %T.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %S.cpp_destructor.type: type = fn_type @S.cpp_destructor [concrete]
// CHECK:STDOUT:   %S.cpp_destructor: %S.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .TakesRValue = %TakesRValue.cpp_overload_set.value
// CHECK:STDOUT:     .T = %T.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.a03: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.38d) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.eab)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.48a = impl_witness_table (%Core.import_ref.a03), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %TakesRValue.cpp_overload_set.value: %TakesRValue.cpp_overload_set.type = cpp_overload_set_value @TakesRValue.cpp_overload_set [concrete = constants.%TakesRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %T.decl: type = class_decl @T [concrete = constants.%T.3b5] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.b2d = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.b2d = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc8: %DefaultOrUnformed.type = facet_value constants.%S, (constants.%DefaultOrUnformed.impl_witness.f99) [concrete = constants.%DefaultOrUnformed.facet.9ab]
// CHECK:STDOUT:   %.loc8_15.1: %DefaultOrUnformed.type = converted constants.%S, %DefaultOrUnformed.facet.loc8 [concrete = constants.%DefaultOrUnformed.facet.9ab]
// CHECK:STDOUT:   %as_type.loc8: type = facet_access_type %.loc8_15.1 [concrete = constants.%S]
// CHECK:STDOUT:   %.loc8_15.2: type = converted %.loc8_15.1, %as_type.loc8 [concrete = constants.%S]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc8_3: ref %S = splice_block %s.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc8: init %S to %.loc8_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn.1()
// CHECK:STDOUT:   assign %s.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc8
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref.loc8 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = ref_binding s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesRValue.ref.loc17: %TakesRValue.cpp_overload_set.type = name_ref TakesRValue, imports.%TakesRValue.cpp_overload_set.value [concrete = constants.%TakesRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type.471 = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.var_patt: %pattern_type.471 = var_pattern %t.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %T.3b5 = var %t.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc19: %DefaultOrUnformed.type = facet_value constants.%T.3b5, (constants.%DefaultOrUnformed.impl_witness.dca) [concrete = constants.%DefaultOrUnformed.facet.fe5]
// CHECK:STDOUT:   %.loc19_15.1: %DefaultOrUnformed.type = converted constants.%T.3b5, %DefaultOrUnformed.facet.loc19 [concrete = constants.%DefaultOrUnformed.facet.fe5]
// CHECK:STDOUT:   %as_type.loc19: type = facet_access_type %.loc19_15.1 [concrete = constants.%T.3b5]
// CHECK:STDOUT:   %.loc19_15.2: type = converted %.loc19_15.1, %as_type.loc19 [concrete = constants.%T.3b5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc19_3: ref %T.3b5 = splice_block %t.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc19: init %T.3b5 to %.loc19_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn.2()
// CHECK:STDOUT:   assign %t.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc19
// CHECK:STDOUT:   %.loc19_13: type = splice_block %T.ref [concrete = constants.%T.3b5] {
// CHECK:STDOUT:     %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %T.ref: type = name_ref T, imports.%T.decl [concrete = constants.%T.3b5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %T.3b5 = ref_binding t, %t.var
// CHECK:STDOUT:   %Cpp.ref.loc28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesRValue.ref.loc28: %TakesRValue.cpp_overload_set.type = name_ref TakesRValue, imports.%TakesRValue.cpp_overload_set.value [concrete = constants.%TakesRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %t.ref: ref %T.3b5 = name_ref t, %t
// CHECK:STDOUT:   %Cpp.ref.loc38_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesRValue.ref.loc38: %TakesRValue.cpp_overload_set.type = name_ref TakesRValue, imports.%TakesRValue.cpp_overload_set.value [concrete = constants.%TakesRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc38_21.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct.a40]
// CHECK:STDOUT:   %Cpp.ref.loc38_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc38_29: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %.loc38_21.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc38_21.3: init %S to %.loc38_21.2 = class_init () [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc38_23: init %S = converted %.loc38_21.1, %.loc38_21.3 [concrete = constants.%S.val]
// CHECK:STDOUT:   %Cpp.ref.loc38_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc38_45: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %const: type = const_type %S.ref.loc38_45 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc38_33.1: init %const = as_compatible %.loc38_23 [concrete = constants.%empty_struct.418]
// CHECK:STDOUT:   %.loc38_33.2: init %const = converted %.loc38_23, %.loc38_33.1 [concrete = constants.%empty_struct.418]
// CHECK:STDOUT:   %T.cpp_destructor.bound: <bound method> = bound_method %t.var, constants.%T.cpp_destructor
// CHECK:STDOUT:   %T.cpp_destructor.call: init %empty_tuple.type = call %T.cpp_destructor.bound(%t.var)
// CHECK:STDOUT:   %S.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_param_const_lvalue_ref_with_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.b2d: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %TakesConstLValue.cpp_overload_set.type: type = cpp_overload_set_type @TakesConstLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesConstLValue.cpp_overload_set.value: %TakesConstLValue.cpp_overload_set.type = cpp_overload_set_value @TakesConstLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %ptr.26e: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.type: type = fn_type @TakesConstLValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk: %TakesConstLValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.055: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %S.cpp_destructor.type: type = fn_type @S.cpp_destructor [concrete]
// CHECK:STDOUT:   %S.cpp_destructor: %S.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .TakesConstLValue = %TakesConstLValue.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %TakesConstLValue.cpp_overload_set.value: %TakesConstLValue.cpp_overload_set.type = cpp_overload_set_value @TakesConstLValue.cpp_overload_set [concrete = constants.%TakesConstLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.decl: %TakesConstLValue__carbon_thunk.type = fn_decl @TakesConstLValue__carbon_thunk [concrete = constants.%TakesConstLValue__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.b2d = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.b2d = var_pattern %s.patt [concrete]
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
// CHECK:STDOUT:   %TakesConstLValue.ref.loc9: %TakesConstLValue.cpp_overload_set.type = name_ref TakesConstLValue, imports.%TakesConstLValue.cpp_overload_set.value [concrete = constants.%TakesConstLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref.loc9: ref %S = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc9_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc9: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %const: type = const_type %S.ref.loc9 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc9_26.1: ref %const = as_compatible %s.ref.loc9
// CHECK:STDOUT:   %.loc9_26.2: ref %const = converted %s.ref.loc9, %.loc9_26.1
// CHECK:STDOUT:   %.loc9_26.3: ref %S = as_compatible %.loc9_26.2
// CHECK:STDOUT:   %.loc9_26.4: ref %S = converted %.loc9_26.2, %.loc9_26.3
// CHECK:STDOUT:   %.loc9_26.5: %S = acquire_value %.loc9_26.4
// CHECK:STDOUT:   %.loc9_26.6: ref %S = value_as_ref %.loc9_26.5
// CHECK:STDOUT:   %addr.loc9: %ptr.055 = addr_of %.loc9_26.6
// CHECK:STDOUT:   %.loc9_40.1: %ptr.26e = as_compatible %addr.loc9
// CHECK:STDOUT:   %.loc9_40.2: %ptr.26e = converted %addr.loc9, %.loc9_40.1
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%TakesConstLValue__carbon_thunk.decl(%.loc9_40.2)
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstLValue.ref.loc11: %TakesConstLValue.cpp_overload_set.type = name_ref TakesConstLValue, imports.%TakesConstLValue.cpp_overload_set.value [concrete = constants.%TakesConstLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref.loc11: ref %S = name_ref s, %s
// CHECK:STDOUT:   %.loc11_24.1: %S = acquire_value %s.ref.loc11
// CHECK:STDOUT:   %.loc11_24.2: ref %S = value_as_ref %.loc11_24.1
// CHECK:STDOUT:   %addr.loc11: %ptr.055 = addr_of %.loc11_24.2
// CHECK:STDOUT:   %.loc11_25.1: %ptr.26e = as_compatible %addr.loc11
// CHECK:STDOUT:   %.loc11_25.2: %ptr.26e = converted %addr.loc11, %.loc11_25.1
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.call.loc11: init %empty_tuple.type = call imports.%TakesConstLValue__carbon_thunk.decl(%.loc11_25.2)
// CHECK:STDOUT:   %S.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_param_const_lvalue_ref_with_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.b2d: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %.f11: ref %S = temporary invalid, %S.val [concrete]
// CHECK:STDOUT:   %TakesConstLValue.cpp_overload_set.type: type = cpp_overload_set_type @TakesConstLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesConstLValue.cpp_overload_set.value: %TakesConstLValue.cpp_overload_set.type = cpp_overload_set_value @TakesConstLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %ptr.26e: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.type: type = fn_type @TakesConstLValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk: %TakesConstLValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.055: type = ptr_type %S [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .TakesConstLValue = %TakesConstLValue.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %TakesConstLValue.cpp_overload_set.value: %TakesConstLValue.cpp_overload_set.type = cpp_overload_set_value @TakesConstLValue.cpp_overload_set [concrete = constants.%TakesConstLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.decl: %TakesConstLValue__carbon_thunk.type = fn_decl @TakesConstLValue__carbon_thunk [concrete = constants.%TakesConstLValue__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.b2d = value_binding_pattern s [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_19.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref.loc8 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_19.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc8_19.3: init %S to %.loc8_19.2 = class_init () [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_19.4: init %S = converted %.loc8_19.1, %.loc8_19.3 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_19.5: ref %S = temporary %.loc8_19.2, %.loc8_19.4 [concrete = constants.%.f11]
// CHECK:STDOUT:   %.loc8_19.6: %S = acquire_value %.loc8_19.5 [concrete = constants.%S.val]
// CHECK:STDOUT:   %s: %S = value_binding s, %.loc8_19.6
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstLValue.ref.loc9: %TakesConstLValue.cpp_overload_set.type = name_ref TakesConstLValue, imports.%TakesConstLValue.cpp_overload_set.value [concrete = constants.%TakesConstLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref.loc9: %S = name_ref s, %s
// CHECK:STDOUT:   %.loc9_24: ref %S = value_as_ref %s.ref.loc9
// CHECK:STDOUT:   %addr.loc9: %ptr.055 = addr_of %.loc9_24
// CHECK:STDOUT:   %.loc9_25.1: %ptr.26e = as_compatible %addr.loc9
// CHECK:STDOUT:   %.loc9_25.2: %ptr.26e = converted %addr.loc9, %.loc9_25.1
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%TakesConstLValue__carbon_thunk.decl(%.loc9_25.2)
// CHECK:STDOUT:   %Cpp.ref.loc11_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstLValue.ref.loc11: %TakesConstLValue.cpp_overload_set.type = name_ref TakesConstLValue, imports.%TakesConstLValue.cpp_overload_set.value [concrete = constants.%TakesConstLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref.loc11: %S = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc11_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc11: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %const: type = const_type %S.ref.loc11 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc11_26.1: %const = as_compatible %s.ref.loc11
// CHECK:STDOUT:   %.loc11_26.2: %const = converted %s.ref.loc11, %.loc11_26.1
// CHECK:STDOUT:   %.loc11_26.3: %S = as_compatible %.loc11_26.2
// CHECK:STDOUT:   %.loc11_26.4: %S = converted %.loc11_26.2, %.loc11_26.3
// CHECK:STDOUT:   %.loc11_26.5: ref %S = value_as_ref %.loc11_26.4
// CHECK:STDOUT:   %addr.loc11: %ptr.055 = addr_of %.loc11_26.5
// CHECK:STDOUT:   %.loc11_40.1: %ptr.26e = as_compatible %addr.loc11
// CHECK:STDOUT:   %.loc11_40.2: %ptr.26e = converted %addr.loc11, %.loc11_40.1
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.call.loc11: init %empty_tuple.type = call imports.%TakesConstLValue__carbon_thunk.decl(%.loc11_40.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- param_value_arg_for_const_rvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TakesConstRValue.cpp_overload_set.type: type = cpp_overload_set_type @TakesConstRValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesConstRValue.cpp_overload_set.value: %TakesConstRValue.cpp_overload_set.type = cpp_overload_set_value @TakesConstRValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.26e: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %TakesConstRValue__carbon_thunk.type: type = fn_type @TakesConstRValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesConstRValue__carbon_thunk: %TakesConstRValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.055: type = ptr_type %S [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .TakesConstRValue = %TakesConstRValue.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %TakesConstRValue.cpp_overload_set.value: %TakesConstRValue.cpp_overload_set.type = cpp_overload_set_value @TakesConstRValue.cpp_overload_set [concrete = constants.%TakesConstRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %TakesConstRValue__carbon_thunk.decl: %TakesConstRValue__carbon_thunk.type = fn_decl @TakesConstRValue__carbon_thunk [concrete = constants.%TakesConstRValue__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%v.param: %S, %cv.param: %const) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstRValue.ref.loc8: %TakesConstRValue.cpp_overload_set.type = name_ref TakesConstRValue, imports.%TakesConstRValue.cpp_overload_set.value [concrete = constants.%TakesConstRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %v.ref: %S = name_ref v, %v
// CHECK:STDOUT:   %.loc8_24: ref %S = value_as_ref %v.ref
// CHECK:STDOUT:   %addr.loc8: %ptr.055 = addr_of %.loc8_24
// CHECK:STDOUT:   %.loc8_25.1: %ptr.26e = as_compatible %addr.loc8
// CHECK:STDOUT:   %.loc8_25.2: %ptr.26e = converted %addr.loc8, %.loc8_25.1
// CHECK:STDOUT:   %TakesConstRValue__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%TakesConstRValue__carbon_thunk.decl(%.loc8_25.2)
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstRValue.ref.loc9: %TakesConstRValue.cpp_overload_set.type = name_ref TakesConstRValue, imports.%TakesConstRValue.cpp_overload_set.value [concrete = constants.%TakesConstRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %cv.ref: %const = name_ref cv, %cv
// CHECK:STDOUT:   %.loc9_24.1: %S = as_compatible %cv.ref
// CHECK:STDOUT:   %.loc9_24.2: %S = converted %cv.ref, %.loc9_24.1
// CHECK:STDOUT:   %.loc9_24.3: ref %S = value_as_ref %.loc9_24.2
// CHECK:STDOUT:   %addr.loc9: %ptr.055 = addr_of %.loc9_24.3
// CHECK:STDOUT:   %.loc9_26.1: %ptr.26e = as_compatible %addr.loc9
// CHECK:STDOUT:   %.loc9_26.2: %ptr.26e = converted %addr.loc9, %.loc9_26.1
// CHECK:STDOUT:   %TakesConstRValue__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%TakesConstRValue__carbon_thunk.decl(%.loc9_26.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_param_lvalue_ref_arg_const_rvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %pattern_type.b2d: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.e5a: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.38d: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.e5a) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.eab: %T.as_type.as.DefaultOrUnformed.impl.Op.type.38d = struct_value () [symbolic]
// CHECK:STDOUT:   %S.Op.type: type = fn_type @S.Op [concrete]
// CHECK:STDOUT:   %S.Op: %S.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b26fff.1: <witness> = custom_witness (%S.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.229: %Default.type = facet_value %S, (%custom_witness.b26fff.1) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.f99: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.48a, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.229) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.9ab: %DefaultOrUnformed.type = facet_value %S, (%DefaultOrUnformed.impl_witness.f99) [concrete]
// CHECK:STDOUT:   %TakesConstRValue.cpp_overload_set.type: type = cpp_overload_set_type @TakesConstRValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesConstRValue.cpp_overload_set.value: %TakesConstRValue.cpp_overload_set.type = cpp_overload_set_value @TakesConstRValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %T.3b5: type = class_type @T [concrete]
// CHECK:STDOUT:   %pattern_type.471: type = pattern_type %T.3b5 [concrete]
// CHECK:STDOUT:   %T.Op.type: type = fn_type @T.Op [concrete]
// CHECK:STDOUT:   %T.Op: %T.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b26fff.2: <witness> = custom_witness (%T.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.dcd: %Default.type = facet_value %T.3b5, (%custom_witness.b26fff.2) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.dca: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.48a, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.dcd) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.fe5: %DefaultOrUnformed.type = facet_value %T.3b5, (%DefaultOrUnformed.impl_witness.dca) [concrete]
// CHECK:STDOUT:   %T.cpp_destructor.type: type = fn_type @T.cpp_destructor [concrete]
// CHECK:STDOUT:   %T.cpp_destructor: %T.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %S.cpp_destructor.type: type = fn_type @S.cpp_destructor [concrete]
// CHECK:STDOUT:   %S.cpp_destructor: %S.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .TakesConstRValue = %TakesConstRValue.cpp_overload_set.value
// CHECK:STDOUT:     .T = %T.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.a03: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.38d) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.eab)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.48a = impl_witness_table (%Core.import_ref.a03), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %TakesConstRValue.cpp_overload_set.value: %TakesConstRValue.cpp_overload_set.type = cpp_overload_set_value @TakesConstRValue.cpp_overload_set [concrete = constants.%TakesConstRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %T.decl: type = class_decl @T [concrete = constants.%T.3b5] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.b2d = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.b2d = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc8: %DefaultOrUnformed.type = facet_value constants.%S, (constants.%DefaultOrUnformed.impl_witness.f99) [concrete = constants.%DefaultOrUnformed.facet.9ab]
// CHECK:STDOUT:   %.loc8_15.1: %DefaultOrUnformed.type = converted constants.%S, %DefaultOrUnformed.facet.loc8 [concrete = constants.%DefaultOrUnformed.facet.9ab]
// CHECK:STDOUT:   %as_type.loc8: type = facet_access_type %.loc8_15.1 [concrete = constants.%S]
// CHECK:STDOUT:   %.loc8_15.2: type = converted %.loc8_15.1, %as_type.loc8 [concrete = constants.%S]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc8_3: ref %S = splice_block %s.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc8: init %S to %.loc8_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn.1()
// CHECK:STDOUT:   assign %s.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc8
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = ref_binding s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstRValue.ref.loc17: %TakesConstRValue.cpp_overload_set.type = name_ref TakesConstRValue, imports.%TakesConstRValue.cpp_overload_set.value [concrete = constants.%TakesConstRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type.471 = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.var_patt: %pattern_type.471 = var_pattern %t.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %T.3b5 = var %t.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc19: %DefaultOrUnformed.type = facet_value constants.%T.3b5, (constants.%DefaultOrUnformed.impl_witness.dca) [concrete = constants.%DefaultOrUnformed.facet.fe5]
// CHECK:STDOUT:   %.loc19_15.1: %DefaultOrUnformed.type = converted constants.%T.3b5, %DefaultOrUnformed.facet.loc19 [concrete = constants.%DefaultOrUnformed.facet.fe5]
// CHECK:STDOUT:   %as_type.loc19: type = facet_access_type %.loc19_15.1 [concrete = constants.%T.3b5]
// CHECK:STDOUT:   %.loc19_15.2: type = converted %.loc19_15.1, %as_type.loc19 [concrete = constants.%T.3b5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc19_3: ref %T.3b5 = splice_block %t.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc19: init %T.3b5 to %.loc19_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn.2()
// CHECK:STDOUT:   assign %t.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call.loc19
// CHECK:STDOUT:   %.loc19_13: type = splice_block %T.ref [concrete = constants.%T.3b5] {
// CHECK:STDOUT:     %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %T.ref: type = name_ref T, imports.%T.decl [concrete = constants.%T.3b5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %T.3b5 = ref_binding t, %t.var
// CHECK:STDOUT:   %Cpp.ref.loc28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstRValue.ref.loc28: %TakesConstRValue.cpp_overload_set.type = name_ref TakesConstRValue, imports.%TakesConstRValue.cpp_overload_set.value [concrete = constants.%TakesConstRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %t.ref: ref %T.3b5 = name_ref t, %t
// CHECK:STDOUT:   %T.cpp_destructor.bound: <bound method> = bound_method %t.var, constants.%T.cpp_destructor
// CHECK:STDOUT:   %T.cpp_destructor.call: init %empty_tuple.type = call %T.cpp_destructor.bound(%t.var)
// CHECK:STDOUT:   %S.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_return_lvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %ReturnsLValue.cpp_overload_set.type: type = cpp_overload_set_type @ReturnsLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ReturnsLValue.cpp_overload_set.value: %ReturnsLValue.cpp_overload_set.type = cpp_overload_set_value @ReturnsLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ReturnsLValue.type: type = fn_type @ReturnsLValue [concrete]
// CHECK:STDOUT:   %ReturnsLValue: %ReturnsLValue.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .ReturnsLValue = %ReturnsLValue.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %ReturnsLValue.cpp_overload_set.value: %ReturnsLValue.cpp_overload_set.type = cpp_overload_set_value @ReturnsLValue.cpp_overload_set [concrete = constants.%ReturnsLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %ReturnsLValue.decl: %ReturnsLValue.type = fn_decl @ReturnsLValue [concrete = constants.%ReturnsLValue] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ReturnsLValue.ref: %ReturnsLValue.cpp_overload_set.type = name_ref ReturnsLValue, imports.%ReturnsLValue.cpp_overload_set.value [concrete = constants.%ReturnsLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %ReturnsLValue.call: ref %S = call imports.%ReturnsLValue.decl()
// CHECK:STDOUT:   %.loc8: type = splice_block %S.ref [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = ref_binding s, %ReturnsLValue.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_return_rvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %ReturnsRValue.cpp_overload_set.type: type = cpp_overload_set_type @ReturnsRValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ReturnsRValue.cpp_overload_set.value: %ReturnsRValue.cpp_overload_set.type = cpp_overload_set_value @ReturnsRValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ReturnsRValue.type: type = fn_type @ReturnsRValue [concrete]
// CHECK:STDOUT:   %ReturnsRValue: %ReturnsRValue.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .ReturnsRValue = %ReturnsRValue.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %ReturnsRValue.cpp_overload_set.value: %ReturnsRValue.cpp_overload_set.type = cpp_overload_set_value @ReturnsRValue.cpp_overload_set [concrete = constants.%ReturnsRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %ReturnsRValue.decl: %ReturnsRValue.type = fn_decl @ReturnsRValue [concrete = constants.%ReturnsRValue] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ReturnsRValue.ref: %ReturnsRValue.cpp_overload_set.type = name_ref ReturnsRValue, imports.%ReturnsRValue.cpp_overload_set.value [concrete = constants.%ReturnsRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %ReturnsRValue.call: ref %S = call imports.%ReturnsRValue.decl()
// CHECK:STDOUT:   %.loc8: type = splice_block %S.ref [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = ref_binding s, %ReturnsRValue.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_return_const_lvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %ReturnConstLValue.cpp_overload_set.type: type = cpp_overload_set_type @ReturnConstLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ReturnConstLValue.cpp_overload_set.value: %ReturnConstLValue.cpp_overload_set.type = cpp_overload_set_value @ReturnConstLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ReturnConstLValue.type: type = fn_type @ReturnConstLValue [concrete]
// CHECK:STDOUT:   %ReturnConstLValue: %ReturnConstLValue.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .ReturnConstLValue = %ReturnConstLValue.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %ReturnConstLValue.cpp_overload_set.value: %ReturnConstLValue.cpp_overload_set.type = cpp_overload_set_value @ReturnConstLValue.cpp_overload_set [concrete = constants.%ReturnConstLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %ReturnConstLValue.decl: %ReturnConstLValue.type = fn_decl @ReturnConstLValue [concrete = constants.%ReturnConstLValue] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ReturnConstLValue.ref: %ReturnConstLValue.cpp_overload_set.type = name_ref ReturnConstLValue, imports.%ReturnConstLValue.cpp_overload_set.value [concrete = constants.%ReturnConstLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %ReturnConstLValue.call: ref %const = call imports.%ReturnConstLValue.decl()
// CHECK:STDOUT:   %.loc8: type = splice_block %const [concrete = constants.%const] {
// CHECK:STDOUT:     %Cpp.ref.loc8_27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %const: type = const_type %S.ref [concrete = constants.%const]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %const = ref_binding s, %ReturnConstLValue.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- forwarding_reference.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %TakeAnything_ExpectConstRRef.cpp_overload_set.type: type = cpp_overload_set_type @TakeAnything_ExpectConstRRef.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakeAnything_ExpectConstRRef.cpp_overload_set.value: %TakeAnything_ExpectConstRRef.cpp_overload_set.type = cpp_overload_set_value @TakeAnything_ExpectConstRRef.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const: type = const_type %X [concrete]
// CHECK:STDOUT:   %ptr.2c7: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %TakeAnything_ExpectConstRRef__carbon_thunk.type: type = fn_type @TakeAnything_ExpectConstRRef__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakeAnything_ExpectConstRRef__carbon_thunk: %TakeAnything_ExpectConstRRef__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.324: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %TakeAnything_ExpectLRef.cpp_overload_set.type: type = cpp_overload_set_type @TakeAnything_ExpectLRef.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakeAnything_ExpectLRef.cpp_overload_set.value: %TakeAnything_ExpectLRef.cpp_overload_set.type = cpp_overload_set_value @TakeAnything_ExpectLRef.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakeAnything_ExpectLRef.type: type = fn_type @TakeAnything_ExpectLRef [concrete]
// CHECK:STDOUT:   %TakeAnything_ExpectLRef: %TakeAnything_ExpectLRef.type = struct_value () [concrete]
// CHECK:STDOUT:   %TakeAnything_ExpectConstLRef.cpp_overload_set.type: type = cpp_overload_set_type @TakeAnything_ExpectConstLRef.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakeAnything_ExpectConstLRef.cpp_overload_set.value: %TakeAnything_ExpectConstLRef.cpp_overload_set.type = cpp_overload_set_value @TakeAnything_ExpectConstLRef.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakeAnything_ExpectConstLRef__carbon_thunk.type: type = fn_type @TakeAnything_ExpectConstLRef__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakeAnything_ExpectConstLRef__carbon_thunk: %TakeAnything_ExpectConstLRef__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %TakeAnything_ExpectRRef.cpp_overload_set.type: type = cpp_overload_set_type @TakeAnything_ExpectRRef.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakeAnything_ExpectRRef.cpp_overload_set.value: %TakeAnything_ExpectRRef.cpp_overload_set.type = cpp_overload_set_value @TakeAnything_ExpectRRef.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakeAnything_ExpectRRef__carbon_thunk.type: type = fn_type @TakeAnything_ExpectRRef__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakeAnything_ExpectRRef__carbon_thunk: %TakeAnything_ExpectRRef__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TakeAnything_ExpectConstRRef = %TakeAnything_ExpectConstRRef.cpp_overload_set.value
// CHECK:STDOUT:     .TakeAnything_ExpectLRef = %TakeAnything_ExpectLRef.cpp_overload_set.value
// CHECK:STDOUT:     .TakeAnything_ExpectConstLRef = %TakeAnything_ExpectConstLRef.cpp_overload_set.value
// CHECK:STDOUT:     .TakeAnything_ExpectRRef = %TakeAnything_ExpectRRef.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakeAnything_ExpectConstRRef.cpp_overload_set.value: %TakeAnything_ExpectConstRRef.cpp_overload_set.type = cpp_overload_set_value @TakeAnything_ExpectConstRRef.cpp_overload_set [concrete = constants.%TakeAnything_ExpectConstRRef.cpp_overload_set.value]
// CHECK:STDOUT:   %TakeAnything_ExpectConstRRef__carbon_thunk.decl: %TakeAnything_ExpectConstRRef__carbon_thunk.type = fn_decl @TakeAnything_ExpectConstRRef__carbon_thunk [concrete = constants.%TakeAnything_ExpectConstRRef__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakeAnything_ExpectLRef.cpp_overload_set.value: %TakeAnything_ExpectLRef.cpp_overload_set.type = cpp_overload_set_value @TakeAnything_ExpectLRef.cpp_overload_set [concrete = constants.%TakeAnything_ExpectLRef.cpp_overload_set.value]
// CHECK:STDOUT:   %TakeAnything_ExpectLRef.decl: %TakeAnything_ExpectLRef.type = fn_decl @TakeAnything_ExpectLRef [concrete = constants.%TakeAnything_ExpectLRef] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakeAnything_ExpectConstLRef.cpp_overload_set.value: %TakeAnything_ExpectConstLRef.cpp_overload_set.type = cpp_overload_set_value @TakeAnything_ExpectConstLRef.cpp_overload_set [concrete = constants.%TakeAnything_ExpectConstLRef.cpp_overload_set.value]
// CHECK:STDOUT:   %TakeAnything_ExpectConstLRef__carbon_thunk.decl: %TakeAnything_ExpectConstLRef__carbon_thunk.type = fn_decl @TakeAnything_ExpectConstLRef__carbon_thunk [concrete = constants.%TakeAnything_ExpectConstLRef__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakeAnything_ExpectRRef.cpp_overload_set.value: %TakeAnything_ExpectRRef.cpp_overload_set.type = cpp_overload_set_value @TakeAnything_ExpectRRef.cpp_overload_set [concrete = constants.%TakeAnything_ExpectRRef.cpp_overload_set.value]
// CHECK:STDOUT:   %TakeAnything_ExpectRRef__carbon_thunk.decl: %TakeAnything_ExpectRRef__carbon_thunk.type = fn_decl @TakeAnything_ExpectRRef__carbon_thunk [concrete = constants.%TakeAnything_ExpectRRef__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassValue(%x.param: %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakeAnything_ExpectConstRRef.ref: %TakeAnything_ExpectConstRRef.cpp_overload_set.type = name_ref TakeAnything_ExpectConstRRef, imports.%TakeAnything_ExpectConstRRef.cpp_overload_set.value [concrete = constants.%TakeAnything_ExpectConstRRef.cpp_overload_set.value]
// CHECK:STDOUT:   %x.ref: %X = name_ref x, %x
// CHECK:STDOUT:   %.loc28_36: ref %X = value_as_ref %x.ref
// CHECK:STDOUT:   %addr: %ptr.324 = addr_of %.loc28_36
// CHECK:STDOUT:   %.loc28_37.1: %ptr.2c7 = as_compatible %addr
// CHECK:STDOUT:   %.loc28_37.2: %ptr.2c7 = converted %addr, %.loc28_37.1
// CHECK:STDOUT:   %TakeAnything_ExpectConstRRef__carbon_thunk.call: init %empty_tuple.type = call imports.%TakeAnything_ExpectConstRRef__carbon_thunk.decl(%.loc28_37.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassRef(%x.param: ref %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakeAnything_ExpectLRef.ref: %TakeAnything_ExpectLRef.cpp_overload_set.type = name_ref TakeAnything_ExpectLRef, imports.%TakeAnything_ExpectLRef.cpp_overload_set.value [concrete = constants.%TakeAnything_ExpectLRef.cpp_overload_set.value]
// CHECK:STDOUT:   %x.ref: ref %X = name_ref x, %x
// CHECK:STDOUT:   %.loc34: %X = ref_tag %x.ref
// CHECK:STDOUT:   %TakeAnything_ExpectLRef.call: init %empty_tuple.type = call imports.%TakeAnything_ExpectLRef.decl(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassConstRef(%x.param: ref %const) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakeAnything_ExpectConstLRef.ref: %TakeAnything_ExpectConstLRef.cpp_overload_set.type = name_ref TakeAnything_ExpectConstLRef, imports.%TakeAnything_ExpectConstLRef.cpp_overload_set.value [concrete = constants.%TakeAnything_ExpectConstLRef.cpp_overload_set.value]
// CHECK:STDOUT:   %x.ref: ref %const = name_ref x, %x
// CHECK:STDOUT:   %.loc40_36.1: ref %X = as_compatible %x.ref
// CHECK:STDOUT:   %.loc40_36.2: ref %X = converted %x.ref, %.loc40_36.1
// CHECK:STDOUT:   %.loc40_36.3: %X = acquire_value %.loc40_36.2
// CHECK:STDOUT:   %.loc40_36.4: ref %X = value_as_ref %.loc40_36.3
// CHECK:STDOUT:   %addr: %ptr.324 = addr_of %.loc40_36.4
// CHECK:STDOUT:   %.loc40_37.1: %ptr.2c7 = as_compatible %addr
// CHECK:STDOUT:   %.loc40_37.2: %ptr.2c7 = converted %addr, %.loc40_37.1
// CHECK:STDOUT:   %TakeAnything_ExpectConstLRef__carbon_thunk.call: init %empty_tuple.type = call imports.%TakeAnything_ExpectConstLRef__carbon_thunk.decl(%.loc40_37.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassInit() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakeAnything_ExpectRRef.ref: %TakeAnything_ExpectRRef.cpp_overload_set.type = name_ref TakeAnything_ExpectRRef, imports.%TakeAnything_ExpectRRef.cpp_overload_set.value [concrete = constants.%TakeAnything_ExpectRRef.cpp_overload_set.value]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Make.call: init %X to %_.var = call %Make.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr: %ptr.324 = addr_of %_.var
// CHECK:STDOUT:   %TakeAnything_ExpectRRef__carbon_thunk.call: init %empty_tuple.type = call imports.%TakeAnything_ExpectRRef__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
