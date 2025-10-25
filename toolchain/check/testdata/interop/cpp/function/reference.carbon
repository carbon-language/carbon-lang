// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/reference.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/reference.carbon

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
  Cpp.TakesLValue(s);
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
  // CHECK:STDERR: ./param_lvalue_ref.h:5:6: note: candidate function not viable: expects an lvalue for 1st argument [CppInteropParseNote]
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

// --- todo_fail_param_value_arg_for_rvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "param_rvalue_ref.h";

fn F() {
  //@dump-sem-ir-begin
  // TODO: We should probably reject binding an rvalue reference to a value
  // expression. If we don't reject, we should instead force a copy to be made,
  // at least if the type has a pointer value representation, so that moving
  // from the reference doesn't alter tne original value.
  let s: Cpp.S = {};
  Cpp.TakesRValue(s);
  //@dump-sem-ir-end
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
  let s: Cpp.S* = Cpp.ReturnsLValue();
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
  var s: Cpp.S* = Cpp.ReturnsRValue();
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
  var s: const Cpp.S* = Cpp.ReturnConstLValue();
  //@dump-sem-ir-end
}

// --- fail_call_return_const_lvalue_ref_const_correctness.carbon

library "[[@TEST_NAME]]";

import Cpp library "return_const_lvalue_ref.h";

fn F() {
  // CHECK:STDERR: fail_call_return_const_lvalue_ref_const_correctness.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `const (const Cpp.S*)` to `Cpp.S*` [ConversionFailure]
  // CHECK:STDERR:   var s: Cpp.S* = Cpp.ReturnConstLValue();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_call_return_const_lvalue_ref_const_correctness.carbon:[[@LINE+4]]:3: note: type `const (const Cpp.S*)` does not implement interface `Core.ImplicitAs(Cpp.S*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var s: Cpp.S* = Cpp.ReturnConstLValue();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  var s: Cpp.S* = Cpp.ReturnConstLValue();
}

// CHECK:STDOUT: --- call_param_lvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %TakesLValue.cpp_overload_set.type: type = cpp_overload_set_type @TakesLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesLValue.cpp_overload_set.value: %TakesLValue.cpp_overload_set.type = cpp_overload_set_value @TakesLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %TakesLValue.type: type = fn_type @TakesLValue [concrete]
// CHECK:STDOUT:   %TakesLValue: %TakesLValue.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.34a: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.016: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.34a = struct_value () [concrete]
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
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc8_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_19.2: init %S = class_init (), %s.var [concrete = constants.%S.val]
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
// CHECK:STDOUT:   %addr.loc9: %ptr.5c7 = addr_of %s.ref
// CHECK:STDOUT:   %TakesLValue.call: init %empty_tuple.type = call imports.%TakesLValue.decl(%addr.loc9)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %s.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.016
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %s.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8: %ptr.5c7 = addr_of %s.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_param_lvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %TakesLValue.cpp_overload_set.type: type = cpp_overload_set_type @TakesLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesLValue.cpp_overload_set.value: %TakesLValue.cpp_overload_set.type = cpp_overload_set_value @TakesLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %T: type = class_type @T [concrete]
// CHECK:STDOUT:   %pattern_type.e6b: type = pattern_type %T [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.7bd: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.34a: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.7bd) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.016: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.34a = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %facet_value.19d: %type_where = facet_value %T, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.431: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.19d) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.2f0: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.431 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b04: type = ptr_type %T [concrete]
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
// CHECK:STDOUT:   %TakesLValue.cpp_overload_set.value: %TakesLValue.cpp_overload_set.type = cpp_overload_set_value @TakesLValue.cpp_overload_set [concrete = constants.%TakesLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %T.decl: type = class_decl @T [concrete = constants.%T] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.7da = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.7da = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %S = var %v.var_patt
// CHECK:STDOUT:   %.loc8: type = splice_block %S.ref.loc8 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %S = ref_binding v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = value_binding_pattern s [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.ref: ref %S = name_ref v, %v
// CHECK:STDOUT:   %.loc9_13: type = splice_block %S.ref.loc9 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc9: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_18: %S = bind_value %v.ref
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
// CHECK:STDOUT:     %t.patt: %pattern_type.e6b = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.var_patt: %pattern_type.e6b = var_pattern %t.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %T = var %t.var_patt
// CHECK:STDOUT:   %.loc30: type = splice_block %T.ref [concrete = constants.%T] {
// CHECK:STDOUT:     %Cpp.ref.loc30: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %T.ref: type = name_ref T, imports.%T.decl [concrete = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %T = ref_binding t, %t.var
// CHECK:STDOUT:   %Cpp.ref.loc39: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesLValue.ref.loc39: %TakesLValue.cpp_overload_set.type = name_ref TakesLValue, imports.%TakesLValue.cpp_overload_set.value [concrete = constants.%TakesLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %t.ref: ref %T = name_ref t, %t
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %u.patt: %pattern_type.7da = ref_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.var_patt: %pattern_type.7da = var_pattern %u.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %u.var: ref %S = var %u.var_patt
// CHECK:STDOUT:   %.loc41: type = splice_block %S.ref.loc41 [concrete = constants.%S] {
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
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc41: <bound method> = bound_method %u.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.016
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc41: <bound method> = bound_method %u.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc41: %ptr.5c7 = addr_of %u.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc41: init %empty_tuple.type = call %bound_method.loc41(%addr.loc41)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc30: <bound method> = bound_method %t.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2f0
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc30: <bound method> = bound_method %t.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc30: %ptr.b04 = addr_of %t.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc30: init %empty_tuple.type = call %bound_method.loc30(%addr.loc30)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %v.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.016
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %v.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc8: %ptr.5c7 = addr_of %v.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8)
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
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %TakesRValue__carbon_thunk.type: type = fn_type @TakesRValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesRValue__carbon_thunk: %TakesRValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.34a: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.016: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.34a = struct_value () [concrete]
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
// CHECK:STDOUT:   %.loc8_20.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc8_25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %.loc8_20.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc8_20.3: init %S = class_init (), %.loc8_20.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_20.4: ref %S = temporary %.loc8_20.2, %.loc8_20.3
// CHECK:STDOUT:   %.loc8_22.1: ref %S = converted %.loc8_20.1, %.loc8_20.4
// CHECK:STDOUT:   %.loc8_22.2: %S = bind_value %.loc8_22.1
// CHECK:STDOUT:   %.loc8_22.3: ref %S = value_as_ref %.loc8_22.2
// CHECK:STDOUT:   %addr.loc8_30: %ptr.5c7 = addr_of %.loc8_22.3
// CHECK:STDOUT:   %TakesRValue__carbon_thunk.call: init %empty_tuple.type = call imports.%TakesRValue__carbon_thunk.decl(%addr.loc8_30)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_20.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.016
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_20.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_20: %ptr.5c7 = addr_of %.loc8_20.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8_20)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_param_value_arg_for_rvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %TakesRValue.cpp_overload_set.type: type = cpp_overload_set_type @TakesRValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesRValue.cpp_overload_set.value: %TakesRValue.cpp_overload_set.type = cpp_overload_set_value @TakesRValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %TakesRValue__carbon_thunk.type: type = fn_type @TakesRValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesRValue__carbon_thunk: %TakesRValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.34a: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.016: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.34a = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .TakesRValue = %TakesRValue.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %TakesRValue.cpp_overload_set.value: %TakesRValue.cpp_overload_set.type = cpp_overload_set_value @TakesRValue.cpp_overload_set [concrete = constants.%TakesRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %TakesRValue__carbon_thunk.decl: %TakesRValue__carbon_thunk.type = fn_decl @TakesRValue__carbon_thunk [concrete = constants.%TakesRValue__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = value_binding_pattern s [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc12_13: type = splice_block %S.ref [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_19.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc12_19.3: init %S = class_init (), %.loc12_19.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc12_19.4: ref %S = temporary %.loc12_19.2, %.loc12_19.3
// CHECK:STDOUT:   %.loc12_19.5: ref %S = converted %.loc12_19.1, %.loc12_19.4
// CHECK:STDOUT:   %.loc12_19.6: %S = bind_value %.loc12_19.5
// CHECK:STDOUT:   %s: %S = value_binding s, %.loc12_19.6
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesRValue.ref: %TakesRValue.cpp_overload_set.type = name_ref TakesRValue, imports.%TakesRValue.cpp_overload_set.value [concrete = constants.%TakesRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: %S = name_ref s, %s
// CHECK:STDOUT:   %.loc13: ref %S = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc13: %ptr.5c7 = addr_of %.loc13
// CHECK:STDOUT:   %TakesRValue__carbon_thunk.call: init %empty_tuple.type = call imports.%TakesRValue__carbon_thunk.decl(%addr.loc13)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc12_19.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.016
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc12_19.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc12: %ptr.5c7 = addr_of %.loc12_19.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc12)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_param_rvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %TakesRValue.cpp_overload_set.type: type = cpp_overload_set_type @TakesRValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesRValue.cpp_overload_set.value: %TakesRValue.cpp_overload_set.type = cpp_overload_set_value @TakesRValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %T: type = class_type @T [concrete]
// CHECK:STDOUT:   %pattern_type.e6b: type = pattern_type %T [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.7bd: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.34a: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.7bd) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.016: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.34a = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %facet_value.19d: %type_where = facet_value %T, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.431: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.19d) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.2f0: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.431 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b04: type = ptr_type %T [concrete]
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
// CHECK:STDOUT:   %TakesRValue.cpp_overload_set.value: %TakesRValue.cpp_overload_set.type = cpp_overload_set_value @TakesRValue.cpp_overload_set [concrete = constants.%TakesRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %T.decl: type = class_decl @T [concrete = constants.%T] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc8: type = splice_block %S.ref.loc8 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = ref_binding s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesRValue.ref.loc17: %TakesRValue.cpp_overload_set.type = name_ref TakesRValue, imports.%TakesRValue.cpp_overload_set.value [concrete = constants.%TakesRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type.e6b = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.var_patt: %pattern_type.e6b = var_pattern %t.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %T = var %t.var_patt
// CHECK:STDOUT:   %.loc19: type = splice_block %T.ref [concrete = constants.%T] {
// CHECK:STDOUT:     %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %T.ref: type = name_ref T, imports.%T.decl [concrete = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %T = ref_binding t, %t.var
// CHECK:STDOUT:   %Cpp.ref.loc28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesRValue.ref.loc28: %TakesRValue.cpp_overload_set.type = name_ref TakesRValue, imports.%TakesRValue.cpp_overload_set.value [concrete = constants.%TakesRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %t.ref: ref %T = name_ref t, %t
// CHECK:STDOUT:   %Cpp.ref.loc38_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesRValue.ref.loc38: %TakesRValue.cpp_overload_set.type = name_ref TakesRValue, imports.%TakesRValue.cpp_overload_set.value [concrete = constants.%TakesRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc38_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc38_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc38_29: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %.loc38_21.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc38_21.3: init %S = class_init (), %.loc38_21.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc38_21.4: ref %S = temporary %.loc38_21.2, %.loc38_21.3
// CHECK:STDOUT:   %.loc38_23: ref %S = converted %.loc38_21.1, %.loc38_21.4
// CHECK:STDOUT:   %Cpp.ref.loc38_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc38_45: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %const: type = const_type %S.ref.loc38_45 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc38_33.1: ref %const = as_compatible %.loc38_23
// CHECK:STDOUT:   %.loc38_33.2: ref %const = converted %.loc38_23, %.loc38_33.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc38: <bound method> = bound_method %.loc38_21.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.016
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc38: <bound method> = bound_method %.loc38_21.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc38: %ptr.5c7 = addr_of %.loc38_21.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc38: init %empty_tuple.type = call %bound_method.loc38(%addr.loc38)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc19: <bound method> = bound_method %t.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2f0
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc19: <bound method> = bound_method %t.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc19: %ptr.b04 = addr_of %t.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc19: init %empty_tuple.type = call %bound_method.loc19(%addr.loc19)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %s.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.016
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %s.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc8: %ptr.5c7 = addr_of %s.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_param_const_lvalue_ref_with_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %TakesConstLValue.cpp_overload_set.type: type = cpp_overload_set_type @TakesConstLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesConstLValue.cpp_overload_set.value: %TakesConstLValue.cpp_overload_set.type = cpp_overload_set_value @TakesConstLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %ptr.ff5: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.type: type = fn_type @TakesConstLValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk: %TakesConstLValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.34a: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.016: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.34a = struct_value () [concrete]
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
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc8_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_19.2: init %S = class_init (), %s.var [concrete = constants.%S.val]
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
// CHECK:STDOUT:   %.loc9_26.5: %S = bind_value %.loc9_26.4
// CHECK:STDOUT:   %.loc9_26.6: ref %S = value_as_ref %.loc9_26.5
// CHECK:STDOUT:   %addr.loc9: %ptr.5c7 = addr_of %.loc9_26.6
// CHECK:STDOUT:   %.loc9_40.1: %ptr.ff5 = as_compatible %addr.loc9
// CHECK:STDOUT:   %.loc9_40.2: %ptr.ff5 = converted %addr.loc9, %.loc9_40.1
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%TakesConstLValue__carbon_thunk.decl(%.loc9_40.2)
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstLValue.ref.loc11: %TakesConstLValue.cpp_overload_set.type = name_ref TakesConstLValue, imports.%TakesConstLValue.cpp_overload_set.value [concrete = constants.%TakesConstLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref.loc11: ref %S = name_ref s, %s
// CHECK:STDOUT:   %.loc11_24.1: %S = bind_value %s.ref.loc11
// CHECK:STDOUT:   %.loc11_24.2: ref %S = value_as_ref %.loc11_24.1
// CHECK:STDOUT:   %addr.loc11: %ptr.5c7 = addr_of %.loc11_24.2
// CHECK:STDOUT:   %.loc11_25.1: %ptr.ff5 = as_compatible %addr.loc11
// CHECK:STDOUT:   %.loc11_25.2: %ptr.ff5 = converted %addr.loc11, %.loc11_25.1
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.call.loc11: init %empty_tuple.type = call imports.%TakesConstLValue__carbon_thunk.decl(%.loc11_25.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %s.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.016
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %s.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8: %ptr.5c7 = addr_of %s.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_param_const_lvalue_ref_with_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %TakesConstLValue.cpp_overload_set.type: type = cpp_overload_set_type @TakesConstLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesConstLValue.cpp_overload_set.value: %TakesConstLValue.cpp_overload_set.type = cpp_overload_set_value @TakesConstLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %ptr.ff5: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.type: type = fn_type @TakesConstLValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk: %TakesConstLValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.34a: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.016: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.34a = struct_value () [concrete]
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
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = value_binding_pattern s [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref.loc8 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_19.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc8_19.3: init %S = class_init (), %.loc8_19.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_19.4: ref %S = temporary %.loc8_19.2, %.loc8_19.3
// CHECK:STDOUT:   %.loc8_19.5: ref %S = converted %.loc8_19.1, %.loc8_19.4
// CHECK:STDOUT:   %.loc8_19.6: %S = bind_value %.loc8_19.5
// CHECK:STDOUT:   %s: %S = value_binding s, %.loc8_19.6
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstLValue.ref.loc9: %TakesConstLValue.cpp_overload_set.type = name_ref TakesConstLValue, imports.%TakesConstLValue.cpp_overload_set.value [concrete = constants.%TakesConstLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref.loc9: %S = name_ref s, %s
// CHECK:STDOUT:   %.loc9_24: ref %S = value_as_ref %s.ref.loc9
// CHECK:STDOUT:   %addr.loc9: %ptr.5c7 = addr_of %.loc9_24
// CHECK:STDOUT:   %.loc9_25.1: %ptr.ff5 = as_compatible %addr.loc9
// CHECK:STDOUT:   %.loc9_25.2: %ptr.ff5 = converted %addr.loc9, %.loc9_25.1
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
// CHECK:STDOUT:   %addr.loc11: %ptr.5c7 = addr_of %.loc11_26.5
// CHECK:STDOUT:   %.loc11_40.1: %ptr.ff5 = as_compatible %addr.loc11
// CHECK:STDOUT:   %.loc11_40.2: %ptr.ff5 = converted %addr.loc11, %.loc11_40.1
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.call.loc11: init %empty_tuple.type = call imports.%TakesConstLValue__carbon_thunk.decl(%.loc11_40.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_19.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.016
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_19.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8: %ptr.5c7 = addr_of %.loc8_19.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_return_lvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.259: type = pattern_type %ptr.5c7 [concrete]
// CHECK:STDOUT:   %ReturnsLValue.cpp_overload_set.type: type = cpp_overload_set_type @ReturnsLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ReturnsLValue.cpp_overload_set.value: %ReturnsLValue.cpp_overload_set.type = cpp_overload_set_value @ReturnsLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const: type = const_type %ptr.5c7 [concrete]
// CHECK:STDOUT:   %ReturnsLValue.type: type = fn_type @ReturnsLValue [concrete]
// CHECK:STDOUT:   %ReturnsLValue: %ReturnsLValue.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %ptr.5c7, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.8de: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.4bc: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.8de = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.dfe: type = ptr_type %ptr.5c7 [concrete]
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
// CHECK:STDOUT:   %ReturnsLValue.decl: %ReturnsLValue.type = fn_decl @ReturnsLValue [concrete = constants.%ReturnsLValue] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.259 = value_binding_pattern s [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ReturnsLValue.ref: %ReturnsLValue.cpp_overload_set.type = name_ref ReturnsLValue, imports.%ReturnsLValue.cpp_overload_set.value [concrete = constants.%ReturnsLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %ReturnsLValue.call: init %const = call imports.%ReturnsLValue.decl()
// CHECK:STDOUT:   %.loc8_15: type = splice_block %ptr [concrete = constants.%ptr.5c7] {
// CHECK:STDOUT:     %Cpp.ref.loc8_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr: type = ptr_type %S.ref [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_37.1: init %ptr.5c7 = as_compatible %ReturnsLValue.call
// CHECK:STDOUT:   %.loc8_37.2: init %ptr.5c7 = converted %ReturnsLValue.call, %.loc8_37.1
// CHECK:STDOUT:   %.loc8_37.3: ref %ptr.5c7 = temporary_storage
// CHECK:STDOUT:   %.loc8_37.4: ref %ptr.5c7 = temporary %.loc8_37.3, %.loc8_37.2
// CHECK:STDOUT:   %.loc8_37.5: %ptr.5c7 = bind_value %.loc8_37.4
// CHECK:STDOUT:   %s: %ptr.5c7 = value_binding s, %.loc8_37.5
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_37.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.4bc
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_37.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.dfe = addr_of %.loc8_37.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_return_rvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.259: type = pattern_type %ptr.5c7 [concrete]
// CHECK:STDOUT:   %ReturnsRValue.cpp_overload_set.type: type = cpp_overload_set_type @ReturnsRValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ReturnsRValue.cpp_overload_set.value: %ReturnsRValue.cpp_overload_set.type = cpp_overload_set_value @ReturnsRValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const: type = const_type %ptr.5c7 [concrete]
// CHECK:STDOUT:   %ReturnsRValue.type: type = fn_type @ReturnsRValue [concrete]
// CHECK:STDOUT:   %ReturnsRValue: %ReturnsRValue.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %ptr.5c7, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.8de: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.4bc: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.8de = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.dfe: type = ptr_type %ptr.5c7 [concrete]
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
// CHECK:STDOUT:   %ReturnsRValue.decl: %ReturnsRValue.type = fn_decl @ReturnsRValue [concrete = constants.%ReturnsRValue] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.259 = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.259 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %ptr.5c7 = var %s.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ReturnsRValue.ref: %ReturnsRValue.cpp_overload_set.type = name_ref ReturnsRValue, imports.%ReturnsRValue.cpp_overload_set.value [concrete = constants.%ReturnsRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %ReturnsRValue.call: init %const = call imports.%ReturnsRValue.decl()
// CHECK:STDOUT:   %.loc8_3.1: init %ptr.5c7 = as_compatible %ReturnsRValue.call
// CHECK:STDOUT:   %.loc8_3.2: init %ptr.5c7 = converted %ReturnsRValue.call, %.loc8_3.1
// CHECK:STDOUT:   assign %s.var, %.loc8_3.2
// CHECK:STDOUT:   %.loc8_15: type = splice_block %ptr [concrete = constants.%ptr.5c7] {
// CHECK:STDOUT:     %Cpp.ref.loc8_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr: type = ptr_type %S.ref [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %ptr.5c7 = ref_binding s, %s.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %s.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.4bc
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %s.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.dfe = addr_of %s.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_return_const_lvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %const.e39: type = const_type %S [concrete]
// CHECK:STDOUT:   %ptr.ff5: type = ptr_type %const.e39 [concrete]
// CHECK:STDOUT:   %pattern_type.32f: type = pattern_type %ptr.ff5 [concrete]
// CHECK:STDOUT:   %ReturnConstLValue.cpp_overload_set.type: type = cpp_overload_set_type @ReturnConstLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ReturnConstLValue.cpp_overload_set.value: %ReturnConstLValue.cpp_overload_set.type = cpp_overload_set_value @ReturnConstLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const.179: type = const_type %ptr.ff5 [concrete]
// CHECK:STDOUT:   %ReturnConstLValue.type: type = fn_type @ReturnConstLValue [concrete]
// CHECK:STDOUT:   %ReturnConstLValue: %ReturnConstLValue.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %ptr.ff5, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.40d: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.c44: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.40d = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.dec: type = ptr_type %ptr.ff5 [concrete]
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
// CHECK:STDOUT:   %ReturnConstLValue.decl: %ReturnConstLValue.type = fn_decl @ReturnConstLValue [concrete = constants.%ReturnConstLValue] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.32f = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.32f = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %ptr.ff5 = var %s.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc8_25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ReturnConstLValue.ref: %ReturnConstLValue.cpp_overload_set.type = name_ref ReturnConstLValue, imports.%ReturnConstLValue.cpp_overload_set.value [concrete = constants.%ReturnConstLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %ReturnConstLValue.call: init %const.179 = call imports.%ReturnConstLValue.decl()
// CHECK:STDOUT:   %.loc8_3.1: init %ptr.ff5 = as_compatible %ReturnConstLValue.call
// CHECK:STDOUT:   %.loc8_3.2: init %ptr.ff5 = converted %ReturnConstLValue.call, %.loc8_3.1
// CHECK:STDOUT:   assign %s.var, %.loc8_3.2
// CHECK:STDOUT:   %.loc8_21: type = splice_block %ptr [concrete = constants.%ptr.ff5] {
// CHECK:STDOUT:     %Cpp.ref.loc8_16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %const: type = const_type %S.ref [concrete = constants.%const.e39]
// CHECK:STDOUT:     %ptr: type = ptr_type %const [concrete = constants.%ptr.ff5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %ptr.ff5 = ref_binding s, %s.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %s.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.c44
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %s.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.dec = addr_of %s.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
