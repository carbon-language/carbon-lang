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

  var t: Cpp.T;
  // CHECK:STDERR: fail_param_lvalue_ref.carbon:[[@LINE+8]]:20: error: no matching function for call to 'TakesLValue' [CppInteropParseError]
  // CHECK:STDERR:    29 |   Cpp.TakesLValue(t);
  // CHECK:STDERR:       |                    ^
  // CHECK:STDERR: fail_param_lvalue_ref.carbon:[[@LINE-20]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_lvalue_ref.h:5:6: note: candidate function not viable: no known conversion from 'T' to 'S &' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | auto TakesLValue(S&) -> void;
  // CHECK:STDERR:       |      ^           ~~
  // CHECK:STDERR:
  Cpp.TakesLValue(t);

  var u: Cpp.S;
  // CHECK:STDERR: fail_param_lvalue_ref.carbon:[[@LINE+8]]:35: error: no matching function for call to 'TakesLValue' [CppInteropParseError]
  // CHECK:STDERR:    40 |   Cpp.TakesLValue(u as const Cpp.S);
  // CHECK:STDERR:       |                                   ^
  // CHECK:STDERR: fail_param_lvalue_ref.carbon:[[@LINE-31]]:10: in file included here [InCppInclude]
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

// --- call_param_const_lvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "param_const_lvalue_ref.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.S = {};
  Cpp.TakesConstLValue(s as const Cpp.S);

  Cpp.TakesConstLValue(s);
  //@dump-sem-ir-end
}

// --- fail_param_const_lvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "param_const_lvalue_ref.h";

fn F() {
  //@dump-sem-ir-begin
  // TODO: Should this work? We could create a temporary. Overload resolution
  // accepts this but the Carbon-side call fails.
  // TODO: The diagnostic here is wrong; we're internally using `addr` but this
  // is not `addr self`.
  let s: Cpp.S = {};
  // CHECK:STDERR: fail_param_const_lvalue_ref.carbon:[[@LINE+5]]:24: error: `addr self` method cannot be invoked on a value [AddrSelfIsNonRef]
  // CHECK:STDERR:   Cpp.TakesConstLValue(s);
  // CHECK:STDERR:                        ^
  // CHECK:STDERR: fail_param_const_lvalue_ref.carbon: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.TakesConstLValue(s);

  var t: Cpp.T;
  // CHECK:STDERR: fail_param_const_lvalue_ref.carbon:[[@LINE+8]]:25: error: no matching function for call to 'TakesConstLValue' [CppInteropParseError]
  // CHECK:STDERR:    29 |   Cpp.TakesConstLValue(t);
  // CHECK:STDERR:       |                         ^
  // CHECK:STDERR: fail_param_const_lvalue_ref.carbon:[[@LINE-20]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_const_lvalue_ref.h:5:6: note: candidate function not viable: no known conversion from 'T' to 'const S' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | auto TakesConstLValue(const S&) -> void;
  // CHECK:STDERR:       |      ^                ~~~~~~~~
  // CHECK:STDERR:
  Cpp.TakesConstLValue(t);
  //@dump-sem-ir-end
}

// ============================================================================
// Lvalue reference as return type
// ============================================================================

// --- fail_todo_call_return_lvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct S {};

auto ReturnsLValue() -> S&;
''';

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_call_return_lvalue_ref.carbon:[[@LINE+4]]:19: error: semantics TODO: `Unsupported: return type: S &` [SemanticsTodo]
  // CHECK:STDERR:   let s: Cpp.S* = Cpp.ReturnsLValue();
  // CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let s: Cpp.S* = Cpp.ReturnsLValue();
  //@dump-sem-ir-end
}

// ============================================================================
// Rvalue reference as return type
// ============================================================================

// --- fail_todo_call_return_rvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct S {};

auto ReturnsRValue() -> S&&;
''';

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_call_return_rvalue_ref.carbon:[[@LINE+4]]:18: error: semantics TODO: `Unsupported: return type: S &&` [SemanticsTodo]
  // CHECK:STDERR:   var s: Cpp.S = Cpp.ReturnsRValue();
  // CHECK:STDERR:                  ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var s: Cpp.S = Cpp.ReturnsRValue();
  //@dump-sem-ir-end
}

// ============================================================================
// Const reference as return type
// ============================================================================

// --- fail_todo_call_return_const_lvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct S {};

auto ReturnConstLValue() -> const S&;
''';

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_call_return_const_lvalue_ref.carbon:[[@LINE+4]]:24: error: semantics TODO: `Unsupported: return type: const S &` [SemanticsTodo]
  // CHECK:STDERR:   var s: const Cpp.S = Cpp.ReturnConstLValue();
  // CHECK:STDERR:                        ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var s: const Cpp.S = Cpp.ReturnConstLValue();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- call_param_lvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %.547: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.a7f: %.547 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %TakesLValue__carbon_thunk.type: type = fn_type @TakesLValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesLValue__carbon_thunk: %TakesLValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.df1: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.cc2: %DestroyT.as_type.as.Destroy.impl.Op.type.df1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .TakesLValue = %.a7f
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.a7f: %.547 = cpp_overload_set_value @<null name> [concrete = constants.%.a7f]
// CHECK:STDOUT:   %TakesLValue__carbon_thunk.decl: %TakesLValue__carbon_thunk.type = fn_decl @TakesLValue__carbon_thunk [concrete = constants.%TakesLValue__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc8_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_19.2: init %S = class_init (), %s.var [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_3.1: init %S = converted %.loc8_19.1, %.loc8_19.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3.1
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = bind_name s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesLValue.ref: %.547 = name_ref TakesLValue, imports.%.a7f [concrete = constants.%.a7f]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %addr.loc9: %ptr.5c7 = addr_of %s.ref
// CHECK:STDOUT:   %TakesLValue__carbon_thunk.call: init %empty_tuple.type = call imports.%TakesLValue__carbon_thunk.decl(%addr.loc9)
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%S, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_3.2: %type_where = converted constants.%S, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %s.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.cc2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %s.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8: %ptr.5c7 = addr_of %s.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_param_lvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %.547: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.a7f: %.547 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %T: type = class_type @T [concrete]
// CHECK:STDOUT:   %pattern_type.e6b: type = pattern_type %T [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.7bd: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.df1: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value.7bd) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.cc2: %DestroyT.as_type.as.Destroy.impl.Op.type.df1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %facet_value.19d: %type_where = facet_value %T, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.896: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value.19d) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.8c8: %DestroyT.as_type.as.Destroy.impl.Op.type.896 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b04: type = ptr_type %T [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .TakesLValue = %.a7f
// CHECK:STDOUT:     .T = %T.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.a7f: %.547 = cpp_overload_set_value @<null name> [concrete = constants.%.a7f]
// CHECK:STDOUT:   %T.decl: type = class_decl @T [concrete = constants.%T] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.7da = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.7da = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %S = var %v.var_patt
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref.loc8 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %S = bind_name v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = binding_pattern s [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.ref: ref %S = name_ref v, %v
// CHECK:STDOUT:   %.loc9_13: type = splice_block %S.ref.loc9 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc9: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_18: %S = bind_value %v.ref
// CHECK:STDOUT:   %s: %S = bind_name s, %.loc9_18
// CHECK:STDOUT:   %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesLValue.ref.loc18: %.547 = name_ref TakesLValue, imports.%.a7f [concrete = constants.%.a7f]
// CHECK:STDOUT:   %s.ref: %S = name_ref s, %s
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type.e6b = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.var_patt: %pattern_type.e6b = var_pattern %t.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %T = var %t.var_patt
// CHECK:STDOUT:   %.loc20_13: type = splice_block %T.ref [concrete = constants.%T] {
// CHECK:STDOUT:     %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %T.ref: type = name_ref T, imports.%T.decl [concrete = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %T = bind_name t, %t.var
// CHECK:STDOUT:   %Cpp.ref.loc29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesLValue.ref.loc29: %.547 = name_ref TakesLValue, imports.%.a7f [concrete = constants.%.a7f]
// CHECK:STDOUT:   %t.ref: ref %T = name_ref t, %t
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %u.patt: %pattern_type.7da = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.var_patt: %pattern_type.7da = var_pattern %u.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %u.var: ref %S = var %u.var_patt
// CHECK:STDOUT:   %.loc31_13: type = splice_block %S.ref.loc31 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc31: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %u: ref %S = bind_name u, %u.var
// CHECK:STDOUT:   %Cpp.ref.loc40_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesLValue.ref.loc40: %.547 = name_ref TakesLValue, imports.%.a7f [concrete = constants.%.a7f]
// CHECK:STDOUT:   %u.ref: ref %S = name_ref u, %u
// CHECK:STDOUT:   %Cpp.ref.loc40_30: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc40: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %const: type = const_type %S.ref.loc40 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc40_21.1: ref %const = as_compatible %u.ref
// CHECK:STDOUT:   %.loc40_21.2: ref %const = converted %u.ref, %.loc40_21.1
// CHECK:STDOUT:   %facet_value.loc31: %type_where = facet_value constants.%S, () [concrete = constants.%facet_value.7bd]
// CHECK:STDOUT:   %.loc31_3: %type_where = converted constants.%S, %facet_value.loc31 [concrete = constants.%facet_value.7bd]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc31: <bound method> = bound_method %u.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.cc2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc31: <bound method> = bound_method %u.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc31: %ptr.5c7 = addr_of %u.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc31: init %empty_tuple.type = call %bound_method.loc31(%addr.loc31)
// CHECK:STDOUT:   %facet_value.loc20: %type_where = facet_value constants.%T, () [concrete = constants.%facet_value.19d]
// CHECK:STDOUT:   %.loc20_3: %type_where = converted constants.%T, %facet_value.loc20 [concrete = constants.%facet_value.19d]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc20: <bound method> = bound_method %t.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.8c8
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc20: <bound method> = bound_method %t.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc20: %ptr.b04 = addr_of %t.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc20: init %empty_tuple.type = call %bound_method.loc20(%addr.loc20)
// CHECK:STDOUT:   %facet_value.loc8: %type_where = facet_value constants.%S, () [concrete = constants.%facet_value.7bd]
// CHECK:STDOUT:   %.loc8_3: %type_where = converted constants.%S, %facet_value.loc8 [concrete = constants.%facet_value.7bd]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %v.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.cc2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %v.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc8: %ptr.5c7 = addr_of %v.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_param_rvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.1b9: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.1f6: %.1b9 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %TakesRValue__carbon_thunk.type: type = fn_type @TakesRValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesRValue__carbon_thunk: %TakesRValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.df1: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.cc2: %DestroyT.as_type.as.Destroy.impl.Op.type.df1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TakesRValue = %.1f6
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1f6: %.1b9 = cpp_overload_set_value @<null name> [concrete = constants.%.1f6]
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
// CHECK:STDOUT:   %TakesRValue.ref: %.1b9 = name_ref TakesRValue, imports.%.1f6 [concrete = constants.%.1f6]
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
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%S, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_20.5: %type_where = converted constants.%S, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_20.4, constants.%DestroyT.as_type.as.Destroy.impl.Op.cc2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_20.4, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_20: %ptr.5c7 = addr_of %.loc8_20.4
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8_20)
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
// CHECK:STDOUT:   %.1b9: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.1f6: %.1b9 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %TakesRValue__carbon_thunk.type: type = fn_type @TakesRValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesRValue__carbon_thunk: %TakesRValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.df1: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.cc2: %DestroyT.as_type.as.Destroy.impl.Op.type.df1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .TakesRValue = %.1f6
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.1f6: %.1b9 = cpp_overload_set_value @<null name> [concrete = constants.%.1f6]
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
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = binding_pattern s [concrete]
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
// CHECK:STDOUT:   %s: %S = bind_name s, %.loc12_19.6
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesRValue.ref: %.1b9 = name_ref TakesRValue, imports.%.1f6 [concrete = constants.%.1f6]
// CHECK:STDOUT:   %s.ref: %S = name_ref s, %s
// CHECK:STDOUT:   %.loc13: ref %S = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc13: %ptr.5c7 = addr_of %.loc13
// CHECK:STDOUT:   %TakesRValue__carbon_thunk.call: init %empty_tuple.type = call imports.%TakesRValue__carbon_thunk.decl(%addr.loc13)
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%S, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc12_19.7: %type_where = converted constants.%S, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc12_19.4, constants.%DestroyT.as_type.as.Destroy.impl.Op.cc2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc12_19.4, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc12: %ptr.5c7 = addr_of %.loc12_19.4
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc12)
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
// CHECK:STDOUT:   %.1b9: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.1f6: %.1b9 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %T: type = class_type @T [concrete]
// CHECK:STDOUT:   %pattern_type.e6b: type = pattern_type %T [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.7bd: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.df1: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value.7bd) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.cc2: %DestroyT.as_type.as.Destroy.impl.Op.type.df1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %facet_value.19d: %type_where = facet_value %T, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.896: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value.19d) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.8c8: %DestroyT.as_type.as.Destroy.impl.Op.type.896 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b04: type = ptr_type %T [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .TakesRValue = %.1f6
// CHECK:STDOUT:     .T = %T.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.1f6: %.1b9 = cpp_overload_set_value @<null name> [concrete = constants.%.1f6]
// CHECK:STDOUT:   %T.decl: type = class_decl @T [concrete = constants.%T] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref.loc8 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = bind_name s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesRValue.ref.loc17: %.1b9 = name_ref TakesRValue, imports.%.1f6 [concrete = constants.%.1f6]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type.e6b = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.var_patt: %pattern_type.e6b = var_pattern %t.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %T = var %t.var_patt
// CHECK:STDOUT:   %.loc19_13: type = splice_block %T.ref [concrete = constants.%T] {
// CHECK:STDOUT:     %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %T.ref: type = name_ref T, imports.%T.decl [concrete = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %T = bind_name t, %t.var
// CHECK:STDOUT:   %Cpp.ref.loc28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesRValue.ref.loc28: %.1b9 = name_ref TakesRValue, imports.%.1f6 [concrete = constants.%.1f6]
// CHECK:STDOUT:   %t.ref: ref %T = name_ref t, %t
// CHECK:STDOUT:   %Cpp.ref.loc38_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesRValue.ref.loc38: %.1b9 = name_ref TakesRValue, imports.%.1f6 [concrete = constants.%.1f6]
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
// CHECK:STDOUT:   %facet_value.loc38: %type_where = facet_value constants.%S, () [concrete = constants.%facet_value.7bd]
// CHECK:STDOUT:   %.loc38_21.5: %type_where = converted constants.%S, %facet_value.loc38 [concrete = constants.%facet_value.7bd]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc38: <bound method> = bound_method %.loc38_21.4, constants.%DestroyT.as_type.as.Destroy.impl.Op.cc2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc38: <bound method> = bound_method %.loc38_21.4, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc38: %ptr.5c7 = addr_of %.loc38_21.4
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc38: init %empty_tuple.type = call %bound_method.loc38(%addr.loc38)
// CHECK:STDOUT:   %facet_value.loc19: %type_where = facet_value constants.%T, () [concrete = constants.%facet_value.19d]
// CHECK:STDOUT:   %.loc19_3: %type_where = converted constants.%T, %facet_value.loc19 [concrete = constants.%facet_value.19d]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc19: <bound method> = bound_method %t.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.8c8
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc19: <bound method> = bound_method %t.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc19: %ptr.b04 = addr_of %t.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc19: init %empty_tuple.type = call %bound_method.loc19(%addr.loc19)
// CHECK:STDOUT:   %facet_value.loc8: %type_where = facet_value constants.%S, () [concrete = constants.%facet_value.7bd]
// CHECK:STDOUT:   %.loc8_3: %type_where = converted constants.%S, %facet_value.loc8 [concrete = constants.%facet_value.7bd]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %s.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.cc2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %s.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc8: %ptr.5c7 = addr_of %s.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_param_const_lvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %.4e1: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.9bc: %.4e1 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %ptr.ff5: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.type: type = fn_type @TakesConstLValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk: %TakesConstLValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.df1: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.cc2: %DestroyT.as_type.as.Destroy.impl.Op.type.df1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .TakesConstLValue = %.9bc
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.9bc: %.4e1 = cpp_overload_set_value @<null name> [concrete = constants.%.9bc]
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
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc8_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_19.2: init %S = class_init (), %s.var [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_3.1: init %S = converted %.loc8_19.1, %.loc8_19.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3.1
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref.loc8 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = bind_name s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc9_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstLValue.ref.loc9: %.4e1 = name_ref TakesConstLValue, imports.%.9bc [concrete = constants.%.9bc]
// CHECK:STDOUT:   %s.ref.loc9: ref %S = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc9_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc9: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %const: type = const_type %S.ref.loc9 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc9_26.1: ref %const = as_compatible %s.ref.loc9
// CHECK:STDOUT:   %.loc9_26.2: ref %const = converted %s.ref.loc9, %.loc9_26.1
// CHECK:STDOUT:   %addr.loc9: %ptr.ff5 = addr_of %.loc9_26.2
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%TakesConstLValue__carbon_thunk.decl(%addr.loc9)
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstLValue.ref.loc11: %.4e1 = name_ref TakesConstLValue, imports.%.9bc [concrete = constants.%.9bc]
// CHECK:STDOUT:   %s.ref.loc11: ref %S = name_ref s, %s
// CHECK:STDOUT:   %addr.loc11: %ptr.5c7 = addr_of %s.ref.loc11
// CHECK:STDOUT:   %.loc11_24.1: %ptr.ff5 = as_compatible %addr.loc11
// CHECK:STDOUT:   %.loc11_24.2: %ptr.ff5 = converted %addr.loc11, %.loc11_24.1
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.call.loc11: init %empty_tuple.type = call imports.%TakesConstLValue__carbon_thunk.decl(%.loc11_24.2)
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%S, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_3.2: %type_where = converted constants.%S, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %s.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.cc2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %s.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8: %ptr.5c7 = addr_of %s.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_param_const_lvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %.4e1: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.9bc: %.4e1 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %ptr.ff5: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.type: type = fn_type @TakesConstLValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk: %TakesConstLValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %T: type = class_type @T [concrete]
// CHECK:STDOUT:   %pattern_type.e6b: type = pattern_type %T [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.19d: %type_where = facet_value %T, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.896: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value.19d) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.8c8: %DestroyT.as_type.as.Destroy.impl.Op.type.896 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b04: type = ptr_type %T [concrete]
// CHECK:STDOUT:   %facet_value.7bd: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.df1: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value.7bd) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.cc2: %DestroyT.as_type.as.Destroy.impl.Op.type.df1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .TakesConstLValue = %.9bc
// CHECK:STDOUT:     .T = %T.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.9bc: %.4e1 = cpp_overload_set_value @<null name> [concrete = constants.%.9bc]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.decl: %TakesConstLValue__carbon_thunk.type = fn_decl @TakesConstLValue__carbon_thunk [concrete = constants.%TakesConstLValue__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T.decl: type = class_decl @T [concrete = constants.%T] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = binding_pattern s [concrete]
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
// CHECK:STDOUT:   %s: %S = bind_name s, %.loc12_19.6
// CHECK:STDOUT:   %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstLValue.ref.loc18: %.4e1 = name_ref TakesConstLValue, imports.%.9bc [concrete = constants.%.9bc]
// CHECK:STDOUT:   %s.ref: %S = name_ref s, %s
// CHECK:STDOUT:   %.loc18_24.1: ref %S = temporary_storage
// CHECK:STDOUT:   %addr.loc18: %ptr.5c7 = addr_of %.loc18_24.1
// CHECK:STDOUT:   %.loc18_24.2: %ptr.ff5 = as_compatible %addr.loc18
// CHECK:STDOUT:   %.loc18_24.3: %ptr.ff5 = converted %addr.loc18, %.loc18_24.2
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.call: init %empty_tuple.type = call imports.%TakesConstLValue__carbon_thunk.decl(%.loc18_24.3)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type.e6b = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.var_patt: %pattern_type.e6b = var_pattern %t.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %T = var %t.var_patt
// CHECK:STDOUT:   %.loc20_13: type = splice_block %T.ref [concrete = constants.%T] {
// CHECK:STDOUT:     %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %T.ref: type = name_ref T, imports.%T.decl [concrete = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %T = bind_name t, %t.var
// CHECK:STDOUT:   %Cpp.ref.loc29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstLValue.ref.loc29: %.4e1 = name_ref TakesConstLValue, imports.%.9bc [concrete = constants.%.9bc]
// CHECK:STDOUT:   %t.ref: ref %T = name_ref t, %t
// CHECK:STDOUT:   %facet_value.loc20: %type_where = facet_value constants.%T, () [concrete = constants.%facet_value.19d]
// CHECK:STDOUT:   %.loc20_3: %type_where = converted constants.%T, %facet_value.loc20 [concrete = constants.%facet_value.19d]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc20: <bound method> = bound_method %t.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.8c8
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc20: <bound method> = bound_method %t.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc20: %ptr.b04 = addr_of %t.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc20: init %empty_tuple.type = call %bound_method.loc20(%addr.loc20)
// CHECK:STDOUT:   %facet_value.loc12: %type_where = facet_value constants.%S, () [concrete = constants.%facet_value.7bd]
// CHECK:STDOUT:   %.loc12_19.7: %type_where = converted constants.%S, %facet_value.loc12 [concrete = constants.%facet_value.7bd]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc12: <bound method> = bound_method %.loc12_19.4, constants.%DestroyT.as_type.as.Destroy.impl.Op.cc2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %.loc12_19.4, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc12: %ptr.5c7 = addr_of %.loc12_19.4
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc12: init %empty_tuple.type = call %bound_method.loc12(%addr.loc12)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_call_return_lvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %.041: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.03d: %.041 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %ReturnsLValue.type: type = fn_type @ReturnsLValue [concrete]
// CHECK:STDOUT:   %ReturnsLValue: %ReturnsLValue.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .ReturnsLValue = %.03d
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.03d: %.041 = cpp_overload_set_value @<null name> [concrete = constants.%.03d]
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
// CHECK:STDOUT:     %s.patt: %pattern_type = binding_pattern s [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ReturnsLValue.ref: %.041 = name_ref ReturnsLValue, imports.%.03d [concrete = constants.%.03d]
// CHECK:STDOUT:   %ReturnsLValue.call: init <error> = call imports.%ReturnsLValue.decl()
// CHECK:STDOUT:   %.loc16: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:     %Cpp.ref.loc16_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr: type = ptr_type %S.ref [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: %ptr = bind_name s, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_call_return_rvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %.c60: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.99d: %.c60 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %ReturnsRValue.type: type = fn_type @ReturnsRValue [concrete]
// CHECK:STDOUT:   %ReturnsRValue: %ReturnsRValue.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.df1: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.cc2: %DestroyT.as_type.as.Destroy.impl.Op.type.df1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .ReturnsRValue = %.99d
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.99d: %.c60 = cpp_overload_set_value @<null name> [concrete = constants.%.99d]
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
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc16_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ReturnsRValue.ref: %.c60 = name_ref ReturnsRValue, imports.%.99d [concrete = constants.%.99d]
// CHECK:STDOUT:   %ReturnsRValue.call: init <error> = call imports.%ReturnsRValue.decl()
// CHECK:STDOUT:   assign %s.var, <error>
// CHECK:STDOUT:   %.loc16_13: type = splice_block %S.ref [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc16_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = bind_name s, %s.var
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%S, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc16_3: %type_where = converted constants.%S, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %s.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.cc2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %s.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.5c7 = addr_of %s.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_call_return_const_lvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.9be: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %.c99: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.bf8: %.c99 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %ReturnConstLValue.type: type = fn_type @ReturnConstLValue [concrete]
// CHECK:STDOUT:   %ReturnConstLValue: %ReturnConstLValue.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %const, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.31d: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.818: %DestroyT.as_type.as.Destroy.impl.Op.type.31d = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.ff5: type = ptr_type %const [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .ReturnConstLValue = %.bf8
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.bf8: %.c99 = cpp_overload_set_value @<null name> [concrete = constants.%.bf8]
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
// CHECK:STDOUT:     %s.patt: %pattern_type.9be = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.9be = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %const = var %s.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc16_24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ReturnConstLValue.ref: %.c99 = name_ref ReturnConstLValue, imports.%.bf8 [concrete = constants.%.bf8]
// CHECK:STDOUT:   %ReturnConstLValue.call: init <error> = call imports.%ReturnConstLValue.decl()
// CHECK:STDOUT:   assign %s.var, <error>
// CHECK:STDOUT:   %.loc16_10: type = splice_block %const [concrete = constants.%const] {
// CHECK:STDOUT:     %Cpp.ref.loc16_16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %const: type = const_type %S.ref [concrete = constants.%const]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %const = bind_name s, %s.var
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%const, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc16_3: %type_where = converted constants.%const, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %s.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.818
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %s.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.ff5 = addr_of %s.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
