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

// --- lvalue_ref.h

struct S {};
struct T {};

auto TakesLValue(S&) -> void;

// --- call_lvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "lvalue_ref.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.S = {};
  Cpp.TakesLValue(s);
  //@dump-sem-ir-end
}

// --- fail_lvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "lvalue_ref.h";

fn F() {
  //@dump-sem-ir-begin
  var v: Cpp.S;
  let s: Cpp.S = v;
  // CHECK:STDERR: fail_lvalue_ref.carbon:[[@LINE+7]]:3: error: no matching function for call to `TakesLValue` [CppOverloadingNoViableFunctionFound]
  // CHECK:STDERR:   Cpp.TakesLValue(s);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_lvalue_ref.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.TakesLValue(s);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.TakesLValue(s);

  var t: Cpp.T;
  // CHECK:STDERR: fail_lvalue_ref.carbon:[[@LINE+7]]:3: error: no matching function for call to `TakesLValue` [CppOverloadingNoViableFunctionFound]
  // CHECK:STDERR:   Cpp.TakesLValue(t);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_lvalue_ref.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.TakesLValue(t);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.TakesLValue(t);

  var u: Cpp.S;
  // CHECK:STDERR: fail_lvalue_ref.carbon:[[@LINE+7]]:3: error: no matching function for call to `TakesLValue` [CppOverloadingNoViableFunctionFound]
  // CHECK:STDERR:   Cpp.TakesLValue(u as const Cpp.S);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_lvalue_ref.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.TakesLValue(u as const Cpp.S);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.TakesLValue(u as const Cpp.S);
  //@dump-sem-ir-end
}

// ============================================================================
// Rvalue reference as a parameter type
// ============================================================================

// --- rvalue_ref.h

struct S {};
struct T {};

auto TakesRValue(S&&) -> void;

// --- call_rvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "rvalue_ref.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.TakesRValue({} as Cpp.S);
  //@dump-sem-ir-end
}

// --- todo_fail_value_arg_for_rvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "rvalue_ref.h";

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

// --- fail_rvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "rvalue_ref.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.S;
  // CHECK:STDERR: fail_rvalue_ref.carbon:[[@LINE+7]]:3: error: no matching function for call to `TakesRValue` [CppOverloadingNoViableFunctionFound]
  // CHECK:STDERR:   Cpp.TakesRValue(s);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_rvalue_ref.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.TakesRValue(s);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.TakesRValue(s);

  var t: Cpp.T;
  // CHECK:STDERR: fail_rvalue_ref.carbon:[[@LINE+7]]:3: error: no matching function for call to `TakesRValue` [CppOverloadingNoViableFunctionFound]
  // CHECK:STDERR:   Cpp.TakesRValue(t);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_rvalue_ref.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.TakesRValue(t);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.TakesRValue(t);

  // CHECK:STDERR: fail_rvalue_ref.carbon:[[@LINE+7]]:3: error: no matching function for call to `TakesRValue` [CppOverloadingNoViableFunctionFound]
  // CHECK:STDERR:   Cpp.TakesRValue(({} as Cpp.S) as const Cpp.S);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_rvalue_ref.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.TakesRValue(({} as Cpp.S) as const Cpp.S);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.TakesRValue(({} as Cpp.S) as const Cpp.S);
  //@dump-sem-ir-end
}

// ============================================================================
// Const reference as a parameter type
// ============================================================================

// --- const_lvalue_ref.h

struct S {};
struct T {};

auto TakesConstLValue(const S&) -> void;

// --- call_const_lvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "const_lvalue_ref.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.S = {};
  Cpp.TakesConstLValue(s as const Cpp.S);

  Cpp.TakesConstLValue(s);
  //@dump-sem-ir-end
}

// --- fail_const_lvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "const_lvalue_ref.h";

fn F() {
  //@dump-sem-ir-begin
  // TODO: Should this work? We could create a temporary. Overload resolution
  // accepts this but the Carbon-side call fails.
  // TODO: The diagnostic here is wrong; we're internally using `addr` but this
  // is not `addr self`.
  let s: Cpp.S = {};
  // CHECK:STDERR: fail_const_lvalue_ref.carbon:[[@LINE+5]]:24: error: `addr self` method cannot be invoked on a value [AddrSelfIsNonRef]
  // CHECK:STDERR:   Cpp.TakesConstLValue(s);
  // CHECK:STDERR:                        ^
  // CHECK:STDERR: fail_const_lvalue_ref.carbon: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.TakesConstLValue(s);

  var t: Cpp.T;
  // CHECK:STDERR: fail_const_lvalue_ref.carbon:[[@LINE+7]]:3: error: no matching function for call to `TakesConstLValue` [CppOverloadingNoViableFunctionFound]
  // CHECK:STDERR:   Cpp.TakesConstLValue(t);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_const_lvalue_ref.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.TakesConstLValue(t);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.TakesConstLValue(t);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- call_lvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %.547: type = cpp_overload_set_type @TakesLValue [concrete]
// CHECK:STDOUT:   %empty_struct: %.547 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %TakesLValue__carbon_thunk.type: type = fn_type @TakesLValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesLValue__carbon_thunk: %TakesLValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.d46: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%S) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.062: %T.as.Destroy.impl.Op.type.d46 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .TakesLValue = %.a7f
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.a7f: %.547 = cpp_overload_set_value @TakesLValue [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:   %.loc8_3: init %S = converted %.loc8_19.1, %.loc8_19.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = bind_name s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesLValue.ref: %.547 = name_ref TakesLValue, imports.%.a7f [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %addr.loc9: %ptr.5c7 = addr_of %s.ref
// CHECK:STDOUT:   %TakesLValue__carbon_thunk.call: init %empty_tuple.type = call imports.%TakesLValue__carbon_thunk.decl(%addr.loc9)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %s.var, constants.%T.as.Destroy.impl.Op.062
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %s.var, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8: %ptr.5c7 = addr_of %s.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_lvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %.547: type = cpp_overload_set_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %empty_struct: %.547 = struct_value () [concrete]
// CHECK:STDOUT:   %T.e15: type = class_type @T [concrete]
// CHECK:STDOUT:   %pattern_type.e6b: type = pattern_type %T.e15 [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.d46: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%S) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.062: %T.as.Destroy.impl.Op.type.d46 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.559: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%T.e15) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.b53: %T.as.Destroy.impl.Op.type.559 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b04: type = ptr_type %T.e15 [concrete]
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
// CHECK:STDOUT:   %.a7f: %.547 = cpp_overload_set_value @Destroy.Op [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %T.decl: type = class_decl @T [concrete = constants.%T.e15] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.7da = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.7da = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %S = var %v.var_patt
// CHECK:STDOUT:   %.loc8: type = splice_block %S.ref.loc8 [concrete = constants.%S] {
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
// CHECK:STDOUT:   %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesLValue.ref.loc17: %.547 = name_ref TakesLValue, imports.%.a7f [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %s.ref: %S = name_ref s, %s
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type.e6b = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.var_patt: %pattern_type.e6b = var_pattern %t.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %T.e15 = var %t.var_patt
// CHECK:STDOUT:   %.loc19: type = splice_block %T.ref [concrete = constants.%T.e15] {
// CHECK:STDOUT:     %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %T.ref: type = name_ref T, imports.%T.decl [concrete = constants.%T.e15]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %T.e15 = bind_name t, %t.var
// CHECK:STDOUT:   %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesLValue.ref.loc27: %.547 = name_ref TakesLValue, imports.%.a7f [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %t.ref: ref %T.e15 = name_ref t, %t
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %u.patt: %pattern_type.7da = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.var_patt: %pattern_type.7da = var_pattern %u.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %u.var: ref %S = var %u.var_patt
// CHECK:STDOUT:   %.loc29: type = splice_block %S.ref.loc29 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc29: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %u: ref %S = bind_name u, %u.var
// CHECK:STDOUT:   %Cpp.ref.loc37_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesLValue.ref.loc37: %.547 = name_ref TakesLValue, imports.%.a7f [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %u.ref: ref %S = name_ref u, %u
// CHECK:STDOUT:   %Cpp.ref.loc37_30: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc37: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %const: type = const_type %S.ref.loc37 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc37_21.1: ref %const = as_compatible %u.ref
// CHECK:STDOUT:   %.loc37_21.2: ref %const = converted %u.ref, %.loc37_21.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc29: <bound method> = bound_method %u.var, constants.%T.as.Destroy.impl.Op.062
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc29: <bound method> = bound_method %u.var, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc29: %ptr.5c7 = addr_of %u.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc29: init %empty_tuple.type = call %bound_method.loc29(%addr.loc29)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc19: <bound method> = bound_method %t.var, constants.%T.as.Destroy.impl.Op.b53
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc19: <bound method> = bound_method %t.var, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc19: %ptr.b04 = addr_of %t.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc19: init %empty_tuple.type = call %bound_method.loc19(%addr.loc19)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %v.var, constants.%T.as.Destroy.impl.Op.062
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %v.var, %T.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc8: %ptr.5c7 = addr_of %v.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_rvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %.1b9: type = cpp_overload_set_type @TakesRValue [concrete]
// CHECK:STDOUT:   %empty_struct: %.1b9 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %TakesRValue__carbon_thunk.type: type = fn_type @TakesRValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesRValue__carbon_thunk: %TakesRValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.d46: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%S) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.062: %T.as.Destroy.impl.Op.type.d46 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TakesRValue = %.1f6
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.1f6: %.1b9 = cpp_overload_set_value @TakesRValue [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:   %TakesRValue.ref: %.1b9 = name_ref TakesRValue, imports.%.1f6 [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_20.4, constants.%T.as.Destroy.impl.Op.062
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_20.4, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_20: %ptr.5c7 = addr_of %.loc8_20.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8_20)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_value_arg_for_rvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %.1b9: type = cpp_overload_set_type @TakesRValue [concrete]
// CHECK:STDOUT:   %empty_struct: %.1b9 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %TakesRValue__carbon_thunk.type: type = fn_type @TakesRValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesRValue__carbon_thunk: %TakesRValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.d46: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%S) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.062: %T.as.Destroy.impl.Op.type.d46 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .TakesRValue = %.1f6
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.1f6: %.1b9 = cpp_overload_set_value @TakesRValue [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:   %TakesRValue.ref: %.1b9 = name_ref TakesRValue, imports.%.1f6 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %s.ref: %S = name_ref s, %s
// CHECK:STDOUT:   %.loc13: ref %S = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc13: %ptr.5c7 = addr_of %.loc13
// CHECK:STDOUT:   %TakesRValue__carbon_thunk.call: init %empty_tuple.type = call imports.%TakesRValue__carbon_thunk.decl(%addr.loc13)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc12_19.4, constants.%T.as.Destroy.impl.Op.062
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc12_19.4, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc12: %ptr.5c7 = addr_of %.loc12_19.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc12)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_rvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %.1b9: type = cpp_overload_set_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %empty_struct: %.1b9 = struct_value () [concrete]
// CHECK:STDOUT:   %T.e15: type = class_type @T [concrete]
// CHECK:STDOUT:   %pattern_type.e6b: type = pattern_type %T.e15 [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.d46: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%S) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.062: %T.as.Destroy.impl.Op.type.d46 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.559: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%T.e15) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.b53: %T.as.Destroy.impl.Op.type.559 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b04: type = ptr_type %T.e15 [concrete]
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
// CHECK:STDOUT:   %.1f6: %.1b9 = cpp_overload_set_value @Destroy.Op [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %T.decl: type = class_decl @T [concrete = constants.%T.e15] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc8: type = splice_block %S.ref.loc8 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = bind_name s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesRValue.ref.loc16: %.1b9 = name_ref TakesRValue, imports.%.1f6 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type.e6b = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.var_patt: %pattern_type.e6b = var_pattern %t.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %T.e15 = var %t.var_patt
// CHECK:STDOUT:   %.loc18: type = splice_block %T.ref [concrete = constants.%T.e15] {
// CHECK:STDOUT:     %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %T.ref: type = name_ref T, imports.%T.decl [concrete = constants.%T.e15]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %T.e15 = bind_name t, %t.var
// CHECK:STDOUT:   %Cpp.ref.loc26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesRValue.ref.loc26: %.1b9 = name_ref TakesRValue, imports.%.1f6 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %t.ref: ref %T.e15 = name_ref t, %t
// CHECK:STDOUT:   %Cpp.ref.loc35_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesRValue.ref.loc35: %.1b9 = name_ref TakesRValue, imports.%.1f6 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc35_21.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc35_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc35_29: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %.loc35_21.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc35_21.3: init %S = class_init (), %.loc35_21.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc35_21.4: ref %S = temporary %.loc35_21.2, %.loc35_21.3
// CHECK:STDOUT:   %.loc35_23: ref %S = converted %.loc35_21.1, %.loc35_21.4
// CHECK:STDOUT:   %Cpp.ref.loc35_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc35_45: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %const: type = const_type %S.ref.loc35_45 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc35_33.1: ref %const = as_compatible %.loc35_23
// CHECK:STDOUT:   %.loc35_33.2: ref %const = converted %.loc35_23, %.loc35_33.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc35: <bound method> = bound_method %.loc35_21.4, constants.%T.as.Destroy.impl.Op.062
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc35: <bound method> = bound_method %.loc35_21.4, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc35: %ptr.5c7 = addr_of %.loc35_21.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc35: init %empty_tuple.type = call %bound_method.loc35(%addr.loc35)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc18: <bound method> = bound_method %t.var, constants.%T.as.Destroy.impl.Op.b53
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc18: <bound method> = bound_method %t.var, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc18: %ptr.b04 = addr_of %t.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc18: init %empty_tuple.type = call %bound_method.loc18(%addr.loc18)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %s.var, constants.%T.as.Destroy.impl.Op.062
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %s.var, %T.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc8: %ptr.5c7 = addr_of %s.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_const_lvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %.4e1: type = cpp_overload_set_type @TakesConstLValue [concrete]
// CHECK:STDOUT:   %empty_struct: %.4e1 = struct_value () [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %ptr.ff5: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.type: type = fn_type @TakesConstLValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk: %TakesConstLValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.d46: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%S) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.062: %T.as.Destroy.impl.Op.type.d46 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .TakesConstLValue = %.9bc
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.9bc: %.4e1 = cpp_overload_set_value @TakesConstLValue [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:   %.loc8_3: init %S = converted %.loc8_19.1, %.loc8_19.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref.loc8 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = bind_name s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc9_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstLValue.ref.loc9: %.4e1 = name_ref TakesConstLValue, imports.%.9bc [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %s.ref.loc9: ref %S = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc9_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc9: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %const: type = const_type %S.ref.loc9 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc9_26.1: ref %const = as_compatible %s.ref.loc9
// CHECK:STDOUT:   %.loc9_26.2: ref %const = converted %s.ref.loc9, %.loc9_26.1
// CHECK:STDOUT:   %addr.loc9: %ptr.ff5 = addr_of %.loc9_26.2
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%TakesConstLValue__carbon_thunk.decl(%addr.loc9)
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstLValue.ref.loc11: %.4e1 = name_ref TakesConstLValue, imports.%.9bc [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %s.ref.loc11: ref %S = name_ref s, %s
// CHECK:STDOUT:   %addr.loc11: %ptr.5c7 = addr_of %s.ref.loc11
// CHECK:STDOUT:   %.loc11_24.1: %ptr.ff5 = as_compatible %addr.loc11
// CHECK:STDOUT:   %.loc11_24.2: %ptr.ff5 = converted %addr.loc11, %.loc11_24.1
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.call.loc11: init %empty_tuple.type = call imports.%TakesConstLValue__carbon_thunk.decl(%.loc11_24.2)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %s.var, constants.%T.as.Destroy.impl.Op.062
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %s.var, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8: %ptr.5c7 = addr_of %s.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_const_lvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %.4e1: type = cpp_overload_set_type @TakesConstLValue [concrete]
// CHECK:STDOUT:   %empty_struct: %.4e1 = struct_value () [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %ptr.ff5: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.type: type = fn_type @TakesConstLValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk: %TakesConstLValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %T.e15: type = class_type @T [concrete]
// CHECK:STDOUT:   %pattern_type.e6b: type = pattern_type %T.e15 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.559: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%T.e15) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.b53: %T.as.Destroy.impl.Op.type.559 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b04: type = ptr_type %T.e15 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.d46: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%S) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.062: %T.as.Destroy.impl.Op.type.d46 = struct_value () [concrete]
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
// CHECK:STDOUT:   %.9bc: %.4e1 = cpp_overload_set_value @TakesConstLValue [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.decl: %TakesConstLValue__carbon_thunk.type = fn_decl @TakesConstLValue__carbon_thunk [concrete = constants.%TakesConstLValue__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T.decl: type = class_decl @T [concrete = constants.%T.e15] {} {}
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
// CHECK:STDOUT:   %TakesConstLValue.ref.loc18: %.4e1 = name_ref TakesConstLValue, imports.%.9bc [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:   %t.var: ref %T.e15 = var %t.var_patt
// CHECK:STDOUT:   %.loc20: type = splice_block %T.ref [concrete = constants.%T.e15] {
// CHECK:STDOUT:     %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %T.ref: type = name_ref T, imports.%T.decl [concrete = constants.%T.e15]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %T.e15 = bind_name t, %t.var
// CHECK:STDOUT:   %Cpp.ref.loc28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstLValue.ref.loc28: %.4e1 = name_ref TakesConstLValue, imports.%.9bc [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %t.ref: ref %T.e15 = name_ref t, %t
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc20: <bound method> = bound_method %t.var, constants.%T.as.Destroy.impl.Op.b53
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc20: <bound method> = bound_method %t.var, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc20: %ptr.b04 = addr_of %t.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc20: init %empty_tuple.type = call %bound_method.loc20(%addr.loc20)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc12: <bound method> = bound_method %.loc12_19.4, constants.%T.as.Destroy.impl.Op.062
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %.loc12_19.4, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc12: %ptr.5c7 = addr_of %.loc12_19.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc12: init %empty_tuple.type = call %bound_method.loc12(%addr.loc12)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
