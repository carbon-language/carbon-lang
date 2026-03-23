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
  // CHECK:STDERR:    55 |   Cpp.TakesLValue(s);
  // CHECK:STDERR:       |                    ^
  // CHECK:STDERR: fail_param_lvalue_ref.carbon:[[@LINE-9]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_lvalue_ref.h:22:6: note: candidate function not viable: 1st argument ('const S') would lose const qualifier [CppInteropParseNote]
  // CHECK:STDERR:    22 | auto TakesLValue(S&) -> void;
  // CHECK:STDERR:       |      ^           ~~
  // CHECK:STDERR:
  Cpp.TakesLValue(s);

  // CHECK:STDERR: fail_param_lvalue_ref.carbon:[[@LINE+8]]:35: error: no matching function for call to 'TakesLValue' [CppInteropParseError]
  // CHECK:STDERR:    65 |   Cpp.TakesLValue(s as const Cpp.S);
  // CHECK:STDERR:       |                                   ^
  // CHECK:STDERR: fail_param_lvalue_ref.carbon:[[@LINE-19]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_lvalue_ref.h:22:6: note: candidate function not viable: 1st argument ('const S') would lose const qualifier [CppInteropParseNote]
  // CHECK:STDERR:    22 | auto TakesLValue(S&) -> void;
  // CHECK:STDERR:       |      ^           ~~
  // CHECK:STDERR:
  Cpp.TakesLValue(s as const Cpp.S);

  var t: Cpp.T;
  // CHECK:STDERR: fail_param_lvalue_ref.carbon:[[@LINE+8]]:20: error: no matching function for call to 'TakesLValue' [CppInteropParseError]
  // CHECK:STDERR:    76 |   Cpp.TakesLValue(t);
  // CHECK:STDERR:       |                    ^
  // CHECK:STDERR: fail_param_lvalue_ref.carbon:[[@LINE-30]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_lvalue_ref.h:22:6: note: candidate function not viable: no known conversion from 'T' to 'S &' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:    22 | auto TakesLValue(S&) -> void;
  // CHECK:STDERR:       |      ^           ~~
  // CHECK:STDERR:
  Cpp.TakesLValue(t);

  var u: Cpp.S;
  // CHECK:STDERR: fail_param_lvalue_ref.carbon:[[@LINE+8]]:35: error: no matching function for call to 'TakesLValue' [CppInteropParseError]
  // CHECK:STDERR:    87 |   Cpp.TakesLValue(u as const Cpp.S);
  // CHECK:STDERR:       |                                   ^
  // CHECK:STDERR: fail_param_lvalue_ref.carbon:[[@LINE-41]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_lvalue_ref.h:22:6: note: candidate function not viable: 1st argument ('const S') would lose const qualifier [CppInteropParseNote]
  // CHECK:STDERR:    22 | auto TakesLValue(S&) -> void;
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

// --- fail_param_value_arg_for_rvalue_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "param_rvalue_ref.h";

fn F() {
  let s: Cpp.S = {};
  // CHECK:STDERR: fail_param_value_arg_for_rvalue_ref.carbon:[[@LINE+8]]:20: error: no matching function for call to 'TakesRValue' [CppInteropParseError]
  // CHECK:STDERR:   130 |   Cpp.TakesRValue(s);
  // CHECK:STDERR:       |                    ^
  // CHECK:STDERR: fail_param_value_arg_for_rvalue_ref.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_rvalue_ref.h:100:6: note: candidate function not viable: 1st argument ('const S') would lose const qualifier [CppInteropParseNote]
  // CHECK:STDERR:   100 | auto TakesRValue(S&&) -> void;
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
  // CHECK:STDERR:   150 |   Cpp.TakesRValue(s);
  // CHECK:STDERR:       |                    ^
  // CHECK:STDERR: fail_param_rvalue_ref.carbon:[[@LINE-8]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_rvalue_ref.h:100:6: note: candidate function not viable: expects an rvalue for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:   100 | auto TakesRValue(S&&) -> void;
  // CHECK:STDERR:       |      ^           ~~~
  // CHECK:STDERR:
  Cpp.TakesRValue(s);

  var t: Cpp.T;
  // CHECK:STDERR: fail_param_rvalue_ref.carbon:[[@LINE+8]]:20: error: no matching function for call to 'TakesRValue' [CppInteropParseError]
  // CHECK:STDERR:   161 |   Cpp.TakesRValue(t);
  // CHECK:STDERR:       |                    ^
  // CHECK:STDERR: fail_param_rvalue_ref.carbon:[[@LINE-19]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_rvalue_ref.h:100:6: note: candidate function not viable: no known conversion from 'T' to 'S' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:   100 | auto TakesRValue(S&&) -> void;
  // CHECK:STDERR:       |      ^           ~~~
  // CHECK:STDERR:
  Cpp.TakesRValue(t);

  // CHECK:STDERR: fail_param_rvalue_ref.carbon:[[@LINE+8]]:47: error: no matching function for call to 'TakesRValue' [CppInteropParseError]
  // CHECK:STDERR:   171 |   Cpp.TakesRValue(({} as Cpp.S) as const Cpp.S);
  // CHECK:STDERR:       |                                               ^
  // CHECK:STDERR: fail_param_rvalue_ref.carbon:[[@LINE-29]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_rvalue_ref.h:100:6: note: candidate function not viable: 1st argument ('const S') would lose const qualifier [CppInteropParseNote]
  // CHECK:STDERR:   100 | auto TakesRValue(S&&) -> void;
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
  // CHECK:STDERR:   232 |   Cpp.TakesConstLValue(t);
  // CHECK:STDERR:       |                         ^
  // CHECK:STDERR: fail_call_param_const_lvalue_ref_with_wrong_type.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./param_const_lvalue_ref.h:184:6: note: candidate function not viable: no known conversion from 'T' to 'const S' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:   184 | auto TakesConstLValue(const S&) -> void;
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


// CHECK:STDOUT: --- call_param_lvalue_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
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
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc22.2: init %S to %s.var = class_init () [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc14: init %S = converted %.loc22.1, %.loc22.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   assign %s.var, %.loc14
// CHECK:STDOUT:   %.loc18: type = splice_block %S.ref [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = ref_binding s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesLValue.ref: %TakesLValue.cpp_overload_set.type = name_ref TakesLValue, imports.%TakesLValue.cpp_overload_set.value [concrete = constants.%TakesLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %.loc28: %S = ref_tag %s.ref
// CHECK:STDOUT:   %TakesLValue.call: init %empty_tuple.type = call imports.%TakesLValue.decl(%.loc28)
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
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.6c6: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.ae1: %T.as.DefaultOrUnformed.impl.Op.type.6c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.6b7: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table, @T.as.DefaultOrUnformed.impl(%S) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.d8f: %DefaultOrUnformed.type = facet_value %S, (%DefaultOrUnformed.impl_witness.6b7) [concrete]
// CHECK:STDOUT:   %TakesLValue.cpp_overload_set.type: type = cpp_overload_set_type @TakesLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesLValue.cpp_overload_set.value: %TakesLValue.cpp_overload_set.type = cpp_overload_set_value @TakesLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %T.e15: type = class_type @T [concrete]
// CHECK:STDOUT:   %pattern_type.e6b: type = pattern_type %T.e15 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.a11: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table, @T.as.DefaultOrUnformed.impl(%T.e15) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.5fe: %DefaultOrUnformed.type = facet_value %T.e15, (%DefaultOrUnformed.impl_witness.a11) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.01d: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.6c6) = import_ref Core//prelude/parts/default, loc34, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.ae1)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table = impl_witness_table (%Core.import_ref.01d), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %TakesLValue.cpp_overload_set.value: %TakesLValue.cpp_overload_set.type = cpp_overload_set_value @TakesLValue.cpp_overload_set [concrete = constants.%TakesLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %T.decl: type = class_decl @T [concrete = constants.%T.e15] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.7da = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.7da = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %S = var %v.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc20: %DefaultOrUnformed.type = facet_value constants.%S, (constants.%DefaultOrUnformed.impl_witness.6b7) [concrete = constants.%DefaultOrUnformed.facet.d8f]
// CHECK:STDOUT:   %.loc20.1: %DefaultOrUnformed.type = converted constants.%S, %DefaultOrUnformed.facet.loc20 [concrete = constants.%DefaultOrUnformed.facet.d8f]
// CHECK:STDOUT:   %as_type.loc20: type = facet_access_type %.loc20.1 [concrete = constants.%S]
// CHECK:STDOUT:   %.loc20.2: type = converted %.loc20.1, %as_type.loc20 [concrete = constants.%S]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc14: ref %S = splice_block %v.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc20: init %S to %.loc14 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.1()
// CHECK:STDOUT:   assign %v.var, %T.as.DefaultOrUnformed.impl.Op.call.loc20
// CHECK:STDOUT:   %.loc18: type = splice_block %S.ref.loc18 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc18: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %S = ref_binding v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = value_binding_pattern s [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.ref: ref %S = name_ref v, %v
// CHECK:STDOUT:   %.loc25: type = splice_block %S.ref.loc25 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc25: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc28: %S = acquire_value %v.ref
// CHECK:STDOUT:   %s: %S = value_binding s, %.loc28
// CHECK:STDOUT:   %Cpp.ref.loc30: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesLValue.ref.loc31: %TakesLValue.cpp_overload_set.type = name_ref TakesLValue, imports.%TakesLValue.cpp_overload_set.value [concrete = constants.%TakesLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref.loc34: %S = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesLValue.ref.loc38: %TakesLValue.cpp_overload_set.type = name_ref TakesLValue, imports.%TakesLValue.cpp_overload_set.value [concrete = constants.%TakesLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref.loc41: %S = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc44: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc45: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %const.loc43: type = const_type %S.ref.loc45 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc42.1: %const = as_compatible %s.ref.loc41
// CHECK:STDOUT:   %.loc42.2: %const = converted %s.ref.loc41, %.loc42.1
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type.e6b = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.var_patt: %pattern_type.e6b = var_pattern %t.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %T.e15 = var %t.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc55: %DefaultOrUnformed.type = facet_value constants.%T.e15, (constants.%DefaultOrUnformed.impl_witness.a11) [concrete = constants.%DefaultOrUnformed.facet.5fe]
// CHECK:STDOUT:   %.loc55.1: %DefaultOrUnformed.type = converted constants.%T.e15, %DefaultOrUnformed.facet.loc55 [concrete = constants.%DefaultOrUnformed.facet.5fe]
// CHECK:STDOUT:   %as_type.loc55: type = facet_access_type %.loc55.1 [concrete = constants.%T.e15]
// CHECK:STDOUT:   %.loc55.2: type = converted %.loc55.1, %as_type.loc55 [concrete = constants.%T.e15]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc49: ref %T.e15 = splice_block %t.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc55: init %T.e15 to %.loc49 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.2()
// CHECK:STDOUT:   assign %t.var, %T.as.DefaultOrUnformed.impl.Op.call.loc55
// CHECK:STDOUT:   %.loc53: type = splice_block %T.ref [concrete = constants.%T.e15] {
// CHECK:STDOUT:     %Cpp.ref.loc52: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %T.ref: type = name_ref T, imports.%T.decl [concrete = constants.%T.e15]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %T.e15 = ref_binding t, %t.var
// CHECK:STDOUT:   %Cpp.ref.loc56: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesLValue.ref.loc57: %TakesLValue.cpp_overload_set.type = name_ref TakesLValue, imports.%TakesLValue.cpp_overload_set.value [concrete = constants.%TakesLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %t.ref: ref %T.e15 = name_ref t, %t
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %u.patt: %pattern_type.7da = ref_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.var_patt: %pattern_type.7da = var_pattern %u.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %u.var: ref %S = var %u.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc69: %DefaultOrUnformed.type = facet_value constants.%S, (constants.%DefaultOrUnformed.impl_witness.6b7) [concrete = constants.%DefaultOrUnformed.facet.d8f]
// CHECK:STDOUT:   %.loc69.1: %DefaultOrUnformed.type = converted constants.%S, %DefaultOrUnformed.facet.loc69 [concrete = constants.%DefaultOrUnformed.facet.d8f]
// CHECK:STDOUT:   %as_type.loc69: type = facet_access_type %.loc69.1 [concrete = constants.%S]
// CHECK:STDOUT:   %.loc69.2: type = converted %.loc69.1, %as_type.loc69 [concrete = constants.%S]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc63: ref %S = splice_block %u.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc69: init %S to %.loc63 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.3()
// CHECK:STDOUT:   assign %u.var, %T.as.DefaultOrUnformed.impl.Op.call.loc69
// CHECK:STDOUT:   %.loc67: type = splice_block %S.ref.loc67 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc66: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc67: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %u: ref %S = ref_binding u, %u.var
// CHECK:STDOUT:   %Cpp.ref.loc70: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesLValue.ref.loc71: %TakesLValue.cpp_overload_set.type = name_ref TakesLValue, imports.%TakesLValue.cpp_overload_set.value [concrete = constants.%TakesLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %u.ref: ref %S = name_ref u, %u
// CHECK:STDOUT:   %Cpp.ref.loc77: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc78: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %const.loc76: type = const_type %S.ref.loc78 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc75.1: ref %const = as_compatible %u.ref
// CHECK:STDOUT:   %.loc75.2: ref %const = converted %u.ref, %.loc75.1
// CHECK:STDOUT:   %S.cpp_destructor.bound.loc63: <bound method> = bound_method %u.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call.loc63: init %empty_tuple.type = call %S.cpp_destructor.bound.loc63(%u.var)
// CHECK:STDOUT:   %T.cpp_destructor.bound: <bound method> = bound_method %t.var, constants.%T.cpp_destructor
// CHECK:STDOUT:   %T.cpp_destructor.call: init %empty_tuple.type = call %T.cpp_destructor.bound(%t.var)
// CHECK:STDOUT:   %S.cpp_destructor.bound.loc14: <bound method> = bound_method %v.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call.loc14: init %empty_tuple.type = call %S.cpp_destructor.bound.loc14(%v.var)
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
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %TakesRValue__carbon_thunk.type: type = fn_type @TakesRValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesRValue__carbon_thunk: %TakesRValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %S.cpp_destructor.type: type = fn_type @S.cpp_destructor [concrete]
// CHECK:STDOUT:   %S.cpp_destructor: %S.cpp_destructor.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesRValue.ref: %TakesRValue.cpp_overload_set.type = name_ref TakesRValue, imports.%TakesRValue.cpp_overload_set.value [concrete = constants.%TakesRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc19.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %.loc19.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc19.3: init %S to %.loc19.2 = class_init () [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc20.1: init %S = converted %.loc19.1, %.loc19.3 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc20.2: ref %S = temporary %.loc19.2, %.loc20.1
// CHECK:STDOUT:   %.loc20.3: %S = acquire_value %.loc20.2
// CHECK:STDOUT:   %.loc20.4: ref %S = value_as_ref %.loc20.3
// CHECK:STDOUT:   %addr: %ptr.5c7 = addr_of %.loc20.4
// CHECK:STDOUT:   %TakesRValue__carbon_thunk.call: init %empty_tuple.type = call imports.%TakesRValue__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %S.cpp_destructor.bound: <bound method> = bound_method %.loc20.2, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%.loc20.2)
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
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.6c6: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.ae1: %T.as.DefaultOrUnformed.impl.Op.type.6c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.6b7: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table, @T.as.DefaultOrUnformed.impl(%S) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.d8f: %DefaultOrUnformed.type = facet_value %S, (%DefaultOrUnformed.impl_witness.6b7) [concrete]
// CHECK:STDOUT:   %TakesRValue.cpp_overload_set.type: type = cpp_overload_set_type @TakesRValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesRValue.cpp_overload_set.value: %TakesRValue.cpp_overload_set.type = cpp_overload_set_value @TakesRValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %T.e15: type = class_type @T [concrete]
// CHECK:STDOUT:   %pattern_type.e6b: type = pattern_type %T.e15 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.a11: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table, @T.as.DefaultOrUnformed.impl(%T.e15) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.5fe: %DefaultOrUnformed.type = facet_value %T.e15, (%DefaultOrUnformed.impl_witness.a11) [concrete]
// CHECK:STDOUT:   %empty_struct.a40: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %empty_struct.6ef: %const = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.01d: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.6c6) = import_ref Core//prelude/parts/default, loc34, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.ae1)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table = impl_witness_table (%Core.import_ref.01d), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %TakesRValue.cpp_overload_set.value: %TakesRValue.cpp_overload_set.type = cpp_overload_set_value @TakesRValue.cpp_overload_set [concrete = constants.%TakesRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %T.decl: type = class_decl @T [concrete = constants.%T.e15] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc20: %DefaultOrUnformed.type = facet_value constants.%S, (constants.%DefaultOrUnformed.impl_witness.6b7) [concrete = constants.%DefaultOrUnformed.facet.d8f]
// CHECK:STDOUT:   %.loc20.1: %DefaultOrUnformed.type = converted constants.%S, %DefaultOrUnformed.facet.loc20 [concrete = constants.%DefaultOrUnformed.facet.d8f]
// CHECK:STDOUT:   %as_type.loc20: type = facet_access_type %.loc20.1 [concrete = constants.%S]
// CHECK:STDOUT:   %.loc20.2: type = converted %.loc20.1, %as_type.loc20 [concrete = constants.%S]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc14: ref %S = splice_block %s.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc20: init %S to %.loc14 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.1()
// CHECK:STDOUT:   assign %s.var, %T.as.DefaultOrUnformed.impl.Op.call.loc20
// CHECK:STDOUT:   %.loc18: type = splice_block %S.ref.loc18 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc18: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = ref_binding s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesRValue.ref.loc22: %TakesRValue.cpp_overload_set.type = name_ref TakesRValue, imports.%TakesRValue.cpp_overload_set.value [concrete = constants.%TakesRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type.e6b = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.var_patt: %pattern_type.e6b = var_pattern %t.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %T.e15 = var %t.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc34: %DefaultOrUnformed.type = facet_value constants.%T.e15, (constants.%DefaultOrUnformed.impl_witness.a11) [concrete = constants.%DefaultOrUnformed.facet.5fe]
// CHECK:STDOUT:   %.loc34.1: %DefaultOrUnformed.type = converted constants.%T.e15, %DefaultOrUnformed.facet.loc34 [concrete = constants.%DefaultOrUnformed.facet.5fe]
// CHECK:STDOUT:   %as_type.loc34: type = facet_access_type %.loc34.1 [concrete = constants.%T.e15]
// CHECK:STDOUT:   %.loc34.2: type = converted %.loc34.1, %as_type.loc34 [concrete = constants.%T.e15]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc28: ref %T.e15 = splice_block %t.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc34: init %T.e15 to %.loc28 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.2()
// CHECK:STDOUT:   assign %t.var, %T.as.DefaultOrUnformed.impl.Op.call.loc34
// CHECK:STDOUT:   %.loc32: type = splice_block %T.ref [concrete = constants.%T.e15] {
// CHECK:STDOUT:     %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %T.ref: type = name_ref T, imports.%T.decl [concrete = constants.%T.e15]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %T.e15 = ref_binding t, %t.var
// CHECK:STDOUT:   %Cpp.ref.loc35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesRValue.ref.loc36: %TakesRValue.cpp_overload_set.type = name_ref TakesRValue, imports.%TakesRValue.cpp_overload_set.value [concrete = constants.%TakesRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %t.ref: ref %T.e15 = name_ref t, %t
// CHECK:STDOUT:   %Cpp.ref.loc42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesRValue.ref.loc43: %TakesRValue.cpp_overload_set.type = name_ref TakesRValue, imports.%TakesRValue.cpp_overload_set.value [concrete = constants.%TakesRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc48.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct.a40]
// CHECK:STDOUT:   %Cpp.ref.loc50: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc51: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %.loc48.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc48.3: init %S to %.loc48.2 = class_init () [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc49: init %S = converted %.loc48.1, %.loc48.3 [concrete = constants.%S.val]
// CHECK:STDOUT:   %Cpp.ref.loc56: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc57: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %const: type = const_type %S.ref.loc57 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc54.1: init %const = as_compatible %.loc49 [concrete = constants.%empty_struct.6ef]
// CHECK:STDOUT:   %.loc54.2: init %const = converted %.loc49, %.loc54.1 [concrete = constants.%empty_struct.6ef]
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
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %TakesConstLValue.cpp_overload_set.type: type = cpp_overload_set_type @TakesConstLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesConstLValue.cpp_overload_set.value: %TakesConstLValue.cpp_overload_set.type = cpp_overload_set_value @TakesConstLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %ptr.ff5: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.type: type = fn_type @TakesConstLValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk: %TakesConstLValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
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
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc22.2: init %S to %s.var = class_init () [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc14: init %S = converted %.loc22.1, %.loc22.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   assign %s.var, %.loc14
// CHECK:STDOUT:   %.loc18: type = splice_block %S.ref.loc18 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc18: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = ref_binding s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstLValue.ref.loc25: %TakesConstLValue.cpp_overload_set.type = name_ref TakesConstLValue, imports.%TakesConstLValue.cpp_overload_set.value [concrete = constants.%TakesConstLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref.loc28: ref %S = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc32: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %const: type = const_type %S.ref.loc32 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc29.1: ref %const = as_compatible %s.ref.loc28
// CHECK:STDOUT:   %.loc29.2: ref %const = converted %s.ref.loc28, %.loc29.1
// CHECK:STDOUT:   %.loc29.3: ref %S = as_compatible %.loc29.2
// CHECK:STDOUT:   %.loc29.4: ref %S = converted %.loc29.2, %.loc29.3
// CHECK:STDOUT:   %.loc29.5: %S = acquire_value %.loc29.4
// CHECK:STDOUT:   %.loc29.6: ref %S = value_as_ref %.loc29.5
// CHECK:STDOUT:   %addr.loc34: %ptr.5c7 = addr_of %.loc29.6
// CHECK:STDOUT:   %.loc34.1: %ptr.ff5 = as_compatible %addr.loc34
// CHECK:STDOUT:   %.loc34.2: %ptr.ff5 = converted %addr.loc34, %.loc34.1
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.call.loc34: init %empty_tuple.type = call imports.%TakesConstLValue__carbon_thunk.decl(%.loc34.2)
// CHECK:STDOUT:   %Cpp.ref.loc36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstLValue.ref.loc37: %TakesConstLValue.cpp_overload_set.type = name_ref TakesConstLValue, imports.%TakesConstLValue.cpp_overload_set.value [concrete = constants.%TakesConstLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref.loc40: ref %S = name_ref s, %s
// CHECK:STDOUT:   %.loc40.1: %S = acquire_value %s.ref.loc40
// CHECK:STDOUT:   %.loc40.2: ref %S = value_as_ref %.loc40.1
// CHECK:STDOUT:   %addr.loc41: %ptr.5c7 = addr_of %.loc40.2
// CHECK:STDOUT:   %.loc41.1: %ptr.ff5 = as_compatible %addr.loc41
// CHECK:STDOUT:   %.loc41.2: %ptr.ff5 = converted %addr.loc41, %.loc41.1
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.call.loc41: init %empty_tuple.type = call imports.%TakesConstLValue__carbon_thunk.decl(%.loc41.2)
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
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %TakesConstLValue.cpp_overload_set.type: type = cpp_overload_set_type @TakesConstLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesConstLValue.cpp_overload_set.value: %TakesConstLValue.cpp_overload_set.type = cpp_overload_set_value @TakesConstLValue.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %ptr.ff5: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.type: type = fn_type @TakesConstLValue__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk: %TakesConstLValue__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
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
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = value_binding_pattern s [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc18: type = splice_block %S.ref.loc18 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc18: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc22.3: init %S to %.loc22.2 = class_init () [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc22.4: init %S = converted %.loc22.1, %.loc22.3 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc22.5: ref %S = temporary %.loc22.2, %.loc22.4
// CHECK:STDOUT:   %.loc22.6: %S = acquire_value %.loc22.5
// CHECK:STDOUT:   %s: %S = value_binding s, %.loc22.6
// CHECK:STDOUT:   %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstLValue.ref.loc25: %TakesConstLValue.cpp_overload_set.type = name_ref TakesConstLValue, imports.%TakesConstLValue.cpp_overload_set.value [concrete = constants.%TakesConstLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref.loc28: %S = name_ref s, %s
// CHECK:STDOUT:   %.loc28: ref %S = value_as_ref %s.ref.loc28
// CHECK:STDOUT:   %addr.loc29: %ptr.5c7 = addr_of %.loc28
// CHECK:STDOUT:   %.loc29.1: %ptr.ff5 = as_compatible %addr.loc29
// CHECK:STDOUT:   %.loc29.2: %ptr.ff5 = converted %addr.loc29, %.loc29.1
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.call.loc29: init %empty_tuple.type = call imports.%TakesConstLValue__carbon_thunk.decl(%.loc29.2)
// CHECK:STDOUT:   %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstLValue.ref.loc32: %TakesConstLValue.cpp_overload_set.type = name_ref TakesConstLValue, imports.%TakesConstLValue.cpp_overload_set.value [concrete = constants.%TakesConstLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref.loc35: %S = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc39: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %const: type = const_type %S.ref.loc39 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc36.1: %const = as_compatible %s.ref.loc35
// CHECK:STDOUT:   %.loc36.2: %const = converted %s.ref.loc35, %.loc36.1
// CHECK:STDOUT:   %.loc36.3: %S = as_compatible %.loc36.2
// CHECK:STDOUT:   %.loc36.4: %S = converted %.loc36.2, %.loc36.3
// CHECK:STDOUT:   %.loc36.5: ref %S = value_as_ref %.loc36.4
// CHECK:STDOUT:   %addr.loc41: %ptr.5c7 = addr_of %.loc36.5
// CHECK:STDOUT:   %.loc41.1: %ptr.ff5 = as_compatible %addr.loc41
// CHECK:STDOUT:   %.loc41.2: %ptr.ff5 = converted %addr.loc41, %.loc41.1
// CHECK:STDOUT:   %TakesConstLValue__carbon_thunk.call.loc41: init %empty_tuple.type = call imports.%TakesConstLValue__carbon_thunk.decl(%.loc41.2)
// CHECK:STDOUT:   %S.cpp_destructor.bound: <bound method> = bound_method %.loc22.5, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%.loc22.5)
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
// CHECK:STDOUT:   %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ReturnsLValue.ref: %ReturnsLValue.cpp_overload_set.type = name_ref ReturnsLValue, imports.%ReturnsLValue.cpp_overload_set.value [concrete = constants.%ReturnsLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %ReturnsLValue.call: ref %S = call imports.%ReturnsLValue.decl()
// CHECK:STDOUT:   %.loc20: type = splice_block %S.ref [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
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
// CHECK:STDOUT:   %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ReturnsRValue.ref: %ReturnsRValue.cpp_overload_set.type = name_ref ReturnsRValue, imports.%ReturnsRValue.cpp_overload_set.value [concrete = constants.%ReturnsRValue.cpp_overload_set.value]
// CHECK:STDOUT:   %ReturnsRValue.call: ref %S = call imports.%ReturnsRValue.decl()
// CHECK:STDOUT:   %.loc20: type = splice_block %S.ref [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
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
// CHECK:STDOUT:   %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ReturnConstLValue.ref: %ReturnConstLValue.cpp_overload_set.type = name_ref ReturnConstLValue, imports.%ReturnConstLValue.cpp_overload_set.value [concrete = constants.%ReturnConstLValue.cpp_overload_set.value]
// CHECK:STDOUT:   %ReturnConstLValue.call: ref %const = call imports.%ReturnConstLValue.decl()
// CHECK:STDOUT:   %.loc19: type = splice_block %const [concrete = constants.%const] {
// CHECK:STDOUT:     %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %const: type = const_type %S.ref [concrete = constants.%const]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %const = ref_binding s, %ReturnConstLValue.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
